# QKNorm + RoPE Fusion: Design Notes and Follow-up

## What we shipped

### Fusion heuristic (`cat` in `lowering.py`)
When all cat inputs are unrealized Pointwise nodes reading from the same buffers,
with equal cat-dim sizes, and the cat output matches the upstream reduction's
iteration space, use pointwise cat. This bypasses the `can_fuse_reduction` guard
and lets the scheduler fuse qknorm + rope into a single kernel (2 kernels → 1).

Works for both half-split RoPE (`cat([out1, out2], dim=-1)`) and interleaved RoPE
(`stack([real, imag], dim=-1).flatten()`).

### ModularIndexing in `pointwise_cat` (`lowering.py`)
When cat inputs share reads, use `ModularIndexing(idx - offset, 1, size)` instead
of `ops.masked(mask, body, 0.0)`. Because `ModularIndexing(r0 - 32, 1, 32)`
simplifies to `r0 % 32` — the same expression as `ModularIndexing(r0, 1, 32)` —
both cat branches produce identical index expressions. CSE deduplicates all shared
loads and computation (rsqrt, normalization, weight multiplication). Only the final
rope operations (mul-sub vs mul-add) differ.

**Result**: 13 loads → 7, 2 rsqrt → 1, 12 masked loads → 0, 3 tl.where → 1.

### `ops.masked` short-circuit (`triton.py`)
When the mask's bounds are statically True or False, `ops.masked` short-circuits:
returns `body()` directly for always-True, returns `constant(other)` for
always-False. Currently a no-op (bounds don't resolve with full-range iteration)
but becomes active once sub-range iteration is available.

## Benchmark results

### vs 2-kernel baseline (no fusion)

| Config | 2-kernel | Fused + ModIdx | Speedup |
|--------|--------:|--------------:|--------:|
| B=4,H=8,S=128,D=64 | 145 us | 108 us | 1.34x |
| B=4,H=32,S=512,D=128 | 164 us | 138 us | 1.19x |
| B=4,H=32,S=2048,D=128 | 339 us | 260 us | 1.31x |
| B=8,H=64,S=4096,D=128 | 1985 us | 1408 us | 1.41x |

### vs sglang/TRT-LLM hand-written CUDA kernel (CUDA graphs, Q+K)

| num_tokens | sglang CUDA | inductor | |
|----------:|----------:|---------:|-|
| 128 | 16.0 us | 19.1 us | sglang wins at small sizes |
| 1024 | 38.7 us | **34.8 us** | **inductor 1.11x** |
| 4096 | 122.4 us | **83.8 us** | **inductor 1.46x** |

### vs hand-optimized Triton (two-section, autotuned)

| Config | Inductor | Hand-optimized | Gap |
|--------|--------:|--------------:|----:|
| small | 113 us | 53 us | 2.1x |
| medium | 138 us | 52 us | 2.7x |
| large | 262 us | 163 us | 1.6x |
| xlarge | 1414 us | 1266 us | 1.1x |

The hand-optimized kernel uses two separate `tl.arange(0, HALF)` sections —
no masks, no wasted computation. This is the target for the sub-range codegen.

## Approaches tried and lessons learned

| Approach | Where | Result |
|----------|-------|--------|
| Clamped indices (sympy.Max/Min) | pointwise_cat | Correct but horrible nested ternaries in generated code, no perf win |
| CSE augment_key skip | codegen | Deduplicated rsqrt (2→1) but didn't help dependent loads, marginal perf |
| Safe-index loads (tl.where on index) | codegen | Removed all masked loads but tl.where+zeros_like overhead ate the savings |
| Skip mask for independent loads | codegen | Helped rsqrt CSE, marginal perf alone |
| ModularIndexing | pointwise_cat | 13→7 loads, 1 rsqrt, 10-16% speedup. **Shipped.** |
| ops.masked short-circuit (bounds) | codegen | Foundation for sub-range work. No-op until ranges are narrowed. **Shipped.** |

Key insight: the codegen can't normalize index expressions without semantic
knowledge. `Identity(r0 - 32)` and `Identity(r0)` look different to the codegen
but are equivalent mod 32. ModularIndexing makes this explicit at the IR level.

The general codegen solution requires sub-range iteration infrastructure rather
than trying to undo masks after the fact.

## Next step: sub-range codegen via nested reduction infrastructure

### The plan

The `nested_red_fusion_tmp` branch adds `create_sub_range` and
`_IterationRangeContext` — infrastructure for creating mid-kernel iteration
variables at a different resolution than the main loop.

Once that lands, the codegen can detect the masked cat pattern and apply sub-range
splitting:

1. Codegen detects `ops.masked(lt(iter_var, CONST), body, 0.0)` + `ops.where`
   in the inner_fn — a deterministic index-based split.

2. Uses `create_sub_range(r_tree, HALF)` to create `h = tl.arange(0, HALF)`.

3. Evaluates each masked body with `h` as the dim index. Since `h` ranges over
   `[0, HALF)`, bounds resolve and `ops.masked` short-circuits (always-True for
   the matching branch, always-False for the other).

4. Emits separate stores for each half.

### Why pointwise_cat stays unchanged

- Pointwise cat creates a fusible Pointwise node — downstream ops (fp8 cast,
  etc.) can fuse through it. ConcatKernel is a NOP and breaks fusion chains.
- The optimization is purely in HOW the codegen emits the masked inner_fn:
  instead of one full-range masked iteration, two clean sub-range iterations.
- No changes to pointwise_cat, the scheduler, or the IR — just smarter codegen.

### What the generated code would look like

```python
# Current (masked):
r0_1 = tl.arange(0, 64)
q = tl.load(in_ptr0 + r0_1 + 64*x0)
sum = tl.sum(q * q, 1)[:, None]
rsqrt = libdevice.rsqrt(sum / 64.0 + 1e-6)
mask = r0_1 < 32
val1 = tl.load(ptr + r0_1, mask, other=0.0)     # masked
val1_n = val1 * rsqrt                            # wasted for [32,64)
...                                               # duplicate rsqrt
result = tl.where(mask, branch1, branch2)
tl.store(out + r0_1, result)

# Target (sub-range):
r0_1 = tl.arange(0, 64)
q = tl.load(in_ptr0 + r0_1 + 64*x0)
sum = tl.sum(q * q, 1)[:, None]
rsqrt = libdevice.rsqrt(sum / 64.0 + 1e-6)
h = tl.arange(0, 32)                             # sub-range
x1 = tl.load(ptr + h) * rsqrt * w1               # clean, no mask
x2 = tl.load(ptr + 32 + h) * rsqrt * w2          # clean, no mask
cos = tl.load(cos_ptr + h)
sin = tl.load(sin_ptr + h)
tl.store(out + h, x1*cos - x2*sin)               # direct store
tl.store(out + 32 + h, x2*cos + x1*sin)          # direct store
```
