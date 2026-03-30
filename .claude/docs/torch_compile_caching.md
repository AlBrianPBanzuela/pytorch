# Compilation Caching in PyTorch and vLLM, and How Third-Party Backends Can Integrate

## Overview

PyTorch's `torch.compile` stack has a multi-layered caching system that avoids
redundant compilation. This document describes each caching layer from bottom to
top, how they compose, and how third-party backends and serving frameworks
integrate with them.

```
┌──────────────────────────────────────────────────────────────┐
│  Layer 4: Frame-level AOT cache (PyTorch + vLLM)             │
│  Entire torch.compile'd function: bytecode + guards +        │
│  FX graph + all compiled artifacts                           │
├──────────────────────────────────────────────────────────────┤
│  Layer 3: PiecewiseBackend (vLLM)                            │
│  Shape-bucket compilation + graph splitting + disk cache     │
├──────────────────────────────────────────────────────────────┤
│  Layer 2: AOTAutogradCache                                   │
│  Bundles OutputCode + ViewAndMutationMeta                    │
├──────────────────────────────────────────────────────────────┤
│  Layer 1: FxGraphCache                                       │
│  Inductor-level compiled graph caching                       │
├──────────────────────────────────────────────────────────────┤
│  Backend compilation (Triton codegen, PjRt, Metal, etc.)     │
└──────────────────────────────────────────────────────────────┘
```

Layers 1 and 2 live in PyTorch, layer 3 lives in vLLM, and layer 4 spans both. A
third-party backend can plug in at whichever layer matches its architecture:

- **New Inductor codegen backend** (e.g., Metal): plugs into Layer 1 via
  `register_backend_for_device`. Gets FxGraphCache for free.
- **Non-Inductor backend** (e.g., torch_tpu): plugs into Layer 2 by
  subclassing `OutputCode` and using `SerializableAOTDispatchCompiler`.
- **LLM serving integration**: plugs into Layer 3 by implementing
  `CompilerInterface`, the pluggable backend within `PiecewiseBackend`.
- **Frame-level AOT caching**: Layer 4 serializes the entire compiled function
  so warm starts skip Dynamo tracing entirely.

---

## Layer 1: FxGraphCache

**Files:** `torch/_inductor/codecache.py`, `torch/_inductor/output_code.py`

FxGraphCache caches the product of Inductor compilation: a `CompiledFxGraph`
object (an `OutputCode` subclass) that wraps generated Python source code,
Triton kernel bundles, and all metadata needed to reconstruct the callable.

### Cache key computation

`FxGraphHashDetails` captures all state that affects compilation output:

- The FX GraphModule (serialized via `FxGraphCachePickler`)
- Example input metadata (shape, dtype, stride, device)
- Inductor config snapshot
- Torch version and system info (Triton compiler version, etc.)
- Custom pass UUIDs (`CustomGraphPass.uuid()`)
- Custom backend pass UUIDs and codegen configs (from
  `register_backend_for_device`)
- Deterministic algorithm and CUDA matmul settings
- Shape hint overrides

The entire `FxGraphHashDetails` object is pickled with a custom pickler (that
handles FakeTensors, SymInts, etc.) and SHA-256 hashed to produce the cache key.

### CompiledFxGraph — the cached artifact

`CompiledFxGraph` (`output_code.py:425`) is the `OutputCode` subclass that
FxGraphCache serializes and deserializes. It contains everything needed to
reconstruct a compiled callable from disk:

- `source_code: str` — the generated Python wrapper code
- `cache_key: str` — PyCodeCache key for the wrapper
- `guards_expr: Optional[str]` — symbolic shape guard expression (see Load
  flow below for how this is used)
- `_triton_bundle` — serialized Triton kernels
- `current_callable` — the live compiled function (nulled during serialization,
  reconstructed on load)
- Graph metadata: `constants`, `device_types`, `mutated_inputs`,
  `output_strides`, `cudagraph_info`, etc.

Each file in the cache directory is a self-contained pickle of a
`CompiledFxGraph`. The `guards_expr` is embedded in the pickle alongside all
other fields — there is no separate mapping file.

### Save/load flow

**Save (`FxGraphCache._save_graph`):**
1. Compute `guards_expr` from `ShapeEnv` (pruned to only reference the graph's
   SymInt inputs)
2. Copy the `CompiledFxGraph`, call `prepare_for_serialization()` to null out
   non-serializable state (the live callable, C++ function pointers)
3. `pickle.dumps()` the prepared object
4. Write to local disk and/or remote cache

**Load (`FxGraphCache.load_with_key`):**
1. Look up all cache entries for the key
2. Evaluate each entry's `guards_expr` against current shape hints. A single
   cache key can map to multiple compiled versions for different symbolic
   shape constraints; `guards_expr` distinguishes between them. The cache
   `eval`s each entry's expression against the current concrete shape hints.
   The first entry whose guards pass is a cache hit.
3. On guard match, unpickle the `CompiledFxGraph`
4. Call `post_compile()` to reconstruct the live callable (writes source code
   to disk, loads it via `PyCodeCache`, re-initializes CUDAGraph wrappers, etc.)

### Integrating a new Inductor codegen backend

Backends that register via `register_backend_for_device()` produce
`CompiledFxGraph` objects through Inductor's standard codegen pipeline. They
get FxGraphCache **for free** — no additional caching work is needed. The
backend's custom pass and codegen config are automatically included in the
cache key via `custom_backend_passes` and `custom_backend_codegen_configs` in
`FxGraphHashDetails`.

If the backend registers a `CustomGraphModulePass`, it must implement `uuid()`
to return a stable identifier for cache key computation (or `None` to disable
caching).

---

## Layer 2: AOTAutogradCache

**Files:** `torch/_functorch/_aot_autograd/`

AOTAutogradCache sits above FxGraphCache and caches the full aot_autograd
artifact: the compiled forward/backward graphs **plus** the
`ViewAndMutationMeta` that describes how aot_autograd handles input mutations
and views. On a cache hit at this layer, neither aot_autograd tracing nor
backend compilation needs to run.

### AOTAutogradCacheDetails - cache key

`AOTAutogradCacheDetails` (`autograd_cache.py:397`) **extends**
`FxGraphHashDetails` (Layer 1's key class), so it includes everything Layer 1
hashes plus additional aot_autograd-specific state:

- Everything from `FxGraphHashDetails` (FX graph, input metadata, inductor
  config, torch version, custom pass UUIDs, etc.)
- `aot_config` — aot_autograd configuration (num_params_buffers,
  keep_inference_input_mutations, dynamic_shapes, pre_dispatch, etc.)
- `grad_enabled` — whether autograd gradient computation is enabled
- `disable_amp` — whether autocast is active
- `deterministic_algorithms`
- `autograd_config` — functorch config snapshot
- `triton_kernel_source_codes` — source of user-defined Triton kernels
- `custom_estimator_solver_uuids` — UUIDs for custom runtime estimators and
  knapsack solvers (for activation checkpointing)

The key is computed by `autograd_cache_key()`: it pickles the
`AOTAutogradCacheDetails` with `AOTAutogradCachePickler` (extends
`FxGraphCachePickler` with reducers for `AOTConfig` and `Tensor`) and SHA-256
hashes the result, prefixed with `"a"` (vs `"f"` for Layer 1 keys).

Placeholder names are normalized before hashing so that isomorphic graphs with
different variable names produce the same key.

### BundledAOTAutogradResult - cache value

When `functorch_config.bundled_autograd_cache = True`, aot_autograd attaches a
`.serialize()` method to the compiled function. Calling it produces a
`BundledAOTAutogradResult` containing:

**Compiled artifacts (the actual `OutputCode`):**
- `compiled_fw: BundledCompiledForward[TOutputCode]` — wraps the forward
  `OutputCode` (e.g., `CompiledFxGraph` or `_TorchTpuCompiledExecutable`)
- `compiled_bw: Optional[BundledCompiledBackward[TOutputCode]]` — wraps the
  backward `OutputCode` (None for inference-only)

**Metadata for reconstructing the runtime wrapper chain:**
- `runtime_metadata: ViewAndMutationMeta` — which inputs are mutated/views
- `dispatch_wrappers: list[CompilerWrapper]` — wrapper classes to re-apply
- `maybe_subclass_meta: Optional[SubclassMeta]` — tensor subclass dispatch info
- `num_fw_outs_saved_for_bw: Optional[int]` — forward outputs needed by backward
- `indices_of_inps_to_detach: list[int]` — which inputs to detach from autograd
- `sanitized_aot_config: AOTConfig` — the aot_autograd config
- `guards_expr: Optional[str]` — shape guards (same concept as Layer 1)

The runtime wrappers themselves are **not pickled as callables**. Only their
configuration is stored. On cache load, `wrap_post_compile()` re-instantiates
the wrapper chain:

1. **Load the `OutputCode`** — for the bundled path, the deserialized
   `OutputCode` is already available; calls `OutputCode.post_compile()` to
   reconstruct the live callable (e.g., for `CompiledFxGraph` this writes
   source code to disk and loads it via `PyCodeCache`)
2. **`AOTDispatchSubclassWrapper`** — wraps for tensor subclass dispatch
3. **`FunctionalizedRngRuntimeWrapper`** — wraps for RNG functionalization
4. **`AOTDispatchAutograd.post_compile`** (training) or
   **`RuntimeWrapper.post_compile`** (inference) — builds the
   `autograd.Function` connecting forward to backward, handles input
   detaching and amp disabling
5. **`dispatch_wrappers`** — applies any additional wrappers

The result is a callable identical to what aot_autograd would have produced on
a fresh compilation, but without any tracing or backend compilation.

### Integrating a non-Inductor backend (e.g., torch_tpu)

Backends that replace Inductor entirely — they receive an FX graph
post-aot_autograd and compile it with their own compiler — plug in at this
layer. The compiled artifact is stored inside the `BundledAOTAutogradResult`
as an `OutputCode` subclass.

`OutputCode` (`torch/_inductor/output_code.py`) is the base class representing
a compiled callable:

```python
@dataclasses.dataclass
class OutputCode:
    _fx_graph_cache_key: Optional[str]
    _time_taken_ns: Optional[int]

    def __call__(self, inputs: Sequence[Any]) -> Any:
        """Execute the compiled code."""

    def prepare_for_serialization(self) -> None:
        """Strip non-serializable state before pickling."""

    def post_compile(self, example_inputs, constants, graph_kwargs) -> None:
        """Reconstruct live callable after deserialization."""

    def set_triton_bundle(self, triton_bundle: Any) -> None:
        """Attach Triton kernel bundle (no-op for non-Triton backends)."""
```

Built-in subclasses include `CompiledFxGraph` (Layer 1's default Inductor
path), `CompiledAOTI` (AOTInductor `.so` binaries), and `RegionalOutputCode`
(regional inductor via `GraphPickler`).

To integrate a new backend:

1. **Subclass `OutputCode`** — implement `__call__` to execute the compiled
   program. The FxGraphCache-specific methods (`prepare_for_serialization`,
   `post_compile`, `set_triton_bundle`) can be no-ops since these backends
   bypass Layer 1.

2. **Make it picklable** — the `BundledAOTAutogradResult` is pickled as a
   whole, so the `OutputCode` inside it must be picklable. Inductor's
   `CompiledFxGraph` doesn't need special handling because all its fields are
   natively picklable (strings, ints, bytes) — `prepare_for_serialization()`
   just nulls out the one non-picklable field (`current_callable`) and
   `post_compile()` reconstructs it on load. But a non-Inductor backend
   typically holds an opaque native object (e.g., a PjRt `LoadedExecutable`)
   that standard pickle can't handle. In that case, implement `__reduce__` to
   convert the native object to/from bytes.

3. **Use `SerializableAOTDispatchCompiler`** — pass
   `output_code_ty=YourOutputCode` so aot_autograd wraps the compiled output
   in a `BundledAOTAutogradResult` parameterized by your type.

4. **Enable bundled caching** — set
   `functorch_config.bundled_autograd_cache = True`.

**Example: torch_tpu**

The `OutputCode` subclass serializes the PjRt executable to bytes via
`__reduce__`:

```python
class _TorchTpuCompiledExecutable(OutputCode):
    _boxed_call = True

    def __call__(self, args):
        tensor_args = self._itemgetter(args)
        return tpu_torch_compile.execute(self._executable, tensor_args)

    # FxGraphCache interface — all no-ops (bypasses Layer 1)
    def prepare_for_serialization(self):
      pass
    def post_compile(self, example_inputs, constants, graph_kwargs):
      pass
    def set_triton_bundle(self, triton_bundle):
      pass

    # Custom pickle via PjRt's native serialization
    def __reduce__(self):
        serialized = tpu_torch_compile.serialize_executable(self._executable)
        return (_unpickle_compiled_executable, (serialized, self._map_output_fn))
```

The backend wires it up via `SerializableAOTDispatchCompiler`:

```python
class TpuBackend:
    def __call__(self, gm, example_inputs):
        compiler = SerializableAOTDispatchCompiler(
            fw_compiler=self._fw_compiler,
            bw_compiler=self._bw_compiler,
            output_code_ty=_TorchTpuCompiledExecutable,  # ← key line
        )
        return aot_autograd(gm, example_inputs, compiler=compiler)
```

---

## Layer 3: PiecewiseBackend (vLLM)

**Files:** `vllm/compilation/backends.py`, `vllm/compilation/piecewise_backend.py`,
`vllm/compilation/compiler_interface.py`

### Multi-shape-bucket compilation

LLM serving sees varying batch sizes at runtime. To produce more optimal
compilations, vLLM pre-compiles each graph for a fixed set of shape buckets
(e.g., batch sizes 1, 2, 4, 8, 16, ..., max). At runtime,
`PiecewiseBackend.__call__` pads the input to the nearest bucket size and
dispatches to the pre-compiled version.

This also saves Dynamo compilation time. Normally, each distinct input shape
would trigger a full Dynamo retrace + backend compilation. Instead, vLLM
traces the model through Dynamo **once** with full dynamic shapes, captures the
FX graph, drops the Dynamo guards (`TorchCompileWithNoGuardsWrapper`), and then
specializes that saved graph for each shape bucket by only running backend
compilation (via `CompilerInterface`).  Dynamo tracing happens once; only the
backend compile is repeated per bucket.

### Deduplication

vLLM detects when repeated transformer layers produce identical FX graphs
(same aot_autograd cache key). On the first occurrence, it compiles and caches.
On subsequent identical layers, it reuses the in-memory artifact, avoiding
redundant compilation.

### Graph splitting

vLLM can optionally split the Dynamo FX graph at "splitting ops" (typically
`vllm.unified_attention`). This separates the model into compilable subgraphs
(matrix multiplications, norms, activations) and non-compilable ops (custom
attention kernels). Each compilable subgraph becomes a separate
`PiecewiseBackend` instance.

Graph splitting exists primarily to enable **piecewise CUDA graph capture** —
CUDA graphs cannot capture across custom attention kernels, so the graph must
be split so that each compilable region can be captured independently. On
platforms like TPU where CUDA graphs don't apply, splitting is disabled
(`splitting_ops = []`) and the entire graph is compiled as a single unit.

```
torch.compile(model, backend=VllmBackend)
    │
    ▼
VllmBackend.__call__(fx_graph)
    │
    ├─ (optional) split_graph() at attention ops
    │
    ▼
PiecewiseCompileInterpreter
    │
    ├─ For each compilable subgraph:
    │   │
    │   ▼
    │   PiecewiseBackend
    │       │
    │       ├─ For each shape bucket:
    │       │   │
    │       │   ▼
    │       │   CompilerInterface.compile()  ← backend-specific
    │       │
    │       └─ Runtime: dispatch by batch size to pre-compiled version
    │
    └─ Non-compilable ops run eagerly (e.g., custom attention kernels)
```

### Caching

After compilation, all subgraph artifacts are saved to a cache directory on
disk. vLLM computes a top-level hash from environment factors, `vllm_config`,
source code of vLLM compilation modules, and
`CompilerInterface.compute_hash()` (backend version). This hash selects the
cache directory — it acts as a coarse-grained invalidation namespace. If any
factor changes, the hash changes, a new directory is created, and everything is
recompiled.

Inside the directory, a `vllm_compile_cache.py` file maps each compiled
subgraph to its artifact on disk:

- **Key:** `(compile_range, graph_index, compiler_name)` — identifies which
  subgraph at which shape bucket, compiled by which backend
- **Value:** `{"graph_handle": handle, "cache_key": cache_key}` — `handle` is
  an opaque value passed to `CompilerInterface.load()` to retrieve the
  compiled artifact (e.g., a file path); `cache_key` is the aot_autograd
  content hash used for in-memory deduplication across identical layers

```
~/.cache/vllm/torch_compile_cache/{hash}/rank_{i}_{j}/
    vllm_compile_cache.py          ← the key→value mapping above
    <artifact files>               ← loaded by CompilerInterface.load(handle)
```

### Integrating a new backend with vLLM

`PiecewiseBackend` delegates the actual compilation and cache I/O to a
`CompilerInterface` implementation. This is how new hardware backends plug into
vLLM's compilation pipeline.

```python
class CompilerInterface:
    def compile(self, graph, example_inputs, compiler_config,
                compile_range, key) -> handle:
        """Compile a subgraph for a specific shape range.
        Returns an opaque handle for later loading."""

    def load(self, handle, graph, example_inputs,
             graph_index, compile_range) -> callable:
        """Load a previously compiled artifact from cache."""

    def compute_hash(self, vllm_config) -> str:
        """Return a hash for cache invalidation
        (e.g., backend version string)."""

    def initialize_cache(self, cache_dir, disable_cache, prefix):
        """Set up cache directory."""
```

New backends implement a `CompilerInterface` and register it via their vLLM
`Platform` class. The `compile()` method calls down to Layer 2 (aot_autograd +
backend compilation), then serializes the result. The `load()` method
deserializes it, skipping all compilation. `compute_hash()` provides a version
string for cache invalidation.

```python
class TpuCompilerAdaptor(CompilerInterface):
    def compile(self, graph, example_inputs, compiler_config,
                compile_range, key):
        # Compile via TpuBackend (Layer 2)
        compiled_fn = _tpu_backend(graph, example_inputs)
        # Serialize via BundledAOTAutogradResult
        entry = compiled_fn.serialize()
        pickle.dump(entry, open(f"{self._cache_dir}/{key}", "wb"))
        return (key, save_path)

    def load(self, handle, graph, example_inputs, graph_index,
             compile_range):
        entry = pickle.load(open(handle[1], "rb"))
        return deserialize_bundled_cache_entry(entry)

    def compute_hash(self, vllm_config):
        return hashlib.sha256(torch_tpu.__version__.encode()).hexdigest()
```

---

## Layer 4: Frame-Level AOT Cache

**PyTorch:** `torch/_dynamo/aot_compile.py` (`AOTCompiledFunction`)
**vLLM:** `vllm/compilation/caching.py` (`VllmSerializableFunction`),
`vllm/compilation/decorators.py` (`@support_torch_compile`)

Layers 1–3 all operate *below* Dynamo — they cache compiled subgraphs but
Dynamo must still trace the Python function on every process start. Layer 4
caches the entire Dynamo frame: the result of `torch.compile(model.forward)` as
a single serialized unit. On warm start, Dynamo tracing, graph splitting, and
all backend compilation are skipped entirely.

### What it serializes

`AOTCompiledFunction` bundles into a single pickle:

- **Dynamo bytecode** — the transformed Python bytecode (`runtime_env`)
- **Guard manager state** — serialized guards for input validation
- **The compiled function** — a `SerializableCallable` (e.g.,
  `VllmSerializableFunction`) containing the FX graph and all compiled
  backend artifacts
- **Function signature and source info** — for cache invalidation

vLLM's `VllmSerializableFunction` extends this with:
- The Dynamo FX graph (serialized via `GraphPickler`)
- Example inputs (as meta-device tensors to save space)
- All compiled Inductor artifacts from every `PiecewiseBackend`, gathered via
  `VllmBackend.collect_standalone_compile_artifacts()`

### Cache key

vLLM hashes `(vllm_config, model_hash_key, aot_compile_hash_factors)`
to produce a directory, and stores the entire artifact as a single file:

```
~/.cache/vllm/torch_compile_cache/torch_aot_compile/{hash}/rank_{i}_{j}/model
```

On load, vLLM checks that all Python source files Dynamo traced through haven't
been modified since the artifact was saved. If any file changed, the cached
artifact is rejected and recompilation happens.

---

## End-to-End Example

This traces the full path through all layers using torchtpu-vllm as a
concrete example.

### Cold start (no cache)

1. vLLM loads model weights onto TPU.
2. Dynamo traces `model.forward` once, producing an FX graph.
3. `capture_model()` pre-compiles all subgraphs for every shape bucket.
4. For each bucket:
   - **Layer 3** (`PiecewiseBackend`): selects the subgraph and shape bucket
   - **Layer 3** (`TpuCompilerAdaptor.compile()`): delegates to `TpuBackend`
   - **Layer 2** (`AOTAutogradCache`): aot_autograd traces the graph, calls
     `TpuBackend`'s forward compiler
   - **Backend**: FX-to-MLIR lowering → PjRt compilation →
     `_TorchTpuCompiledExecutable`
   - **Layer 2**: `compiled_fn.serialize()` produces
     `BundledAOTAutogradResult`
   - **Layer 3**: pickles the result to `cache_dir/key`
5. **Layer 4**: if AOT compile is enabled, serializes the entire compiled
   function (Dynamo bytecode + guards + all artifacts) to disk.

### Warm start (Layer 3 cache hit, no Layer 4)

1. Dynamo re-traces the model (unavoidable without Layer 4).
2. For each bucket:
   - **Layer 3** (`TpuCompilerAdaptor.load()`): unpickles
     `BundledAOTAutogradResult` from disk
   - **Layer 2**: `deserialize_bundled_cache_entry()` reconstructs the callable
     from the pickled `_TorchTpuCompiledExecutable` (loads the PjRt executable
     from serialized bytes)
   - No backend compilation happens.

### Warm start (Layer 4 cache hit)

1. **Layer 4**: loads the entire AOT artifact from a single file.
2. Dynamo tracing, graph splitting, and all backend compilation are skipped.
3. `PiecewiseBackend` instances are reconstructed with pre-compiled runnables.

### Runtime (inference)

1. **Layer 3**: pads input to nearest bucket, dispatches to pre-compiled
   callable.
2. The callable executes the PjRt program directly on TPU.
