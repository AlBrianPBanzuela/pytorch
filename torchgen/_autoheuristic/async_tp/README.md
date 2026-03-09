# AutoHeuristic for Async TP Fuse Decision

Learned heuristic for async TP fuse/no_fuse decision: whether to fuse
`all_gather` with `matmul` in the micro-pipeline TP pass (`micro_pipeline_tp.py`).

Replaces the hardcoded `K_shard < 1024` guard with a shape-aware decision tree
trained on multi-model benchmark data (Llama, Mixtral).

## Quick Start: Re-generate heuristic from existing data

```bash
# Train decision tree and generate artifact
bash gen_async_tp_heuristic_h100.sh
```

This will overwrite `torch/_inductor/autoheuristic/artifacts/_AsyncTPFuseH100.py`.

## Collecting new benchmark data

```bash
cd custom_op_autotuning/autoheuristic_poc
torchrun --nproc_per_node=8 run_full_benchmark.py
```

Then convert and retrain:

```bash
python convert_benchmark_data.py --output async_tp_h100_data.txt
bash gen_async_tp_heuristic_h100.sh
```

## For other GPUs (e.g., A100)

1. Run benchmarks on the target GPU
2. Convert with correct device info:
   ```bash
   python convert_benchmark_data.py \
       --output async_tp_a100_data.txt \
       --shared-memory 166912 \
       --device-capa 8,0
   ```
3. Generate heuristic:
   ```bash
   bash gen_async_tp_heuristic_a100.sh
   ```

## Files

| File | Purpose |
|---|---|
| `convert_benchmark_data.py` | Convert benchmark JSON → AutoHeuristic txt format |
| `train_decision_async_tp.py` | Training script (inherits `AHTrainDecisionTree`) |
| `async_tp_h100_data.txt` | H100 training data (40 configs × 2 choices = 80 rows) |
| `gen_async_tp_heuristic_h100.sh` | One-liner to generate H100 artifact |
| `gen_async_tp_heuristic_a100.sh` | One-liner to generate A100 artifact |

## Features used by the decision tree

| Feature | Description |
|---|---|
| `m` | Batch/sequence dimension (M) |
| `k` | Contraction dimension (K_shard, per-rank) |
| `n` | Output dimension (N) |
| `arith_intensity` | M×K×N / (M×K + K×N + M×N) — compute-to-memory ratio |
| `m_times_k` | M × K |
| `m_times_n` | M × N — output matrix size |
| `k_times_n` | K × N |

## How the trained heuristic is used at runtime

The generated artifact (`_AsyncTPFuseH100.py`) is a `LearnedHeuristicDecision`
subclass that:

1. Checks preconditions (device capability, shared memory) via `check_precondition()`
2. Takes shape features via `AHContext`
3. Returns ranked choices via `get_best_choices()`, or `None` if unsure (unsafe leaf)
4. When `None`, the caller falls back to no_fuse (safe default)

Integration point: `torch/_inductor/fx_passes/micro_pipeline_tp.py` →
`_should_fuse_async_tp()` replaces the old `K_shard < 1024` guard.

Enabled via: `TORCHINDUCTOR_AUTOHEURISTIC_USE=async_tp_fuse`
