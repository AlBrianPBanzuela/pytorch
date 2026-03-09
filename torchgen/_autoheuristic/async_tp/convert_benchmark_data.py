# mypy: ignore-errors
"""
Convert async TP benchmark JSON data to AutoHeuristic training format.

The AutoHeuristic training pipeline expects a .txt file where:
  - Line 1: JSON metadata (device_capa, shared_memory, name, features)
  - Line 2: CSV header
  - Lines 3+: CSV data rows with features + choice + feedback columns

Usage:
    python convert_benchmark_data.py \
        --ag-mm-json ../../custom_op_autotuning/autoheuristic_poc/data/ag_mm_benchmark_results.json \
        --ffn-json ../../custom_op_autotuning/autoheuristic_poc/data/ffn_benchmark_results.json \
        --output async_tp_h100_data.txt
"""

import argparse
import json
import os
import sys


MODEL_DIMS = {
    "Llama3-70B": {"hidden_size": 8192, "intermediate_size": 28672},
    "Llama3-405B": {"hidden_size": 16384, "intermediate_size": 53248},
    "Mixtral-8x7B": {"hidden_size": 4096, "intermediate_size": 14336},
    "DeepSeek-V2": {"hidden_size": 4096, "intermediate_size": 11008},
    "Qwen2.5-72B": {"hidden_size": 8192, "intermediate_size": 29568},
}

NUMERICAL_FEATURES = [
    "m",
    "k",
    "n",
    "arith_intensity",
    "m_times_k",
    "m_times_n",
    "k_times_n",
]

# H100 device info
H100_SHARED_MEMORY = 232448
H100_DEVICE_CAPA = [9, 0]


def compute_features(m, k, n):
    """Compute derived features for a given (M, K, N) shape."""
    m_times_k = m * k
    m_times_n = m * n
    k_times_n = k * n
    denom = m_times_k + k_times_n + m_times_n
    arith_intensity = round(m * k * n / denom, 4) if denom > 0 else 0
    return {
        "m": m,
        "k": k,
        "n": n,
        "arith_intensity": arith_intensity,
        "m_times_k": m_times_k,
        "m_times_n": m_times_n,
        "k_times_n": k_times_n,
    }


def load_ag_mm_data(json_path):
    """Load AG+MM benchmark JSON and convert to (features, choice, feedback) rows."""
    with open(json_path) as f:
        data = json.load(f)

    rows = []
    for model_name, entries in data.items():
        dims = MODEL_DIMS.get(model_name)
        if not dims:
            print(f"  WARNING: Unknown model {model_name}, skipping", file=sys.stderr)
            continue
        for entry in entries:
            m = entry["M"]
            k = entry.get("K", dims["intermediate_size"])
            n = entry.get("N", dims["hidden_size"])
            bl = entry["baseline_us"]
            fu = entry["fused_us"]
            if bl == float("inf") or fu == float("inf"):
                continue
            feats = compute_features(m, k, n)
            # Add both choices with their timings as feedback
            rows.append({**feats, "choice": "no_fuse", "feedback": bl})
            rows.append({**feats, "choice": "fuse", "feedback": fu})
    return rows


def load_ffn_data(json_path):
    """Load FFN benchmark JSON and convert to (features, choice, feedback) rows."""
    with open(json_path) as f:
        data = json.load(f)

    rows = []
    for model_name, entries in data.items():
        dims = MODEL_DIMS.get(model_name)
        if not dims:
            print(f"  WARNING: Unknown model {model_name}, skipping", file=sys.stderr)
            continue
        hidden = dims["hidden_size"]
        intermediate = dims["intermediate_size"]
        for entry in entries:
            m = entry.get("M_shard", entry.get("M_total", 0) // 8)
            k = hidden
            n = intermediate
            bl = entry["baseline_us"]
            fu = entry["fused_us"]
            if bl == float("inf") or fu == float("inf"):
                continue
            feats = compute_features(m, k, n)
            rows.append({**feats, "choice": "no_fuse", "feedback": bl})
            rows.append({**feats, "choice": "fuse", "feedback": fu})
    return rows


def deduplicate_rows(rows):
    """
    Deduplicate by (m, k, n, choice) — keep the row with the lowest feedback (best timing).
    AutoHeuristic expects one feedback value per (features, choice) combination.
    """
    seen = {}
    for row in rows:
        key = (row["m"], row["k"], row["n"], row["choice"])
        if key not in seen or row["feedback"] < seen[key]["feedback"]:
            seen[key] = row
    return list(seen.values())


def write_autoheuristic_txt(rows, output_path, shared_memory, device_capa):
    """
    Write data in AutoHeuristic format:
      Line 1: JSON metadata
      Line 2: CSV header
      Lines 3+: CSV data
    """
    metadata = {
        "shared_memory": shared_memory,
        "device_capa": device_capa,
        "name": "async_tp_fuse",
        "numerical_features": NUMERICAL_FEATURES,
        "categorical_features": [],
    }

    header_fields = NUMERICAL_FEATURES + ["choice", "feedback"]

    with open(output_path, "w") as f:
        f.write(json.dumps(metadata) + "\n")
        f.write(",".join(header_fields) + "\n")
        for row in rows:
            values = [str(row[field]) for field in header_fields]
            f.write(",".join(values) + "\n")

    # Count unique configs
    unique_configs = len({(r["m"], r["k"], r["n"]) for r in rows})
    print(f"Written: {output_path}")
    print(f"  {len(rows)} rows, {unique_configs} unique (M,K,N) configs")
    print(f"  Choices: {sorted(set(r['choice'] for r in rows))}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert async TP benchmark JSON to AutoHeuristic training format"
    )
    parser.add_argument(
        "--ag-mm-json",
        default=None,
        help="Path to AG+MM benchmark JSON",
    )
    parser.add_argument(
        "--ffn-json",
        default=None,
        help="Path to FFN benchmark JSON",
    )
    parser.add_argument(
        "--output",
        default="async_tp_h100_data.txt",
        help="Output .txt file in AutoHeuristic format",
    )
    parser.add_argument(
        "--shared-memory",
        type=int,
        default=H100_SHARED_MEMORY,
        help="GPU shared memory size (bytes)",
    )
    parser.add_argument(
        "--device-capa",
        type=str,
        default="9,0",
        help="Device capability, comma-separated (e.g., '9,0' for H100)",
    )
    args = parser.parse_args()

    device_capa = [int(x) for x in args.device_capa.split(",")]

    # Default paths
    poc_data_dir = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "custom_op_autotuning",
        "autoheuristic_poc",
        "data",
    )
    if args.ag_mm_json is None:
        args.ag_mm_json = os.path.join(poc_data_dir, "ag_mm_benchmark_results.json")
    if args.ffn_json is None:
        args.ffn_json = os.path.join(poc_data_dir, "ffn_benchmark_results.json")

    all_rows = []

    if os.path.exists(args.ag_mm_json):
        print(f"Loading AG+MM: {args.ag_mm_json}")
        ag_rows = load_ag_mm_data(args.ag_mm_json)
        all_rows.extend(ag_rows)
        print(f"  {len(ag_rows)} rows from AG+MM")
    else:
        print(f"  AG+MM JSON not found: {args.ag_mm_json}")

    if os.path.exists(args.ffn_json):
        print(f"Loading FFN: {args.ffn_json}")
        ffn_rows = load_ffn_data(args.ffn_json)
        all_rows.extend(ffn_rows)
        print(f"  {len(ffn_rows)} rows from FFN")
    else:
        print(f"  FFN JSON not found: {args.ffn_json}")

    if not all_rows:
        print("ERROR: No data found.")
        sys.exit(1)

    all_rows = deduplicate_rows(all_rows)
    print(f"\nAfter dedup: {len(all_rows)} rows")

    write_autoheuristic_txt(all_rows, args.output, args.shared_memory, device_capa)


if __name__ == "__main__":
    main()
