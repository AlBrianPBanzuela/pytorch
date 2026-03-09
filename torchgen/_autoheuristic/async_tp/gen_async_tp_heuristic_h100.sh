#!/bin/bash
# Generate AsyncTP heuristic for H100 from existing data.
# Assumes async_tp_h100_data.txt exists (run convert_benchmark_data.py first).

set -e

data="async_tp_h100_data.txt"

if [ ! -f "$data" ]; then
    echo "Data file not found: $data"
    echo "Run: python convert_benchmark_data.py --output $data"
    exit 1
fi

python train_decision_async_tp.py "${data}" --heuristic-name AsyncTPFuseH100
