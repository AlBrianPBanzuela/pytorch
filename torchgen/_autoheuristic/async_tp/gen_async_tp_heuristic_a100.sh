#!/bin/bash
# Generate AsyncTP heuristic for A100 from existing data.

set -e

data="async_tp_a100_data.txt"

if [ ! -f "$data" ]; then
    echo "Data file not found: $data"
    echo "Run benchmarks on A100 and convert: python convert_benchmark_data.py --output $data --shared-memory 166912 --device-capa 8,0"
    exit 1
fi

python train_decision_async_tp.py "${data}" --heuristic-name AsyncTPFuseA100
