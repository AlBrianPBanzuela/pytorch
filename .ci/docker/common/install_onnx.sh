#!/bin/bash

set -ex

source "$(dirname "${BASH_SOURCE[0]}")/common_utils.sh"

retry () {
    "$@" || (sleep 10 && "$@") || (sleep 20 && "$@") || (sleep 40 && "$@")
}

# ONNXRuntime should be installed before installing
# onnx-weekly. Otherwise, onnx-weekly could be
# overwritten by onnx.
# Note: parameterized, pytest-subtests, tabulate, packaging are already
# installed via requirements-ci.txt
pip_install \
  transformers==5.2.0 \
  onnxruntime==1.23.1

# Cache the transformers model to be used later by ONNX tests. We need to run the transformers
# package to download the model. By default, the model is cached at ~/.cache/huggingface/hub/
# Need a PyTorch version for transformers to work
pip_install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu
conda_run python -c 'import transformers; transformers.GPTJForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gptj")'

# Cleaning up
conda_run pip uninstall -y torch
