#!/bin/bash
# Launch the SGLang judge server (Meta-Llama-3-70B-Instruct on port 30000).
# Requires the cycle_env_sglang conda env — see install_sg_lang.sh.
# Tensor parallelism --tp controls how many GPUs to use (70B needs >= 2 x 80GB).

base_path="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME=cycle_env_sglang
CONDA_ENV="$(conda info --base)/envs/${ENV_NAME}"
PYTHON=${CONDA_ENV}/bin/python

# Point nvcc to GCC 12 from the conda env (system GCC 9 lacks C++20 support)
export CC=${CONDA_ENV}/bin/gcc
export CXX=${CONDA_ENV}/bin/g++

# Load HF_HOME, HUGGINGFACE_HUB_CACHE from .env
set -a
source "${base_path}/.env"
set +a

PYTHONNOUSERSITE=1 $PYTHON -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3-70B-Instruct \
    --port 30000 \
    --tp 2 \
    --mem-fraction-static 0.85
