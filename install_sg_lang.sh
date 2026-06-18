#!/bin/bash
# Install SGLang into a dedicated conda environment isolated from the eval pipeline.
#
# Setup:
#   conda create -n cycle_env_sglang python=3.10 -y
#   conda install -n cycle_env_sglang -c conda-forge gcc=12 gxx=12 -y
#   conda activate cycle_env_sglang
#   PYTHONNOUSERSITE=1 bash install_sg_lang.sh
#
# PYTHONNOUSERSITE=1 is required to prevent the Determined cluster's user
# site-packages from interfering with dependency resolution.

export PYTHONNOUSERSITE=1

pip install --upgrade pip

# PyTorch 2.5.1 + CUDA 12.4 — matches the cluster GPU drivers
pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124

# flashinfer must match torch + CUDA exactly
pip install flashinfer-python -i https://flashinfer.ai/whl/cu124/torch2.5/

# SGLang with flashinfer backend (cu124/torch2.5)
pip install "sglang[all]>=0.4.3.post2" \
    --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python
