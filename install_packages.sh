#!/usr/bin/env bash

# Core cyclist package
pip install -e .

# PyTorch with CUDA 12.4 — must come before .[eval] to get the GPU wheel
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# LLaVA-NeXT is not on PyPI — install from source
pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git

# VLM inference stack and API clients (torch already satisfied above)
pip install ".[eval]"

# flash-attn # check you machine if this works - can cause OOM problems for some models
pip install flash-attn==2.6.3 --no-build-isolation