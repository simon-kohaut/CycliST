pip install --upgrade pip
# pip install sgl-kernel --force-reinstall --no-deps
# The flash infer package needs to match the torch version including the correct CUDA version
# pip install "sglang[all]>=0.4.3.post2" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python
# pip install "sglang[all]>=0.4.5" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python

# '2.5.1+cu124'

#pip install flashinfer-python -i https://flashinfer.ai/whl/cu124/torch2.5/


pip install sgl-kernel -i https://docs.sglang.ai/whl/cu118
pip install "sglang[all]<0.4.3"
