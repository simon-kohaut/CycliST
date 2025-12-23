#Install PyTorch 2.5.1 with Cuda 12.4
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
pip install accelerate
pip install openai
pip install flash-attn