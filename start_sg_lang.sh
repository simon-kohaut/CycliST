export HF_HOME=/pfss/mlde/workspaces/mlde_wsp_PI_Kersting/LLaVA-cake/Cyclist/hfcache
export HUGGINGFACE_HUB_CACHE=/pfss/mlde/workspaces/mlde_wsp_PI_Kersting/LLaVA-cake/Cyclist/hfcache
export HF_TOKEN="TOKEN"


#python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-7B-Instruct --port 30000
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-70B-Instruct --tp 2
