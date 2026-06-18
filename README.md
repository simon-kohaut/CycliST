# CycliST: A Benchmark for Visual Reasoning on Cyclical State Transitions

<p align="center">
    <a href="https://arxiv.org/abs/2512.01095"><img src="https://img.shields.io/badge/arXiv-2512.01095-B31B1B.svg?logo=arxiv" alt="arXiv"></a>
    <a href="https://openreview.net/forum?id=l03g53HUL2"><img src="https://img.shields.io/badge/OpenReview-DMLR_2026-8C1A11.svg" alt="OpenReview"></a>
    <a href="https://simon-kohaut.github.io/CycliST/"><img src="https://img.shields.io/badge/Website-CycliST-2176BC?logo=GoogleChrome" alt="Website"></a>
    <a href="https://huggingface.co/datasets/AIML-TUDA/CycliST"><img src="https://img.shields.io/badge/HuggingFace-Dataset-FFD21E.svg?logo=huggingface" alt="Dataset"></a>
</p>

This is the official GitHub repository for the paper **CycliST: A Video Language Model Benchmark for Reasoning on Cyclical State Transitions**, Simon Kohaut, Daniel Ochs, Shun Zhang, Benedict Flade, Julian Eggert, Kristian Kersting, and Devendra Singh Dhami, DMLR 2026.

CycliST is a synthetic video benchmark for evaluating vision-language models (VLMs) on **cyclical state transitions**. It generates 3D-rendered scenes with cyclic motion patterns (orbits, linear motion, rotations, attribute changes), produces structured question-answer pairs about those cycles, and evaluates VLMs on these.

The pipeline has four stages:

```
Scene Rendering  →  Question Generation  →  VLM Inference  →  LLM Judging
 (Blender)           (question_engine)       (eval_vqa)       (judge_vqa)
```

---

## Installation

### Prerequisites

- **Python 3.10**
- **CUDA-capable GPU** for local model inference
- **Blender 4.0** for scene rendering (not needed for inference/judging only)

### Python environment

Create and activate the conda environment (sets up Python 3.10):

```bash
conda env create -f environment.yml
conda activate cycle_env
```

Install all Python packages — core deps, PyTorch with CUDA 12.4, LLaVA-NeXT, and the full inference stack:

```bash
bash install_packages.sh
```

If the machine injects system-level packages that conflict with the conda env (run once, then reactivate):

```bash
conda env config vars set PYTHONNOUSERSITE=1 -n cycle_env
conda activate cycle_env
```

### Blender for scene generation

Install Blender 4.0 and register the `cyclist` package with Blender's bundled Python:

```bash
sudo bash install_blender.sh
```

This downloads Blender 4.0.1 to `/opt/blender`, installs its system dependencies, adds it to `PATH`, and writes a `.pth` file so Blender's Python can import `cyclist`. Run it from the repo root so the path registration points to the right directory.

### LLM judge server (SGLang)

The judge runs a local [SGLang](https://github.com/sgl-project/sglang) server and requires a **separate** conda environment to avoid dependency conflicts with the eval stack:

```bash
conda create -n cycle_env_sglang python=3.10 -y
conda install -n cycle_env_sglang -c conda-forge gcc=12 gxx=12 -y   # C++20 needed for JIT kernels
conda activate cycle_env_sglang
PYTHONNOUSERSITE=1 bash install_sg_lang.sh
```

### API keys

Create a `.env` file in the repo root with the keys you need:

```bash
HF_HOME=/path/to/huggingface/cache
HUGGINGFACE_HUB_CACHE=/path/to/huggingface/cache
HF_TOKEN=your_huggingface_token          # required to download gated models (Llama judge)
GOOGLE_API_KEY=your_google_api_key       # required for Gemini models
TOKENIZERS_PARALLELISM=false
```

---

## Usage

### 1 — Scene Rendering

Render scenes using the provided shell scripts. Each script generates `train/test/val` splits (300/150/150 videos).

```bash
# Single-cycle scenes (one object with one cycle type)
bash scripts/scene/unicycle.sh

# Two-cycle scenes
bash scripts/scene/bicycle.sh

# Three-cycle scenes
bash scripts/scene/tricycle.sh

# Single-cycle scenes with cluttered background (4–9 extra objects)
bash scripts/scene/unicycle_cluttered.sh
```

Output is written to `output/scenes/` (scene JSONs) and `output/videos/` (rendered MP4s).

> Requires Blender 4.0. The code was run on linux and installation might change for other OS.

### 2 — Question Generation

Generate question-answer pairs from the rendered scenes:

```bash
bash scripts/questions/question_gen.sh
```

This runs all 12 question templates across the scene output. Output is written to `output/questions/`.

To inspect generated questions interactively, use `cyclist/questions/question_inspection.ipynb`.

### 3 — VLM Inference

Run VLM inference on the generated questions. Models are selected by number (1–9):

| # | Model |
|---|---|
| 1 | `lmms-lab/LLaVA-Video-7B-Qwen2` |
| 2 | `llava-hf/llava-onevision-qwen2-7b-ov-chat-hf` |
| 3 | `OpenGVLab/InternVideo2_5_Chat_8B` |
| 4 | `gemini-2.0-flash` |
| 5 | `gemini-2.5-flash` |
| 6 | `lmms-lab/LLaVA-Video-72B-Qwen2` |
| 7 | `llava-hf/llava-onevision-qwen2-72b-ov-chat-hf` |
| 8 | `OpenGVLab/InternVL3-8B` |
| 9 | `OpenGVLab/InternVL3-78B` |

```bash
# VQA inference (model 1 = LLaVA-Video-7B, dataset = unicycle, fps = 8)
bash scripts/eval/eval_vqa.sh --model 1 --dataset unicycle --fps 8

# Scene understanding inference
bash scripts/eval/eval_scene_understanding.sh --model 1 --dataset unicycle --fps 8
```

For Gemini models, upload videos first:

```bash
bash scripts/eval/upload_gemini.sh --dataset unicycle
```

Answers are written to `output/eval/answers/`.

### 4 — LLM Judging

Judging uses a local [SGLang](https://github.com/sgl-project/sglang) server running `meta-llama/Meta-Llama-3-70B-Instruct`.

Start the SGLang server in a separate terminal before running the judge (loads `meta-llama/Meta-Llama-3-70B-Instruct` on port 30000):

```bash
bash start_sg_lang.sh
```

Initialise the request offset tracker once per run:

```bash
echo '{"offset": 0}' > offset.json
```

Run the judge from the repo root (requires `cycle_env`):

```bash
# Judge VQA answers
PYTHONNOUSERSITE=1 python -m cyclist.judge.judge_vqa

# Judge scene understanding captions
PYTHONNOUSERSITE=1 python -m cyclist.judge.judge_scene_understanding
```

Answers are read from `output/eval/answers/` and metrics are written as `*_metrics.json` files alongside each answer CSV. Use the companion notebooks in `cyclist/judge/` to generate results tables and LaTeX output:

- `vqa_results_counting.ipynb`
- `vqa_results_descriptive.ipynb`
- `vqa_results_representative.ipynb`
- `vqa_results_scene_understanding.ipynb`

---

## Citation

```bibtex
@article{kohaut2026cyclist,
      title={CycliST: A Video Language Model Benchmark for Reasoning on Cyclical State Transitions},
      author={Simon Kohaut and Daniel Ochs and Shun Zhang and Benedict Flade and Julian Eggert and Kristian Kersting and Devendra Singh Dhami},
      journal={Journal of Data-centric Machine Learning Research},
      year={2026},
      url={https://openreview.net/forum?id=l03g53HUL2},
}
```

---

## Acknowledgments

CycliST builds on ideas from [CLEVR](https://github.com/facebookresearch/clevr-dataset-gen).
