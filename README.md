# Anonymous Submission

This repository contains the source code accompanying our anonymous submission.
All author-, institution- and infrastructure-identifying information has been
removed for the double-blind review process.

## Overview

The codebase studies **knowledge localization, erasure, debiasing and
discretization in pretrained Transformers** (BERT / GPT-2 / GPT-J / CLIP), and
also includes a Stable-Diffusion plug-in training pipeline used in our
image-side experiments.

The high-level pipeline implemented in `src/` is:

1. **Locate** knowledge / bias-related neurons in a frozen model.
2. **Discretize** the located sub-network into interpretable components.
3. **Erase** / **Debias** the targeted behaviour while keeping the rest of the
   model's capability intact.
4. **Evaluate** the resulting model on the corresponding downstream task.

## Repository layout

```
.
├── config/                 # global configs and per-model hyper-parameters
├── custom/                 # forked HF model implementations with hooks
│   ├── bert/  clip/  gpt2/  gptj/
│   └── image_models/       # ViT / CNN used in the CIFAR experiments
├── data/                   # dataset loaders (Wikipedia, Pararel, Wino-Bias,
│                           #  CIFAR, ImageNet, caption datasets, ...)
├── src/                    # core algorithms: locate / discretize / erase /
│                           # debias / evaluate / generate
├── analysis/               # post-hoc analysis notebooks
├── scripts/                # training launch scripts
├── pretrain_vit.py         # ViT pre-training on CIFAR-100
├── train_sd.py             # Stable-Diffusion fine-tuning (HF reference impl.)
├── train_sd_plugin.py      # SD plug-in training (ours)
├── text2img.ipynb          # qualitative SD inference demo
├── visualize.ipynb         # plots used in the paper
├── learning_2W.ipynb       # language-model learning-dynamics notebook
└── learning_cifar.ipynb    # vision-model learning-dynamics notebook
```

## Installation

```bash
# Python >= 3.10 is recommended.
pip install -r requirements.txt
```

A CUDA-enabled PyTorch matching your local CUDA toolkit is also required
(install it separately from <https://pytorch.org/get-started/locally/>).

## Configuring paths

The code does **not** assume any particular filesystem layout. All paths to
pretrained models and datasets are read from environment variables (with sane
local defaults under `./models` and `./data`):

| Variable           | Used for                                | Default                                |
| ------------------ | --------------------------------------- | -------------------------------------- |
| `LLM_DIR`          | parent dir for BERT / GPT-2 / GPT-J     | `./models`                             |
| `SD_MODEL_DIR`     | Stable-Diffusion v1.5 checkpoint dir    | `./models/stable-diffusion-v1-5`       |
| `OPEN_CLIP_CKPT`   | open_clip ViT-B/32 weights              | `./models/.../open_clip_pytorch_model.bin` |
| `CIFAR_PATH`       | CIFAR-100 root                          | `./data/cifar-100`                     |
| `DATASET_DIR`      | caption dataset for SD plug-in training | `./data/imagenet_clip_1token`          |
| `IMAGENET_PARQUET` | ImageNet-1k parquet glob                | `./data/imagenet-1k-256x256/.../*.parquet` |
| `CKPT_DIR`         | output checkpoint dir for `pretrain_vit.py` | `./checkpoints/cifar-test/`        |

Either symlink your model/data into these defaults, or `export` the variables
to point at the locations you actually have, e.g.:

```bash
export LLM_DIR=/path/to/your/llms
export SD_MODEL_DIR=/path/to/stable-diffusion-v1-5
```

## Quick start

### 1. Language-model experiments (knowledge / bias)

```bash
# Locate -> discretize -> erase / debias -> evaluate
# The model & method are selected via config/__init__.py and src/*.py entry
# points; see the notebooks under analysis/ for end-to-end usage.
```

### 2. ViT pre-training on CIFAR-100

```bash
export CIFAR_PATH=./data/cifar-100
export CKPT_DIR=./checkpoints/cifar-test/
python pretrain_vit.py
```

### 3. Stable-Diffusion fine-tuning (HF reference baseline)

```bash
# Optionally log in to your experiment tracker first (do NOT commit any API
# keys); see scripts/train_sd.sh for environment variables.
bash scripts/train_sd.sh
```

### 4. Stable-Diffusion plug-in training (ours)

```bash
export SD_MODEL_DIR=/path/to/stable-diffusion-v1-5
export DATASET_DIR=/path/to/imagenet_clip_1token
python train_sd_plugin.py
```

## Notes for reviewers

* The files under `custom/` are adapted from the HuggingFace `transformers`
  library; original Apache-2.0 copyright headers are preserved at the top of
  each file.
* All notebook outputs have been stripped to keep the repository small and
  to avoid leaking environment information; rerun the notebooks locally to
  reproduce the figures.
* Random seeds are fixed inside the individual entry points; small numerical
  differences may still occur across GPU architectures and library versions.

## License

The original portions of this code are released for academic review purposes
only. Third-party files retain their upstream licenses (Apache-2.0 for the
HuggingFace-derived modules under `custom/`).
