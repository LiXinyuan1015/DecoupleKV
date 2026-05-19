#!/bin/bash
# Optional: log in to your experiment tracker (e.g. swanlab/wandb) beforehand.
# Do NOT hard-code API keys here. Use environment variables instead, e.g.:
#   export SWANLAB_API_KEY=your_key_here
# or run `swanlab login` interactively.

MODEL_DIR=${MODEL_DIR:-./models/stable-diffusion-v1-5}
DATASET_DIR=${DATASET_DIR:-./data/naruto-blip-captions}
OUTPUT_DIR=${OUTPUT_DIR:-./outputs/sd-test}

python train_sd.py \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=64 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --max_train_steps=2000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --seed=42 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --checkpointing_steps=2000 \
  --dataset_name="${DATASET_DIR}" \
  --output_dir="${OUTPUT_DIR}" \
  --pretrained_model_name_or_path "${MODEL_DIR}"
