import os
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn.functional as F

import tqdm
from transformers import CLIPTokenizer
from diffusers import (
    AutoencoderKL, 
    DDPMScheduler, 
    UNet2DConditionModel
)
from diffusers.optimization import get_scheduler

from data.caption_loader import get_caption_loader
from custom.clip import CLIPTextModel

seed = 42
device = "cuda"
model_id = os.environ.get("SD_MODEL_DIR", "./models/stable-diffusion-v1-5")
dataset_id = os.environ.get("DATASET_DIR", "./data/imagenet_clip_1token")
weight_dtype = torch.bfloat16
text_encoder = CLIPTextModel.from_pretrained(
    model_id, 
    subfolder="text_encoder", 
    dtype=weight_dtype,
    device_map=device,
)
vae = AutoencoderKL.from_pretrained(
    model_id, 
    subfolder="vae", 
    torch_dtype=weight_dtype,
    device_map=device,
)
unet = UNet2DConditionModel.from_pretrained(
    model_id, 
    subfolder="unet", 
    torch_dtype=weight_dtype,
    device_map=device,
)

noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")

learning_rate = 1e-2
weight_decay = 1e-2
lr_warmup_steps = 0
max_grad_norm = 1
resolution = 512
train_batch_size = 32
num_train_epochs = 1600
lr_scheduler = "constant"

vae.requires_grad_(False)
text_encoder.requires_grad_(False)
unet.requires_grad_(False)

# activate plug-ins
text_encoder.init_plugin(
    use_act = False,
    weight_dtype = weight_dtype,
    device = device,
)
text_encoder.plugin.enable_training()

train_dataloader, dataset = get_caption_loader(
    dataset_id = dataset_id,
    tokenizer = tokenizer,
    resolution = resolution,
    train_batch_size = train_batch_size,
    label_shuffle_ratio = 1.,
)

optimizer = torch.optim.SGD(
    text_encoder.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay,
)

lr_scheduler = get_scheduler(
    lr_scheduler,
    optimizer=optimizer,
    num_warmup_steps=lr_warmup_steps,
)

loss_hist = []

for epoch in tqdm.tqdm(range(num_train_epochs), desc="Steps"):
    epoch_loss = 0
    num_batches = 0
    for step, batch in enumerate(train_dataloader):
        # Convert images to latent space
        for key, value in batch.items():
            batch[key] = value.to(device)
        latents = vae.encode(batch["pixel_values"].to(weight_dtype)).latent_dist.sample()
        latents = latents * vae.config.scaling_factor

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        #timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        ratio = noise_scheduler.config.num_train_timesteps // num_train_epochs
        timesteps = torch.full((bsz,), epoch * ratio, device=device)
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # Get the text embedding for conditioning
        encoder_hidden_states = text_encoder(batch["input_ids"], return_dict=False)[0]

        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

        # Predict the noise residual and compute loss
        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]

        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        # Gather the losses across all processes for logging (if we use distributed training).
        step_loss = loss.item()
        epoch_loss += step_loss; num_batches += 1

        # Backpropagate
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(unet.parameters(), max_grad_norm)
        # print(unet.down_blocks[0].hs.grad.std(), unet.down_blocks[0].ehs.grad.std())

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    loss_hist.append(epoch_loss / num_batches)

print("Training steps: %s" % len(loss_hist))

base_path = "./checkpoints/sd-plugin/"
ckpt_path = base_path + "plugin.bin"
fig_path = base_path + "loss.png"

torch.save(text_encoder.plugin.state_dict(), ckpt_path)

plt.figure(figsize = (8, 4), dpi = 150)
plt.xlabel("iteration num", fontsize=14)
plt.ylabel("loss", fontsize=14)
sns.lineplot(x = list(range(len(loss_hist))), y = loss_hist, zorder=1)
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
