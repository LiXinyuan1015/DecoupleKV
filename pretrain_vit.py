import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import time
import tqdm
import numpy as np
import torch
import torch.optim as optim

from custom.image_models import VitModel, example_config
from data.cifar_loader import CifarLoader

cifar_path = os.environ.get("CIFAR_PATH", "./data/cifar-100")
cifar_pretraining_dataset = CifarLoader(
    path = cifar_path,
    splits = "train",
    # splits = [f"data_batch_{i}" for i in range(1,6)],
    batch_size = 128,
)
cifar_eval_dataset = CifarLoader(
    path = cifar_path,
    splits = "test",
    # splits = "test_batch",
    batch_size = 128,
)

num_examples = 5
random_images, random_labels = cifar_pretraining_dataset.random_example(num_examples)
fig, axes = plt.subplots(1, num_examples, figsize=(3 * num_examples, 3), dpi = 150)

# 遍历每个矩阵并显示
for i, ax in enumerate(axes):
    # 将 (3, 32, 32) 转换为 (32, 32, 3) 以便 matplotlib 显示
    img = np.transpose(random_images[i], (1, 2, 0))
    ax.imshow(img)
    ax.set_title(f"Image {i+1}")
    ax.axis('off')  # 关闭坐标轴

plt.tight_layout()
plt.show()

model_config = example_config
num_labels = len(set(cifar_pretraining_dataset.cifar_iter.labels))
model_config.num_labels = num_labels
model = VitModel(model_config)
# print([n for n, p in model.named_parameters() if p.requires_grad])


device = "cuda"
model.to(device)
cifar_pretraining_dataset.to(device)
cifar_eval_dataset.to(device)
num_epoch = 1
lr = 1e-3
scaler = torch.amp.GradScaler()
optimizer = optim.Adam(model.parameters(), lr = lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch)
loss_hist = []

for i in tqdm.tqdm(range(num_epoch)):
    # reset train dataloader
    cifar_pretraining_dataset._reset()
    # start training
    model.train()
    epoch_start_time = time.time()
    num_correct, num_train, num_loss = 0, 0, 0
    loss_total = 0.
    for batch in cifar_pretraining_dataset:
        input_feature, label = batch["input"], batch["label"]
        with torch.amp.autocast(device):
            output = model(input_feature, label)
        loss, label_predict = output.loss, output.label_predict
        model.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        num_train += label.size(0)
        num_loss += 1
        num_correct += torch.sum(label_predict == label).item()
        loss_total += loss.item()
        
        # print(loss.item())
    loss_hist.append(loss_total / num_loss)

base_path = os.environ.get("CKPT_DIR", "./checkpoints/cifar-test/")
os.makedirs(base_path, exist_ok=True)
ckpt_path = base_path + "vit.bin"
fig_path = base_path + "loss.png"
print(f"Saving model weights in {ckpt_path}...")
torch.save(model.state_dict(), ckpt_path)

plt.figure(figsize = (8, 4), dpi = 150)
plt.xlabel("iteration num", fontsize=14)
plt.ylabel("loss", fontsize=14)

sns.lineplot(x = list(range(len(loss_hist))), y = loss_hist, zorder=1)

plt.savefig(fig_path, dpi=150, bbox_inches='tight')
