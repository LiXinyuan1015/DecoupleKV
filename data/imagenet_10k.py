import os
import tqdm
import tqdm.auto
import torch
import open_clip

from datasets import load_dataset, Dataset
from transformers import CLIPTokenizer

# ========= Configuration =========
# All paths below can be overridden via environment variables so that
# the script does not depend on any local/shared filesystem layout.
OUT_DIR = os.environ.get("OUT_DIR", "imagenet_clip_1token")
os.makedirs(OUT_DIR, exist_ok=True)
clip_tokenizer_path = os.environ.get(
    "SD_MODEL_DIR", "./models/stable-diffusion-v1-5"
)
clip_model_path = os.environ.get(
    "OPEN_CLIP_CKPT",
    "./models/CLIP-ViT-B-32-DataComp.XL-s13B-b90K/open_clip_pytorch_model.bin",
)
IMAGENET_PARQUET = os.environ.get(
    "IMAGENET_PARQUET",
    "./data/imagenet-1k-256x256/data/validation*",
)

tokenizer = CLIPTokenizer.from_pretrained(clip_tokenizer_path, subfolder="tokenizer")

# ========= 1. 加载 ImageNet-1K =========
# 注意：需要你本地已经有 imagenet 访问权限（HF token）
ds = load_dataset(
    "parquet",
    data_files = IMAGENET_PARQUET,
    split = "train",
)

# label id -> text
id2label = ds.features["label"].names

# ========= 2. 筛选 token 长度为 1 的标签 =========
valid_label_ids = set()
label_text = {}

for i, name in enumerate(id2label):
    # ImageNet label 可能是 "golden retriever, retriever"
    texts = name.split(",")
    for text in texts:
        text = text.replace("_", " ").strip()
        tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(tokens) == 1:
            valid_label_ids.add(i)
            label_text[i] = text
            break

print(f"Valid labels (token_len=1): {len(valid_label_ids)}")
print([label_text[id] for id in valid_label_ids])

# ========= 3. 为每个 label 选一张代表图像 =========
selected = {t: [] for _,t in label_text.items()}
device = "cuda"
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', 
    pretrained=clip_model_path,
)
model.to(device)
tokenizer = open_clip.get_tokenizer(clip_model_path)

def collate_fn(xs):
    batch, images, texts = {}, [], []
    for x in xs:
        images.append(preprocess(x["image"]).unsqueeze(0))
        texts.append(label_text[x["label"]])
    images = torch.cat(images, dim = 0)
    batch = {"image": images, "text": texts}
    return batch

dataset = ds.filter(lambda x: x["label"] in valid_label_ids)
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=128, collate_fn=collate_fn
)

global_idx = 0
with torch.no_grad():
    for batch in tqdm.tqdm(dataloader):
        image, text = batch["image"].to(device), tokenizer(batch["text"]).to(device)
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        sim = (image_features @ text_features.T).diag().cpu().tolist()
        for idx, (t, s) in enumerate(zip(batch["text"], sim)):
            selected[t].append((s, global_idx + idx))
        global_idx += len(sim)
print([(t, len(v)) for t,v in selected.items()])
ti_pairs = {"text": [], "image": []}
for text, cands in selected.items():
    if len(cands) == 0:
        continue
    cand_idx = sorted(cands, key = lambda x:-x[0])[0][1]
    ti_pairs["text"].append(text)
    ti_pairs["image"].append(dataset[cand_idx]["image"])

dd = Dataset.from_dict(ti_pairs)
dd.save_to_disk(OUT_DIR)

print("Done.")
