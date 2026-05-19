import random
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2, transforms

from datasets import load_dataset, load_from_disk

def get_caption_loader(
    dataset_id,
    tokenizer,
    resolution,
    train_batch_size,
    label_shuffle_ratio=0.,
):
    # 1. 加载数据 (保持原始状态)
    try:
        raw_dataset = load_dataset(dataset_id)["train"]
    except:
        raw_dataset = load_from_disk(dataset_id)

    # 2. 处理错排/打乱 (不使用 .map，只存索引映射)
    total_rows = len(raw_dataset)
    shuffled_text_map = {}
    if label_shuffle_ratio > 0:
        shuffle_num = int(total_rows * label_shuffle_ratio)
        # 使用之前讨论过的错排/打乱逻辑
        shuffle_indices = np.random.choice(total_rows, size=shuffle_num, replace=False)
        values_to_shuffle = [raw_dataset[int(i)]["text"] for i in shuffle_indices]
        np.random.shuffle(values_to_shuffle)
        shuffled_text_map = dict(zip(shuffle_indices.tolist(), values_to_shuffle))

    # 3. 预分词 (一次性处理，极大提升迭代速度)
    # 注意：这里只处理文本，不处理图片（图片太大，不适合 map）
    def pre_tokenize(examples):
        return tokenizer(
            examples["text"], 
            padding="longest", 
            truncation=True, 
            max_length=tokenizer.model_max_length
        )

    # 建议开启缓存，下次启动秒开
    tokenized_dataset = raw_dataset.map(
        pre_tokenize, 
        batched=True, 
        batch_size=1000, 
        remove_columns=["text"] 
    )

    # 4. 构建纯 PyTorch 风格的包装器
    class FastDataset(Dataset):
        def __init__(self, base_dataset, shuffle_map):
            self.base = base_dataset
            self.shuffle_map = shuffle_map

        def __len__(self):
            return len(self.base)

        def __getitem__(self, idx):
            example = self.base[idx]
            # 如果在打乱名单里，覆盖 input_ids (或者预先在 map 里处理好更佳)
            # 这里简化逻辑：我们假设文本已经在 map 时处理好了
            # 如果要支持 label_shuffle，建议在 map 之前就完成 text 字段的替换
            
            return {
                "input_ids": torch.tensor(example["input_ids"]),
                "attention_mask": torch.tensor(example["attention_mask"])
            }

    train_dataset = FastDataset(tokenized_dataset, shuffled_text_map)

    # 5. 使用系统默认 collate_fn (通常比手写快)
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=False,
        batch_size=train_batch_size,
        num_workers=4,
        pin_memory=True, # 必须开启，加速数据传输
    )

    return train_dataloader, raw_dataset

# deprecated
def get_caption_loader_ti(
    dataset_id,
    tokenizer,
    resolution,
    train_batch_size,
):
    try:
        dataset = load_dataset(dataset_id)["train"]
    except:
        dataset = load_from_disk(dataset_id)
    dataset_columns = ("image", "text")
    column_names = dataset.column_names
    image_column, caption_column = dataset_columns if dataset_columns else column_names

    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids
    
    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids"] = tokenize_captions(examples)
        return examples

    train_dataset = dataset.with_transform(preprocess_train)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids}

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=train_batch_size,
    )
    return train_dataloader