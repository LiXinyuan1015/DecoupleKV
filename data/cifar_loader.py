import os
import pickle
import random
import torch
import numpy as np
import datasets

def cifar_img_totensor(img):
    return np.reshape(img, (3, 32, 32)) / 255.

def cifar_img_normalize(img, mu = [0.4914,0.4882,0.4465], std = [0.2023,0.1994,0.2010]):
    mu, std = np.array(mu, dtype = np.float32), np.array(std, dtype = np.float32)
    mu, std = mu[:,np.newaxis,np.newaxis], std[:,np.newaxis,np.newaxis]
    return (img - mu) / std

class DataLoader(object):
    def __init__(self, data_iterator, batch_size):
        self.data_iterator = data_iterator
        self.batch_size = batch_size

    def collate_fn(self):
        raise NotImplementedError
    
    def __len__(self):
        raise NotImplementedError
    
    def __next__(self):
        batch = []
        for _ in range(self.batch_size):
            try:
                item = next(self.data_iterator)
                batch += [item]
            except StopIteration:
                break

        if not batch:
            raise StopIteration
        return self.collate_fn(batch)
    
    def __iter__(self):
        return self
    
class CifarLoader(DataLoader):
    def __init__(self, path, splits, max_batch_size = 4, shuffle = False, sample_rate = 1., num_example_per_label = -1, shuffle_rate = 0, device = "cpu"):
        self.cifar_iter = CifarIterator(path, splits, shuffle, sample_rate, num_example_per_label, shuffle_rate, device)
        batch_size = min(max_batch_size, len(self.cifar_iter.images))
        if batch_size < max_batch_size:
            print(f"Batch size is set to {batch_size} because `max_batch_size` is greator than the number of examples.")

        super(CifarLoader, self).__init__(self.cifar_iter.generate_examples(), batch_size)

    def to(self, device):
        self.cifar_iter.device = device

    def collate_fn(self, batch):
        img_batch, label_batch = [], []
        for img, label in batch:
            img_batch += [img.unsqueeze(0)]
            label_batch += [label.unsqueeze(0)]

        img_batch = torch.cat(img_batch, 0)
        label_batch = torch.cat(label_batch, 0)

        return {"input": img_batch, "label": label_batch}
    
    def __len__(self):
        return len(self.cifar_iter.images)
    
    def _reset(self):
        self.data_iterator = self.cifar_iter.generate_examples()
        
    def random_example(self, num = 1):
        random_idx = np.random.choice(len(self), num, False)
        images = self.cifar_iter.images[random_idx]
        labels = self.cifar_iter.labels[random_idx]
        return images, labels
    
class CifarIterator(object):
    def __init__(self, path, splits, shuffle, sample_rate, num_example_per_label, shuffle_rate, device):
        self.device = device
        if not isinstance(splits, list):
            splits = [splits]
        
        try:
            dicts = []
            for split in splits:
                with open(os.path.join(path, split), "rb") as fo:
                    dicts.append(pickle.load(fo, encoding="bytes"))
            self.images = np.array([dict[b"data"] for dict in dicts], dtype = np.int64)
            self.images = self.images.reshape(-1, 3, 32, 32)
            self.labels = np.array([dict[b"labels"] for dict in dicts], dtype = np.int64)
            self.labels = self.labels.reshape(-1)
        except:
            data = datasets.load_dataset(path)
            self.images = np.array([data[split]["img"] for split in splits], dtype = np.int64)
            self.images = self.images.reshape(-1, 32, 32, 3).transpose(0, 3, 1, 2)
            self.labels = np.array([data[split]["fine_label"] for split in splits], dtype = np.int64)
            self.labels = self.labels.reshape(-1)
        
        total_num = self.images.shape[0]
        num_labels = len(set(self.labels))
        
        if shuffle:
            self.shuffle()

        if num_example_per_label > 0:
            if sample_rate < 1.:
                print("Parameter: `sample_rate` will not be used because `num_example_per_label` is specified.")
            label_groups = [[] for _ in range(num_labels)]
            for idx in range(total_num):
                label_groups[self.labels[idx]].append(idx)
            choice_idxs = []
            for group in label_groups:
                choice_idxs.extend(random.sample(group, k = num_example_per_label))
            self.images = self.images[choice_idxs]
            self.labels = self.labels[choice_idxs]
        elif sample_rate < 1.:
            self.images = self.images[:int(sample_rate * total_num)]
            self.labels = self.labels[:int(sample_rate * total_num)]
            
        total_num = self.images.shape[0]
        if shuffle_rate > 0:
            shuffle_size = int(shuffle_rate * total_num)
            indices = list(range(shuffle_size))
            random.shuffle(indices)
            shuffled_indices = indices + list(range(shuffle_size, total_num))
            self.images = self.images[shuffled_indices]
        
    def shuffle(self):
        permutation_idx = np.random.permutation(len(self.images))

        self.images = self.images[permutation_idx]
        self.labels = self.labels[permutation_idx]
        
    def generate_examples(self):
        for idx, _ in enumerate(self.images):
            img_feature = cifar_img_totensor(self.images[idx])
            img_feature = cifar_img_normalize(img_feature)

            yield (
                torch.tensor(img_feature, dtype = torch.float32, device = self.device),
                torch.tensor(self.labels[idx], dtype = torch.int64, device = self.device),
            )
