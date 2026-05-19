from pathlib import Path
from datasets import load_dataset
import numpy as np

class WikipediaDataset(object):
    def __init__(self, data_dir, split = "train", dataset_size_limit = -1):
        data_dir = Path(data_dir)
        files = data_dir / "*.parquet"
        data = load_dataset(
            'parquet', 
            data_files={'train': str(files)}
        )[split]
        data = self.length_sample(data)
        if dataset_size_limit > 0:
            self.data = data[:dataset_size_limit]
        else:
            self.data = data
        self.data_seek = 0
        
    def length_sample(self, data):
        data = np.asarray(data, dtype = object)
        lengths = np.array([len(prompt) for prompt in data])
        mu, sigma = lengths.mean(), lengths.std()
        indices = np.where(abs(lengths - mu) <= sigma)
        data = data[indices]
        return list(data)
            
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.data_seek >= len(self.data):
            self.data_seek = 0
            raise StopIteration
        else:
            item = self.data[self.data_seek]['text']
            self.data_seek += 1
            return item
            
    