import json
import random
import pathlib

class PARARELDataset(object):
    def __init__(self, data_dir, task_type = 'ntp', dataset_size_limit = -1):
        self.task_type = task_type
        if task_type == 'ntp' or task_type == 'qa':
            path = pathlib.Path(data_dir) / "PARAREL" / "data_all_for_qa.json"
            if not path.exists():
                source_path = pathlib.Path(data_dir) / "PARAREL" / "data_all_allbags.json"
                print(f"Preprocessed data not exists, preparing data from {source_path}, and it will be saved at {path}.")
                data = self.preprocess(source_path, path)
        else:
            path = pathlib.Path(data_dir) / "PARAREL" / "data_all_allbags.json"
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        data = self.flatten(data)
        if dataset_size_limit > 0:
            self.data = data[:dataset_size_limit]
        else:
            self.data = data

    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)

    def flatten(self, data, n_samples_bag = 1):
        data_new = []
        for rel, rel_bags in data.items():
            for bag in rel_bags:
                sample_size = min(len(bag), n_samples_bag)
                bag = random.sample(bag, k = sample_size)
                for example in bag:
                    source, target, relation = example
                    if self.task_type in ('qa', 'ntp'):
                        if not target.startswith(' '):
                            target = ' ' + target
                    data_new += [{'source': source, 'target': target, 'relation': relation, 'id': len(data_new)}]
        return data_new

    def preprocess(self, source_path, target_path):
        with open(source_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        data_new = {}
        for rel, rel_bags in data.items():
            rel_bags_new = []
            for bag in rel_bags:
                bag_new = []
                for example in bag:
                    source, target, relation = example
                    if source.endswith(' [MASK]'):
                        source = source.strip(' [MASK]')
                        bag_new += [[source, target, relation]]
                    elif source.endswith(' [MASK].'):
                        source = source.strip(' [MASK].')
                        bag_new += [[source, target, relation]]
                    else:
                        pass
                rel_bags_new += [bag_new]
            data_new[rel] = rel_bags_new

        with open(target_path, "w", encoding="utf-8") as f:
            data = json.dump(data_new, f, indent=2, ensure_ascii=False)
        
        return data_new
