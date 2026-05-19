from pathlib import Path
from datasets import load_dataset

class WinoBiasDataset(object):
    def __init__(self, data_dir, task_type = "ntp", dataset_size_limit = -1):
        data_dir = Path(data_dir)
        files = data_dir / "wino_bias"
        print(files)
        self.task_type = task_type
        data = load_dataset(str(files))
        data = self.preprocess([item for item in data['validation']] + [item for item in data['test']])
        if dataset_size_limit > 0:
            self.data = data[:dataset_size_limit]
        else:
            self.data = data
        self.data_seek = 0
        
        
    def preprocess(self, examples):
        pronouns = {"he", "she", "him", "her", "his"}
        lines = []
        # sorted examples by case id
        sorted_examples = sorted(examples, key = lambda x:x['document_id'].split("/")[-1])
        for i in range(0, len(sorted_examples), 2):
            line = dict()
            ex1, ex2 = sorted_examples[i], sorted_examples[i+1]
            ex1_id, ex2_id = ex1['document_id'], ex2['document_id']
            assert ex1_id.split("/")[-1] == ex2_id.split("/")[-1], "%s != %s" % (ex1_id, ex2_id)
            ex1, ex2 = (ex1, ex2) if 'not_stereotype' in ex2_id else (ex2, ex1)
            pro_tokens = ex1['tokens']
            anti_tokens = ex2['tokens']
            pro_pronoun_idx = next((i for i, x in enumerate(pro_tokens) if x in pronouns), None)
            anti_pronoun_idx = next((i for i, x in enumerate(anti_tokens) if x in pronouns), None)
            assert  pro_pronoun_idx is not None, "Pronoun dose not exist in the tokens, pro: {}, anti: {}".format(pro_tokens, anti_tokens)
            line['id'] = len(lines)
            line['pro_target'] = pro_tokens[pro_pronoun_idx]
            line['anti_target'] = anti_tokens[anti_pronoun_idx]
            if self.task_type == "ntp":
                line['source'] = " ".join(pro_tokens[:pro_pronoun_idx])
            elif self.task_type == "mlm":
                pro_tokens[pro_pronoun_idx] = '[MASK]'
                line['source'] = " ".join(pro_tokens)
            lines.append(line)
        return lines
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
            
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.data_seek >= len(self.data):
            self.data_seek = 0
            raise StopIteration
        else:
            item = self.data[self.data_seek]
            self.data_seek += 1
            return item
            
    