import os
import json
import random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def set_seed_all(seed = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_attr(module, param_name, global_prefix = ""):
    param_name = global_prefix + param_name
    attrs = param_name.split('.')
    current_module = module
    for attr in attrs:
        if attr.isdigit():
            current_module = current_module[int(attr)]
        else:
            current_module = getattr(current_module, attr)
    return current_module

def set_attr(module, param_name, value, global_prefix = ""):
    param_name = global_prefix + param_name
    attrs = param_name.split('.')
    current_module = module
    for attr in attrs[:-1]:
        if attr.isdigit():
            current_module = current_module[int(attr)]
        else:
            current_module = getattr(current_module, attr)
    setattr(current_module, attrs[-1], value)

def cross_entropy(x, y):
    v = x.size(-1)
    x = x.view(-1, v)
    y = F.one_hot(y, v).view(-1, v)
    assert x.size() == y.size(), f'input size mismatch, got x: {x.size()} and y: {y.size()}'
    likelihood = F.log_softmax(x, -1)
    return -torch.sum(likelihood * y) / x.size(0)

def entropy(x, eps = 1e-8):
    p = F.softmax(x, dim = -1)
    entro = -torch.diag(p @ torch.log(p + eps).T).mean()
    return entro

def znorm(x, eps = 1e-8):
    mu = x.mean(dim = -1, keepdims = True)
    var = x.var(dim = -1, keepdims = True)
    return (x - mu) / torch.sqrt(var + eps)

def inverse_store(kn_store):
    dic = dict()
    for l, layer_store in kn_store.items():
        for layer_idx, kn in enumerate(layer_store):
            if kn['case_id'] not in dic:
                dic[kn['case_id']] = {
                    'layers': [l], 
                    'layer_ids': [layer_idx],
                    'source': kn['source'], 
                    'target': kn['target'] if 'target' in kn else kn['pro_target'], 
                    'x': {l:kn['states']},
                    'z': {l:kn['grads'] if 'grads' in kn else kn['pro_grads']},
                    'index': [kn['index']],
                    'entropy': [kn['entropy'] if 'entropy' in kn else None],
                }
            else:
                dic[kn['case_id']]['layers'].append(l)
                dic[kn['case_id']]['layer_ids'].append(layer_idx)
                dic[kn['case_id']]['index'].append(kn['index'])
                dic[kn['case_id']]['x'][l] = kn['states']
                dic[kn['case_id']]['z'][l] = kn['grads'] if 'grads' in kn else kn['pro_grads']
                if 'entropy' in kn:
                    dic[kn['case_id']]['entropy'].append(kn['entropy'])
    return dic

def filter_by_entro(kn_dic, sample_rate = 1.0, temp_file = None):
    if temp_file and os.path.exists(temp_file):
        sample_ids = json.load(open(temp_file, 'r'))
    else:
        if temp_file is not None:
            id2entro = [(key, np.mean(value['entropy'])) for (key, value) in kn_dic.items()]
            sample_ids = sorted(id2entro, key=lambda x:-x[-1])[:int(sample_rate * len(id2entro))]
            sample_ids = [item[0] for item in sample_ids]
            if temp_file:
                json.dump(sample_ids, open(temp_file, 'w'), ensure_ascii=False, indent=4)
        else:
            sample_ids = random.sample(sorted(kn_dic.keys()), int(sample_rate * len(kn_dic.keys())))
    return {case_id: kn_dic[case_id] for case_id in sample_ids if case_id in kn_dic}
    

def split_kn_dic(kn_dic, num_local = 64, seed = 42):
    random.seed(seed)
    key_set = kn_dic.keys()
    keys = sorted(key_set)
    local_keys = random.sample(keys, num_local)
    test_keys = key_set - set(local_keys)
    test_dic = {k: kn_dic[k] for k in test_keys}
    local_dic = {k: kn_dic[k] for k in local_keys}
    return test_dic, local_dic


def chunks(arr, chunk_size):
    for i in range(0, len(arr), chunk_size):
        yield arr[i: i + chunk_size]

def json_auto_save(result_dir, objects, file_names, history_obj = None):
    os.makedirs(result_dir, exist_ok=True)
    # Get all existing subfolders with three-digit names
    existing_dirs = [
        name for name in os.listdir(result_dir) 
        if os.path.isdir(os.path.join(result_dir, name)) and name.isdigit() and len(name) == 3
    ]
    # Determine the next folder name
    next_index = max(map(int, existing_dirs), default=-1) + 1
    new_folder_name = f"{next_index:03}"
    new_folder_path = os.path.join(result_dir, new_folder_name)
    # Create the new folder
    os.makedirs(new_folder_path, exist_ok=True)
    
    # Save the objects to JSON files
    for obj, fn in zip(objects, file_names):
        obj_path = os.path.join(new_folder_path, fn)
        with open(obj_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=4)

    if (history_obj is not None) and (history_obj != []):
        for l, layer_history in enumerate(history_obj):
            hist_path = os.path.join(new_folder_path, f"layer-{l}.png")
            save_plot(layer_history, hist_path)
    
    print(f"Results saved in folder: {new_folder_path}")

def save_plot(history, path):
    entro_loss_history, ortho_loss_history, recon_loss_history = history
    plt.figure(figsize = (10, 8), dpi = 100)
    sns.lineplot(x = range(len(entro_loss_history)), y = entro_loss_history, zorder=1, label = "entropy")
    sns.lineplot(x = range(len(ortho_loss_history)), y = ortho_loss_history, zorder=1, label = "orthogonal")
    sns.lineplot(x = range(len(recon_loss_history)), y = recon_loss_history, zorder=1, label = "reconstruction")
    plt.legend(loc = "upper right")

    plt.savefig(path)

