import random
from transformers import AutoTokenizer
from typing import Dict, List
from tqdm import tqdm
from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import inverse_store, split_kn_dic, filter_by_entro
from .logging import init_logger
from .erase import Eraser

import tracemalloc, sys


def eval_erasing(
    model: nn.Module,
    tok: AutoTokenizer,
    kn_store: Dict,
    config: Dict,
    expose_matrix: object = None,
    method: str = "kv",
    logger: object = None,
):

    if logger is None:
        logger = init_logger(config['log_file'])
    device = config['device']
    kn_id_store = inverse_store(kn_store)
    kn_items, local_items = split_kn_dic(
        kn_dic = kn_id_store,
        num_local = config['num_local'] if 'num_local' in config else 100,
        seed = config['seed'],
    )
    logger.info("Selecting {} as locality evaluation cases.".format(local_items.keys()))
    if config['sample_rate'] < 1:
        kn_items = filter_by_entro(kn_items, config['sample_rate'], config['sample_file'])
    kn_items = list(kn_items.items())

    local_ans = tok(
        text = [value['target'] for _, value in local_items.items()], 
        padding = True,
        add_special_tokens = False, 
        return_tensors = "pt",
    )['input_ids'][:, 0].to(device)
    # Initialize entity-relation eraser
    eraser = Eraser(
        model = model,
        tok = tok,
        config = config,
        expose_matrix = expose_matrix,
    )

    case_metrics = []
    for case_id, case_dict in tqdm(kn_items):
        if method == "kn":
            old, new, old_local, new_local = eraser.erase_kn(
                case_dict['source'],
                list(zip(case_dict['layers'], case_dict['index'])),
                local_cases = local_items,
            )
        elif method == "kv":
            old, new, old_local, new_local = eraser.erase_kv(
                case_dict['source'],
                list(zip(case_dict['layers'], case_dict['layer_ids'])),
                X = case_dict['x'],
                Z = case_dict['z'],
                local_cases = local_items,
            )
        else:
            raise NotImplementedError
        
        ans = tok.encode(
            text = case_dict['target'], 
            add_special_tokens = False, 
            return_tensors="pt"
        )[0, 0].to(device)

        case_result = {
            'old': old, 
            'new': new, 
            'ans': ans,
            'old_local': old_local, 
            'new_local': new_local,
            'local_ans': local_ans,
        }
        logger.info("Case ID: %s, prompt: %s, predict before: %s predict after: %s" % (
            case_id,
            case_dict['source'],
            tok.decode(case_result['old'].argmax(-1)),
            tok.decode(case_result['new'].argmax(-1)),
        ))

        case_metric = dict()
        case_metric.update(erase_acc(case_result))
        case_metric.update(ppl(case_result))
        case_metric.update(locality(case_result))

        case_metrics.append(case_metric)

    result_dict = dict()
    for key in case_metrics[0].keys():
        if key in ("T2T", "T2F", "F2T", "F2F"):
            result_dict[key] = np.sum([line[key] for line in case_metrics])
        else:
            result_dict[key] = np.mean([line[key] for line in case_metrics])
    result_dict = summarize_acc(result_dict)

    logger.info(
        '\n'.join(
            ["{}: {:.4f}".format(key, value) 
            for (key, value) in result_dict.items()]
        )
    )

    return result_dict

def erase_acc(result):
    y2y = 0
    y2n = 0
    n2n = 0
    n2y = 0

    ans = result['ans']
    acc_old = result['old'].argmax() == ans
    acc_new = result['new'].argmax() == ans
    if acc_old and acc_new:
        y2y = 1
    elif acc_old and (not acc_new):
        y2n = 1
    elif (not acc_old) and (not acc_new):
        n2n = 1
    else:
        n2y = 1

    return {
        "T2T": y2y,
        "T2F": y2n,
        "F2T": n2y,
        "F2F": n2n,
    }

def summarize_acc(result, eps = 1e-8):
    T2T, T2F, F2T, F2F = result["T2T"], result["T2F"], result["F2T"], result["F2F"]
    reject_rate = (T2F + F2F) / (T2T + T2F + F2F + F2T + eps)
    erase_success = (T2F) / (T2T + T2F + eps)
    stay_rate = (F2F) / (F2F + F2T + eps)

    del result["T2T"], result["T2F"], result["F2T"], result["F2F"]
    result.update({
        "reject_rate": reject_rate,
        "erase_success": erase_success,
        "stay_rate": stay_rate,
    })

    return result
    

def ppl(result):
    ppl_old = -F.log_softmax(result['old'], -1)[result['ans']].item()
    ppl_new = -F.log_softmax(result['new'], -1)[result['ans']].item()
    
    return {
        "perplexity_acc": int(ppl_new > ppl_old),
        "perplexity_delta": ppl_new - ppl_old,
    }

def locality(result):
    ppl_old = -F.log_softmax(result['old_local'], -1).gather(-1, result['local_ans'].unsqueeze(1)).squeeze()
    ppl_new = -F.log_softmax(result['new_local'], -1).gather(-1, result['local_ans'].unsqueeze(1)).squeeze()
    
    return {
        "locality_perplexity_delta": torch.mean(abs(ppl_new - ppl_old)).item()
    }
        
def out_of_scope_acc(results):
    num_consistency, num_all = 0, 0
    for dic in results:
        logits_old = dic['old_local']
        logits_new = dic['new_local']
        pred_old = logits_old.argmax(-1)
        pred_new = logits_new.argmax(-1)
        num_consistency += (pred_old == pred_new).sum().item()
        num_all += torch.prod(torch.tensor(pred_old.size())).item()
    return {
        "locality_acc": num_consistency / num_all
    }


