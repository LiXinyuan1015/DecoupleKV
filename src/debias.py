import os
import time
from transformers import AutoTokenizer
from typing import Dict, List
from tqdm import tqdm
from collections import OrderedDict
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .discretize import DiscretizationModel
from .utils import get_attr, chunks, json_auto_save
from .logging import init_logger
from .evaluate import eval_erasing

def locate_kn_ig(
    model: nn.Module,
    tok: AutoTokenizer,
    config: Dict,
    source_target: List[Dict] = None,
    logger: object = None,
):
    if logger is None:
        logger = init_logger(config['log_file'])

    def scaled_input(emb, num_cuts):
        # emb: (1, ffn_size)
        baseline = torch.zeros_like(emb)  # (1, ffn_size)
        
        step = (emb - baseline) / num_cuts  # (1, ffn_size)

        res = torch.cat([torch.add(baseline, step * i) for i in range(num_cuts)], dim=0)  # (num_points, ffn_size)
        return res, step[0]
    
    def convert_to_dict_ig(ig_list, threshold = 0.75):
        ig = np.array(ig_list)  # 12, 3072
        ig_dict = {i: [] for i in range(ig.shape[0])}
        max_ig = ig.max()
        if max_ig > 0:
            for i in range(ig.shape[0]):
                for j in range(ig.shape[1]):
                    if ig[i][j] >= max_ig * threshold:
                        ig_dict[i].append([j, ig[i][j]])
        return ig_dict
    
    device = config['device']
    num_layers = config['num_layers']
    # name_attn = config['transformer.h.{}.attn']
    name_mlp = config['name_mlp']
    # name_emb = config['lm_head.weight']
    num_layers = config['num_layers']
    num_cuts = config['num_cuts']

    neurons = OrderedDict({i: [] for i in range(num_layers)})
    map_saved = False
    ig_maps = []
    for dic in tqdm(source_target):
        source, pro_target, anti_target = (
            dic['source'], 
            dic['pro_target'] if 'pro_target' in dic else dic['target'],
            dic['anti_target'] if 'anti_target' in dic else None,
        )
        inputs = tok([source], return_tensors="pt", add_special_tokens = False).to(device)
        repeated_inputs = tok(
            [source] * num_cuts, 
            return_tensors="pt", 
            add_special_tokens = False
        ).to(device)
        # last token
        if config['structure'] == 'uni':
            pos = inputs['attention_mask'][0].sum() - 1
        elif config['structure'] == 'bi':
            pos = torch.argwhere(inputs['input_ids'] == config['mask_id'])[0,-1]
        pro = tok.encode(
            text = pro_target, 
            add_special_tokens = False, 
            return_tensors="pt",
        )[0, 0].to(device)
        if anti_target:
            anti = tok.encode(
                text = anti_target, 
                add_special_tokens = False, 
                return_tensors="pt",
            )[0, 0].to(device)
        else:
            anti = None

        ig_list = []

        output = model(**inputs)
        logits = output.logits[0, pos, :]

        pro_losses = -logits.log_softmax(-1)[pro]
        anti_losses = -logits.log_softmax(-1)[anti]
        
        zs = []
        for layer in range(num_layers):
            mlp_block = get_attr(model, name_mlp.format(layer))
            z = mlp_block.cache['z']
            z.requires_grad_(True)
            zs += [z]
        pro_grads = torch.autograd.grad(pro_losses.mean(), zs, retain_graph=True)
        pro_grads = [-grad[0, pos, :].squeeze().clone().detach().cpu() for grad in pro_grads]

        if anti_target:
            anti_grads = torch.autograd.grad(anti_losses.mean(), zs)
            anti_grads = [-grad[0, pos, :].squeeze().clone().detach().cpu() for grad in anti_grads]
        else:
            anti_grads = [torch.empty(0)] * num_layers

        input_states = [get_attr(model, name_mlp.format(l)).cache['x'][..., pos, :] for l in range(num_layers)]
        intermediates = [get_attr(model, name_mlp.format(l)).cache['f(xK)'][..., pos, :] for l in range(num_layers)]

        for layer in range(num_layers):
            scaled_weights, weights_step = scaled_input(intermediates[layer], num_cuts)
            scaled_weights.requires_grad_(True)

            custom = {
                "custom_layer": layer,
                "custom_intermediate": scaled_weights,
                "custom_pos": pos,
            }
            
            outputs = model(**repeated_inputs, **custom)
            logits = outputs.logits[..., pos, :]
            tgt_prob = F.softmax(logits, -1)
            gradient = torch.autograd.grad(torch.unbind(tgt_prob[:, pro]), scaled_weights)[0]
            ig_grad = gradient.sum(dim = 0) * weights_step

            ig_list += [ig_grad.tolist()]

        if len(ig_maps) < 1500:
            ig_maps.append(ig_list)
        else:
            if not map_saved:
                map_dir = os.path.dirname(config['attribution_map_path'])
                if not os.path.exists(map_dir):
                    os.makedirs(map_dir)
                np.save(config['attribution_map_path'], np.array(ig_maps))
                map_saved = True

        ig_dict = convert_to_dict_ig(ig_list)

        for layer, ig_info_layer in ig_dict.items():
            cross_neurons = [x[0] for x in ig_info_layer]
            attributions = [x[1] for x in ig_info_layer]
            if len(cross_neurons) > 0:
                neurons[layer].append({
                    "source": source, 
                    "pro_target": pro_target,
                    "anti_target": anti_target,
                    "index": cross_neurons,
                    "attr": attributions,
                    "attr_list": ig_list[layer],
                    "case_id": dic['id'],
                    "states": input_states[layer].squeeze().cpu().tolist(),
                    "pro_grads": pro_grads[layer].tolist(),
                    "anti_grads": anti_grads[layer].tolist(),
                })
                
    return neurons

def locate_kn_ca(
    model: nn.Module,
    tok: AutoTokenizer,
    config: Dict,
    source_target: List[Dict] = None,
    logger: object = None,
):
    if logger is None:
        logger = init_logger(config['log_file'])
    device = config['device']
    num_layers = config['num_layers']
    # name_attn = config['transformer.h.{}.attn']
    name_mlp = config['name_mlp']
    # name_emb = config['lm_head.weight']
    chunk_size = config['chunk_size']
    ca_ppl_delta = config['ca_ppl_delta']

    neurons = OrderedDict({i: [] for i in range(num_layers)})
    for dics in tqdm(
        chunks(source_target, chunk_size), 
        total = len(source_target) // chunk_size + (1 if len(source_target) % chunk_size else 0)
    ):
        # first token of answer string
        source, pro_target, anti_target = (
            [dic['source'] for dic in dics], 
            [dic['pro_target'] for dic in dics],
            [dic['anti_target'] for dic in dics],
        )
        pro = tok(
            text = pro_target, 
            padding = True,
            add_special_tokens = False, 
            return_tensors="pt"
        )['input_ids'][:, 0].to(device)
        anti = tok(
            text = anti_target, 
            padding = True,
            add_special_tokens = False, 
            return_tensors="pt"
        )['input_ids'][:, 0].to(device)
        # run forward propagation
        inputs = tok(
            text = source, 
            padding = True,
            return_tensors="pt", 
            add_special_tokens = False
        ).to(device)
        if config['structure'] == 'uni':
            target_position = inputs['attention_mask'].sum(-1) - 1
        elif config['structure'] == 'bi':
            target_position = torch.argwhere(inputs['input_ids'] == config['mask_id'])[:,-1]
            assert target_position.size(0) == inputs['input_ids'].size(0), 'May be the number of [MASK] != 1? Source {}'.format(source)
        logits = model.forward(**inputs).logits[
            torch.arange(target_position.size(0), device = device), target_position, :
        ].squeeze(1)
        
        pro_losses = -logits[
            torch.arange(logits.size(0), device = device), pro
        ].log_softmax(-1)
        anti_losses = -logits[
            torch.arange(logits.size(0), device = device), anti
        ].log_softmax(-1)
        ppl_origin_model = pro_losses.clone().detach().cpu()
        
        zs = []
        for layer in range(num_layers):
            mlp_block = get_attr(model, name_mlp.format(layer))
            z = mlp_block.cache['z']
            z.requires_grad_(True)
            zs += [z]
        pro_grads = torch.autograd.grad(pro_losses.mean(), zs, retain_graph=True)
        pro_grads = [
            -grad[
                torch.arange(target_position.size(0), device = device), target_position, :
            ].squeeze().clone().detach().cpu() for grad in pro_grads
        ]
        anti_grads = torch.autograd.grad(anti_losses.mean(), zs)
        anti_grads = [
            -grad[
                torch.arange(target_position.size(0), device = device), target_position, :
            ].squeeze().clone().detach().cpu() for grad in anti_grads
        ]
        
        with torch.no_grad():
            for layer in range(num_layers):
                block_to_drop = get_attr(model, name_mlp.format(layer))
                block_to_drop.drop = True

                logits = model.forward(**inputs).logits[
                    torch.arange(target_position.size(0), device = device), target_position, :
                ].squeeze(1)
                ppl_drop_block = -logits[
                    torch.arange(logits.size(0), device = device), pro
                ].log_softmax(-1).cpu()

                states = block_to_drop.cache['x'][
                    torch.arange(target_position.size(0), device = device), target_position, :
                ].squeeze(1).cpu()
                block_to_drop.drop = False

                for ppl_old, ppl_new, dic, state, pro_grad, anti_grad in zip(
                    ppl_origin_model, ppl_drop_block, dics, states, pro_grads[layer], anti_grads[layer]
                ):
                    if (ppl_new - ppl_old).div(ppl_old) > ca_ppl_delta:
                        neurons[layer].append({
                            "source": dic['source'], 
                            "pro_target": dic['pro_target'],
                            "anti_target": dic['anti_target'],
                            "index": [],
                            "case_id": dic['id'],
                            "states": state.tolist(),
                            "pro_grads": pro_grad.tolist(),
                            "anti_grads": anti_grad.tolist(),
                        })

    return neurons

def discretize_kn(
    model: object,
    config: Dict,
    kn_store: Dict,
    logger: object = None,
):
    if logger is None:
        logger = init_logger(config['log_file'])
    num_layers = config['num_layers']
    device = config['device']
    
    model.eval()
    weights = []
    history = []
    for layer in range(num_layers):
        if config['skip_first_layer'] and layer == 0:
            weights += [(None, None, None)]
            continue
        if config['skip_last_layer'] and layer == num_layers - 1:
            weights += [(None, None, None)]
            continue
        layer_kn_store = kn_store[layer]
        config['C_rank'] = config['intermediate_size']
        discretizer = DiscretizationModel(
            module = get_attr(model, config['name_mlp'].format(layer)),
            config = config,
        )
        # No training
        logger.info(
            f"Start rewriting reweighting matrix C at layer {layer}, size: {str((discretizer.m, len(layer_kn_store)))}\n"
            f"Number of examples at layer {layer}: {len(layer_kn_store)}\n"
        )
        start_time = time.time()
        X = torch.cat([torch.tensor(line['states'], device=device).unsqueeze(0) for line in layer_kn_store], dim = 0)
        Y = torch.cat([torch.tensor(line['pro_grads'] if 'pro_grads' in line else line['grads'], device=device).unsqueeze(0) for line in layer_kn_store], dim = 0)
        if config['decomposite'] == 'V':
            C, _, _ = discretizer.rewrite_grad(Y)
        elif config['decomposite'] == 'K':
            C, _, _ = discretizer.rewrite(X)
        elif config['decomposite'] == 'KV':
            C, _, _ = discretizer.rewrite_dual(X, Y)
        Sigma_K, Sigma_V = discretizer.compute_coefficient(X, Y, C)
        end_time = time.time()
        logger.info(
            "Rewrtiting finish, which costs {:.4f}s.".format(end_time - start_time)
        )
        weights += [(C.cpu().numpy(), Sigma_K.cpu().numpy(), Sigma_V.cpu().numpy())]

    return weights, history

def debias_pipeline(
    model: nn.Module,
    tok: AutoTokenizer,
    config: Dict,
    source_target: List[Dict] = None,
    logger: object = None,
):
    if os.path.exists(config['neuron_path']):
        kn_store = torch.load(config['neuron_path'], weights_only=False)
    else:
        locate = {"ca": locate_kn_ca, "ig": locate_kn_ig}[config['locate_method']]
        kn_store = locate(
            model=model,
            tok=tok,
            config=config,
            source_target=source_target,
            logger=logger,
        )
        store_dir = os.path.dirname(config['neuron_path'])
        if not os.path.exists(store_dir):
            os.makedirs(store_dir)
        torch.save(kn_store, config['neuron_path'])

    logger.info("\n".join(
        ["layer {} detected: {}".format(layer, len(items)) for layer, items in kn_store.items()]
    ))
    
    if os.path.exists(config['weight_path']):
        pass
        '''
        weights_all = np.load(config['weight_path'], allow_pickle=True)
        weights = [weights_all[f"layer_{i}"] for i in range(config['num_layers'])]
        Sigma_K = [weights_all[f"layer_{i}_K"] for i in range(config['num_layers'])]
        Sigma_V = [weights_all[f"layer_{i}_V"] for i in range(config['num_layers'])]
        weights = list(zip(weights, Sigma_K, Sigma_V))
        '''
    else:
        weights, _ = discretize_kn(
            model=model,
            config=config,
            kn_store=kn_store,
            logger=logger
        )
        '''
        weights_to_save = {f"layer_{i}": weights[i][0] for i in range(len(weights))}
        Sigma_K_to_save = {f"layer_{i}_K": weights[i][1] for i in range(len(weights))}
        Sigma_V_to_save = {f"layer_{i}_V": weights[i][2] for i in range(len(weights))}
        weights_to_save.update(Sigma_K_to_save)
        weights_to_save.update(Sigma_V_to_save)
        weights_dir = os.path.dirname(config['weight_path'])
        if not os.path.exists(weights_dir):
            os.makedirs(weights_dir)
        np.savez(config['weight_path'], **weights_to_save)
        '''
        

    results = eval_erasing(
        model = model,
        tok = tok,
        kn_store = kn_store,
        config = config,
        expose_matrix = weights,
        method = config['erase_method'],
        logger = logger,
    )
    json_auto_save(
        result_dir = config['result_path'],
        objects = [results, config],
        file_names = ['results.json', 'config.json'],
    )