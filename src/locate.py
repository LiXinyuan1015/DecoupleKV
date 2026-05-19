import json
from transformers import AutoTokenizer
from typing import Dict, List
from tqdm import tqdm
from collections import OrderedDict
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import cross_entropy, entropy, get_attr, chunks, znorm
from .logging import init_logger

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
    
    def convert_to_dict_ig(ig_list, threshold = 0.5):
        ig = np.array(ig_list)  # 12, 3072
        ig_dict = {i: [] for i in range(ig.shape[0])}
        entros = []
        max_ig = ig.max()
        if max_ig > 0:
            for i in range(ig.shape[0]):
                attr_norm = znorm(torch.tensor(ig[i]).unsqueeze(0))
                entros.append(entropy(attr_norm).cpu().item())
                for j in range(ig.shape[1]):
                    if ig[i][j] >= max_ig * threshold:
                        ig_dict[i].append([j, ig[i][j]])
        return ig_dict, entros
    
    device = config['device']
    num_layers = config['num_layers']
    # name_attn = config['transformer.h.{}.attn']
    name_mlp = config['name_mlp']
    # name_emb = config['lm_head.weight']
    num_layers = config['num_layers']
    num_cuts = config['num_cuts']

    neurons = OrderedDict({i: [] for i in range(num_layers)})
    for dic in tqdm(source_target):
        source, target = dic['source'], dic['target']
        inputs = tok([source], return_tensors="pt", add_special_tokens = False).to(device)
        repeated_inputs = tok(
            [source] * num_cuts, 
            return_tensors="pt", 
            add_special_tokens = False
        ).to(device)
        # last token
        pos = inputs['attention_mask'][0].sum() - 1
        gold = tok.encode(
            text = target, 
            add_special_tokens = False, 
            return_tensors="pt",
        )[0, 0].to(device)

        ig_list = []

        model(**inputs)
        input_states = [get_attr(model, name_mlp.format(l)).cache['x'][..., -1, :] for l in range(num_layers)]
        intermediates = [get_attr(model, name_mlp.format(l)).cache['f(xK)'][..., -1, :] for l in range(num_layers)]

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
            gradient = torch.autograd.grad(torch.unbind(tgt_prob[:, gold]), scaled_weights)[0]
            ig_grad = gradient.sum(dim = 0) * weights_step

            ig_list += [ig_grad.tolist()]

        ig_dict, entros = convert_to_dict_ig(ig_list)

        for layer, ig_info_layer in ig_dict.items():
            cross_neurons = [x[0] for x in ig_info_layer]
            attributions = [x[1] for x in ig_info_layer]
            if len(cross_neurons) > 0:
                neurons[layer].append({
                    "source": source, 
                    "target": target,
                    "index": cross_neurons,
                    "attr": attributions,
                    "case_id": dic['id'],
                    "states": input_states[layer].squeeze().cpu().tolist(),
                    "entropy": entros[layer],
                })

    return neurons
            
        

def locate_kn(
    model: nn.Module,
    tok: AutoTokenizer,
    config: Dict,
    source_target: List[Dict] = None,
    logger: object = None,
):
    if logger is None:
        logger = init_logger(config['log_file'])
    device = config['device']
    top_num = config['top_num']
    num_layers = config['num_layers']
    # name_attn = config['transformer.h.{}.attn']
    name_mlp = config['name_mlp']
    # name_emb = config['lm_head.weight']
    subname_K = config['subname_K']
    subname_V = config['subname_V']

    neurons = OrderedDict({i: [] for i in range(num_layers)})
    for dic in tqdm(source_target):
        # first token of answer string
        source, target = dic['source'], dic['target']
        gold = tok.encode(
            text = target, 
            add_special_tokens = False, 
            return_tensors="pt"
        )[0, 0].to(device)
        # run forward propagation
        model.eval()
        outputs = model.forward(
            **tok([source], return_tensors="pt", add_special_tokens = False).to(device), 
        )
        logits = outputs.logits
        prediction = tok.decode(logits.max(-1)[1][..., -1])
        loss = cross_entropy(logits[..., -1, :], gold)
        logger.info("LOSS: {:.4f}, PROB: {:.4f}, MODEL PREDICTION: {}".format(loss.item(), torch.exp(-loss).item(), prediction))
        # collect nodes require gradient
        zs = []
        for layer in range(num_layers):
            mlp_block = get_attr(model, name_mlp.format(layer))
            z = mlp_block.cache['z']
            fxK = mlp_block.cache['f(xK)']
            z.requires_grad_(True)
            fxK.requires_grad_(True)
            zs += [z]
        # compute gradients
        grads = torch.autograd.grad(loss, zs)
        # compute activation neurons
        for layer, dz in enumerate(grads):
            mlp_block = get_attr(model, name_mlp.format(layer))
            x = mlp_block.cache['x'][..., -1, :]
            dz = -dz[..., -1, :]
            K = get_attr(mlp_block, subname_K)
            V = get_attr(mlp_block, subname_V)
            # adjust shape of matrices
            if isinstance(K, nn.Linear) and isinstance(V, nn.Linear):
                K = K.weight.transpose(0,1)
                V = V.weight.transpose(0,1)
            else:
                K = K.weight
                V = V.weight
            # get intermediate states directly
            intermediate = mlp_block.cache['f(xK)'][..., -1, :]
            # compute top-k activated neurons
            mKs = torch.topk(intermediate, dim = -1, k = top_num)[1].tolist()
            mVs = torch.topk(dz @ V.T, dim = -1, k = top_num)[1].tolist()
            # collect cross neurons between xK and Vy
            cross_neurons = list(set(mKs[0]) & set(mVs[0]))
            if cross_neurons:
                neurons[layer].append({
                    "source": source, 
                    "target": target,
                    "index": cross_neurons,
                    "case_id": dic['id'],
                    "states": x.squeeze().cpu().tolist(),
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
        source, target = [dic['source'] for dic in dics], [dic['target'] for dic in dics]
        gold = tok(
            text = target, 
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
        last_non_mask = inputs['attention_mask'].sum(-1) - 1
        logits = model.forward(**inputs).logits[
            torch.arange(last_non_mask.size(0), device = device), last_non_mask, :
        ].squeeze(1)
        
        losses = -logits[
            torch.arange(logits.size(0), device = device), gold
        ].log_softmax(-1)
        ppl_origin_model = losses.clone().detach().cpu()
        
        zs = []
        for layer in range(num_layers):
            mlp_block = get_attr(model, name_mlp.format(layer))
            z = mlp_block.cache['z']
            z.requires_grad_(True)
            zs += [z]
        grads = torch.autograd.grad(losses.mean(), zs)
        grads = [
            -grad[
                torch.arange(last_non_mask.size(0), device = device), last_non_mask, :
            ].squeeze().clone().detach().cpu() for grad in grads
        ]
        
        with torch.no_grad():
            for layer in range(num_layers):
                block_to_drop = get_attr(model, name_mlp.format(layer))
                block_to_drop.drop = True

                logits = model.forward(**inputs).logits[
                    torch.arange(last_non_mask.size(0), device = device), last_non_mask, :
                ].squeeze(1)
                ppl_drop_block = -logits[
                    torch.arange(logits.size(0), device = device), gold
                ].log_softmax(-1).cpu()

                states = block_to_drop.cache['x'][
                    torch.arange(last_non_mask.size(0), device = device), last_non_mask, :
                ].squeeze(1).cpu()
                block_to_drop.drop = False

                for ppl_old, ppl_new, dic, state, grad in zip(ppl_origin_model, ppl_drop_block, dics, states, grads[layer]):
                    if (ppl_new - ppl_old).div(ppl_old) > ca_ppl_delta:
                        neurons[layer].append({
                            "source": dic['source'], 
                            "target": dic['target'],
                            "index": [],
                            "case_id": dic['id'],
                            "states": state.tolist(),
                            "grads": grad.tolist(),
                        })
    logger.info("\n".join(
        ["layer {} detected: {}".format(layer, len(items)) for layer, items in neurons.items()]
    ))

    return neurons