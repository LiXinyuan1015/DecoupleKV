import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from typing import Dict
from tqdm import tqdm
from transformers import AutoTokenizer

from .utils import get_attr, chunks
from .logging import init_logger

class Eraser(nn.Module):
    def __init__(
        self,
        model,
        tok,
        config,
        expose_matrix = None,
    ):
        super().__init__()
        self.model = model
        self.tok = tok
        self.expose_matrix = [line[0] for line in expose_matrix]
        self.Sigma_K = [line[1] for line in expose_matrix]
        self.Sigma_V = [line[2] for line in expose_matrix]
        self.config = config
        
    @torch.no_grad()
    def erase_kn(self, source, position, local_cases = None, recover = True, inf = 1e4):
        device = self.config['device']
        # forward propagation before erasing
        input_dict = self.tok(
            [source], 
            return_tensors="pt", 
            add_special_tokens = False,
        ).to(device)
        outputs_old = self.model.forward(**input_dict).logits[..., target_position, :].squeeze()
        outputs_new = None
        # forward propagation with local prompt
        if local_cases:
            outputs_old_local = []
            local_items = chunks(list(local_cases.items()), chunk_size=self.config['chunk_size'])
            for chunk_items in local_items:
                local_dict = self.tok(
                    [item['source'] for (_, item) in chunk_items], 
                    return_tensors="pt", 
                    padding = True,
                    max_length = 256,
                    truncation = True,
                    add_special_tokens = False
                ).to(device)
                if self.config['structure'] == 'uni':
                    target_position = local_dict['attention_mask'].sum(-1) - 1
                elif self.config['structure'] == 'bi':
                    target_position = torch.argwhere(local_dict['input_ids'] == self.config['mask_id'])[:,-1]
                logits = self.model.forward(**local_dict).logits[
                    torch.arange(target_position.size(0), device = target_position.device), target_position, :
                ].squeeze(1)
                outputs_old_local.append(logits)
            outputs_old_local = torch.cat(outputs_old_local, dim = 0)
        
        KV_cache, module = [], None
        for l, idxs in position:
            if self.config['skip_first_layer'] and l == 0:
                continue
            if self.config['skip_last_layer'] and l == self.config['num_layers'] - 1:
                continue
            # get module at layer l
            module = get_attr(self.model, self.config['name_mlp'].format(l))
            # get weights and intermediate states of MLP module
            K = get_attr(module, self.config['subname_K'])
            V = get_attr(module, self.config['subname_V'])
            # compute intermediate states with C
            is_linear = isinstance(K, nn.Linear)
            KV_cache += [(l, K.weight.clone(), K.bias.clone(), V.weight.clone())]
            idxs = torch.tensor(idxs, device = device).long()
            if is_linear:
                V.weight.data[:, idxs] = 0.
                K.weight.data[idxs, :] = 0.
            else:
                K.weight.data[:, idxs] = 0.
                V.weight.data[idxs, :] = 0.
            K.bias.data[idxs] = 0.
            
        # forward propagation after erasing
        outputs_new = self.model.forward(
            **input_dict
        ).logits[..., target_position, :].squeeze()
        outputs = (outputs_old, outputs_new)
        if local_cases:
            outputs_new_local = []
            local_items = chunks(list(local_cases.items()), chunk_size=self.config['chunk_size'])
            for chunk_items in local_items:
                local_dict = self.tok(
                    [item['source'] for (_, item) in chunk_items], 
                    return_tensors="pt", 
                    padding = True,
                    max_length = 256,
                    truncation = True,
                    add_special_tokens = False
                ).to(device)
                if self.config['structure'] == 'uni':
                    target_position = local_dict['attention_mask'].sum(-1) - 1
                elif self.config['structure'] == 'bi':
                    target_position = torch.argwhere(local_dict['input_ids'] == self.config['mask_id'])[:,-1]
                logits = self.model.forward(**local_dict).logits[
                    torch.arange(target_position.size(0), device = target_position.device), target_position, :
                ].squeeze(1)
                outputs_new_local.append(logits)
            outputs_new_local = torch.cat(outputs_new_local, dim = 0)
            outputs += (outputs_old_local, outputs_new_local)
        # recover K,V's weight if necessary
        for l, K_weight_data, K_bias_data, V_weight_data in KV_cache:
            module = get_attr(self.model, self.config['name_mlp'].format(l))
            K = get_attr(module, self.config['subname_K'])
            V = get_attr(module, self.config['subname_V'])
            K.weight.copy_(K_weight_data)
            K.bias.copy_(K_bias_data)
            V.weight.copy_(V_weight_data)
                    
        # release cuda cache manually
        del KV_cache,
        torch.cuda.empty_cache()

        return outputs


    @torch.no_grad()
    def erase_kv(self, source, position, X = None, Z = None, local_cases = None, inf = 1e4):
        device = self.config['device']
        # forward propagation before erasing
        input_dict = self.tok(
            [source], 
            return_tensors="pt", 
            add_special_tokens = False,
        ).to(device)
        if self.config['structure'] == 'uni':
            target_position = input_dict['attention_mask'].sum() - 1
        elif self.config['structure'] == 'bi':
            target_position = torch.argwhere(input_dict['input_ids'] == self.config['mask_id'])[0,-1]
        outputs_old = self.model.forward(**input_dict).logits[..., target_position, :].squeeze()
        outputs_new = None
        # cache hidden states of source text
        pre_cache = {}
        for l, _ in position:
            module = get_attr(self.model, self.config['name_mlp'].format(l))
            pre_cache[l] = module.cache['x'][..., target_position, :]
        # forward propagation with local prompt
        if local_cases:
            outputs_old_local = []
            local_items = chunks(list(local_cases.items()), chunk_size=self.config['chunk_size'])
            for chunk_items in local_items:
                local_dict = self.tok(
                    [item['source'] for (_, item) in chunk_items], 
                    return_tensors="pt", 
                    padding = True,
                    max_length = 256,
                    truncation = True,
                    add_special_tokens = False
                ).to(device)
                if self.config['structure'] == 'uni':
                    local_target_position = local_dict['attention_mask'].sum(-1) - 1
                elif self.config['structure'] == 'bi':
                    local_target_position = torch.argwhere(local_dict['input_ids'] == self.config['mask_id'])[:,-1]
                logits = self.model.forward(**local_dict).logits[
                    torch.arange(local_target_position.size(0), device = local_target_position.device), local_target_position, :
                ].squeeze(1)
                outputs_old_local.append(logits)
            outputs_old_local = torch.cat(outputs_old_local, dim = 0)
        
        KV_cache, C, module = [], None, None
        for l, idx in position:
            if self.config['skip_first_layer'] and l == 0:
                continue
            if self.config['skip_last_layer'] and l == self.config['num_layers'] - 1:
                continue
            # get C at layer l
            module = get_attr(self.model, self.config['name_mlp'].format(l))
            C = torch.from_numpy(self.expose_matrix[l]).to(device)
            # get weights and intermediate states of MLP module
            K = get_attr(module, self.config['subname_K'])
            V = get_attr(module, self.config['subname_V'])

            # compute intermediate states with C
            is_linear = isinstance(K, nn.Linear)
            
            # cache K, V 
            KV_cache += [(l, K.weight.detach().clone(), K.bias.detach().clone(), V.weight.detach().clone())]

            if X is None or Z is None:
                x = pre_cache[l]
                xK = F.linear(
                    input = x,
                    weight = C.T @ K.weight.data if is_linear else K.weight.data @ C,
                    bias = K.bias.data @ C
                )
                kn = xK.argmax(-1).squeeze().item()
                c = C[:,kn].unsqueeze(1)
                I = torch.eye(C.size(0), device = C.device)
                M = I - c @ c.T
                if is_linear:
                    if 'V' in self.config['modify_weight']:
                        V.weight.copy_(V.weight.data @ M)
                    if 'K' in self.config['modify_weight']:
                        K.weight.copy_(M @ K.weight.data)
                        K.bias.copy_(K.bias.data @ M)
                else:
                    if 'V' in self.config['modify_weight']:
                        V.weight.copy_(M @ V.weight.data)
                    if 'K' in self.config['modify_weight']:
                        K.weight.copy_(K.weight.data @ M)
                        K.bias.copy_(K.bias.data @ M)
            else:
                x = torch.tensor(X[l], device=device)
                z = torch.tensor(Z[l], device=device)
                # xK = x @ K.weight.T @ C
                # zV = z @ V.weight @ C
                # kn = xK.argmax(-1).squeeze().item()
                kn = idx
                c = C[:,kn].unsqueeze(1)
                key_map = c @ x.unsqueeze(0)
                value_map = z.unsqueeze(1) @ c.T
                sigma_K = self.Sigma_K[l][kn]
                sigma_V = self.Sigma_V[l][kn]
                K.weight.copy_(K.weight.data - sigma_K * key_map)
                V.weight.copy_(V.weight.data - sigma_V * value_map)
            
        # forward propagation after erasing
        outputs_new = self.model.forward(
            **input_dict
        ).logits[..., target_position, :].squeeze()
        outputs = (outputs_old, outputs_new)
        if local_cases:
            outputs_new_local = []
            local_items = chunks(list(local_cases.items()), chunk_size=self.config['chunk_size'])
            for chunk_items in local_items:
                local_dict = self.tok(
                    [item['source'] for (_, item) in chunk_items], 
                    return_tensors="pt", 
                    padding = True,
                    max_length = 256,
                    truncation = True,
                    add_special_tokens = False
                ).to(device)
                if self.config['structure'] == 'uni':
                    target_position = local_dict['attention_mask'].sum(-1) - 1
                elif self.config['structure'] == 'bi':
                    target_position = torch.argwhere(local_dict['input_ids'] == self.config['mask_id'])[:,-1]
                logits = self.model.forward(**local_dict).logits[
                    torch.arange(target_position.size(0), device = target_position.device), target_position, :
                ].squeeze(1)
                outputs_new_local.append(logits)
            outputs_new_local = torch.cat(outputs_new_local, dim = 0)
            outputs += (outputs_old_local, outputs_new_local)
        # recover K,V's weight if necessary
        for l, K_weight_data, K_bias_data, V_weight_data in KV_cache:
            module = get_attr(self.model, self.config['name_mlp'].format(l))
            K = get_attr(module, self.config['subname_K'])
            V = get_attr(module, self.config['subname_V'])
            K.weight.copy_(K_weight_data)
            K.bias.copy_(K_bias_data)
            V.weight.copy_(V_weight_data)

        del KV_cache, C
        torch.cuda.empty_cache()
                
        return outputs
