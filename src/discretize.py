import math
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from typing import Dict
from tqdm import tqdm
from transformers import AutoTokenizer

from .utils import get_attr, chunks, cross_entropy
from .linalg_utils import solve_Procrustes, gram_schmidt_complete, complete_orthogonal_matrix, solve_Procrustes_multi
from .logging import init_logger

class DiscretizationModel(nn.Module):
    def __init__(self, module, config) -> None:
        super().__init__()
        self.m = config['intermediate_size']
        self.r = config['C_rank']
        device = config['device']
        if not config['apply_model_rewriting']:
            self.C = nn.Parameter(torch.randn(self.m, self.r, device = device) / math.sqrt(m))
        self.module = module
        self.device = device
        self.ortho_coef = config['ortho_coef'] if 'ortho_coef' in config else 1.
        self.entro_coef = config['entro_coef'] if 'entro_coef' in config else 1.
        self.recon_coef = config['recon_coef'] if 'recon_coef' in config else 1.
        self.lam = config['lambda'] if 'lambda' in config else 1.

        self.subname_K = config['subname_K']
        self.subname_V = config['subname_V']
        self.subname_act = config['subname_act']

    def reweighted_forward(self, x, C = None):
        if C is None:
            output = self.module(x)
            inter = self.module.cache['f(xK)']
            return output, inter
        
        K = get_attr(self.module, self.subname_K)
        V = get_attr(self.module, self.subname_V)
        act = get_attr(self.module, self.subname_act)
        is_linear = isinstance(K, nn.Linear)

        inter = F.linear(
            input = x,
            weight = C.T @ K.weight if is_linear else K.weight @ C,
            bias = K.bias @ C
        )
        inter = act(inter)
        output = F.linear(
            input = inter,
            weight = V.weight @ C if is_linear else C.T @ K.weight,
            bias = V.bias
        )
        
        return output, inter

    # discretization loss
    def discretization(self, x, eps = 1e-8):
        p = F.softmax(x, dim = -1)
        l = -torch.diag(p @ torch.log(p + eps).T).mean()
        return l
    
    # index-wise discretization loss
    def index_wise_discretization(self, x, y):
        return cross_entropy(x * self.lam, y)
    
    # reconstruction loss for evaluation
    def reconstruction(self, X, Z):
        with torch.no_grad():
            Y = self.module(X)
            # kl_div = F.kl_div(
            #     F.log_softmax(Z, -1), 
            #     F.softmax(Y, -1),
            #     reduction = "batchmean"
            # )
            div = (Y - Z).square().sum().sqrt()
        return div
    
    # orthogonal reconstruction
    def orthogonalization(self, Q):
        size = Q.size(1)
        I = torch.eye(size, device=self.device)
        R = Q.T @ Q - I
        l = R.square().sum().sqrt()
        return l
    
    # forward
    def forward(self, X, idxs = None, output_recon = False):
        Z, I = self.reweighted_forward(X, self.C)
        if idxs is not None:
            # index-wise discretization
            entro = self.index_wise_discretization(I, idxs) * self.entro_coef
        else:
            entro = self.discretization(I) * self.entro_coef
        ortho = self.orthogonalization(self.C) * self.ortho_coef
        outputs = (entro, ortho)
        if output_recon:
            recon = self.reconstruction(X, Z) * self.recon_coef
            outputs += (recon,)
        return outputs
    
    def rewrite(self, X):
        K = get_attr(self.module, self.subname_K)
        if isinstance(K, nn.Linear):
            K = K.weight.data.T
        else:
            K = K.weight.data
        # minimize Tr(XKC), given C^TC=I
        C = solve_Procrustes(K, X.T)
        # C = complete_orthogonal_matrix(C)
        return C, X, K
    
    def rewrite_grad(self, Y):
        V = get_attr(self.module, self.subname_V)
        if isinstance(V, nn.Linear):
            V = V.weight.data.T
        else:
            V = V.weight.data
        # minimize Tr(XKC), given C^TC=I
        C = solve_Procrustes(V.T, Y.T)
        # C = complete_orthogonal_matrix(C)
        return C, Y, V
    
    def rewrite_dual(self, X, Y):
        K = get_attr(self.module, self.subname_K)
        V = get_attr(self.module, self.subname_V)
        if isinstance(V, nn.Linear):
            K = K.weight.data.T
            V = V.weight.data.T
        else:
            K = K.weight.data
            V = V.weight.data
        # minimize Tr(XKC + YV.TC), given C^TC=I
        C = solve_Procrustes_multi(K, X.T, V.T, Y.T)
        # C = complete_orthogonal_matrix(C)
        return C, Y, V
    
    def compute_coefficient(self, X, Y, C):
        K = get_attr(self.module, self.subname_K)
        V = get_attr(self.module, self.subname_V)
        if isinstance(V, nn.Linear):
            K = K.weight.data.T
            V = V.weight.data.T
        else:
            K = K.weight.data
            V = V.weight.data
        # (n,d) * (d.m) * (m,n)
        Sigma_K = torch.diag(X @ K @ C) / torch.diag(X @ X.T)
        # (n,m) * (m,d) * (d,n)
        Sigma_V = torch.diag(C.T @ V @ Y.T) / torch.diag(Y @ Y.T)
        return Sigma_K, Sigma_V


def discretize_kn(
    model: object,
    tok: AutoTokenizer,
    config: Dict,
    kn_store: Dict,
    logger: object = None,
):
    if logger is None:
        logger = init_logger(config['log_file'])
    num_layers = config['num_layers']
    chunk_size = config['chunk_size']
    num_epochs = config['num_epochs']
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
        if config['low_rank']:
            config['C_rank'] = int(1 << math.ceil(math.log2(len(layer_kn_store))))
            assert config['C_rank'] >= len(layer_kn_store), f"{config['C_rank']} < {len(layer_kn_store)}!!!"
        else:
            config['C_rank'] = config['intermediate_size']
        discretizer = DiscretizationModel(
            module = get_attr(model, config['name_mlp'].format(layer)),
            config = config,
        )

        # No training
        if config['apply_model_rewriting']:
            logger.info(
                f"Start rewriting reweighting matrix C at layer {layer}, size: {str((discretizer.m, len(layer_kn_store)))}\n"
                f"Number of examples at layer {layer}: {len(layer_kn_store)}\n"
                f"C will be expanded as {str((discretizer.m, discretizer.m))} via completement orthogonalization"
            )
            start_time = time.time()

            X = torch.stack([torch.tensor(line['states'], device=device) for line in layer_kn_store], dim = 0)
            Y = torch.stack([torch.tensor(line['grads'], device=device) for line in layer_kn_store], dim = 0)
            
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
            continue
        # gradient descending
        else:
            logger.info(
                f"Start training reweighting matrix C at layer {layer}, size: {str(tuple(discretizer.C.size()))}\n"
                f"Number of examples at layer {layer}: {len(layer_kn_store)}"
            )
            opt = optim.Adam(
                [
                    {'params': discretizer.C},
                ],
                lr = config['lr'],
            )
            num_chunks = math.ceil(len(layer_kn_store) / chunk_size)
            entro_layer_history, ortho_layer_history, recon_layer_history = [], [], []
            for i in tqdm(range(num_epochs)):
                # minibatch training
                entro_epoch, ortho_epoch, recon_epoch = 0., 0., 0.
                minibatch_idx = 0
                for chunk in chunks(layer_kn_store, chunk_size):
                    X = torch.cat([torch.tensor(line['states'], device=device) for line in chunk], dim = 0)
                    if config['index_wise']:
                        idxs = torch.arange(
                            minibatch_idx * chunk_size,
                            minibatch_idx * chunk_size + len(chunk), 
                            device = device,
                        )
                    else:
                        idxs = None
                    entro, ortho, recon = discretizer(X, idxs = idxs, output_recon = True)
                    # entro, ortho, recon = entro / num_chunks, ortho / num_chunks, recon / num_chunks
                    # print(entro.item(), ortho.item(), num_chunks)

                    entro_epoch += entro.item() / num_chunks
                    ortho_epoch += ortho.item() / num_chunks
                    recon_epoch += recon.item() / num_chunks

                    loss = (entro + ortho)
                    loss.backward()
                    opt.step()
                    opt.zero_grad()
                    minibatch_idx += 1
                entro_layer_history.append(entro_epoch)
                ortho_layer_history.append(ortho_epoch)
                recon_layer_history.append(recon_epoch)
                # minibatch training
                # opt.step()
                # opt.zero_grad()
                
            logger.info("Finish reweighting at layer {} , Loss: {:.4f}({:.4f}, {:.4f}, {:.4f}) -> {:.4f}({:.4f}, {:.4f}, {:.4f})".format(
                layer,
                sum([entro_layer_history[0], ortho_layer_history[0], recon_layer_history[0]]), 
                entro_layer_history[0], ortho_layer_history[0], recon_layer_history[0],
                sum([entro_layer_history[-1], ortho_layer_history[-1], recon_layer_history[-1]]), 
                entro_layer_history[-1], ortho_layer_history[-1], recon_layer_history[-1],
            ))
            history.append((entro_layer_history, ortho_layer_history, recon_layer_history))
            weights += [discretizer.C.data.cpu().numpy()]
            del discretizer
            logger.info("Emptying cuda cache...")
            torch.cuda.empty_cache()

    return weights, history
