import math
import torch
import torch.nn as nn

from collections import OrderedDict

from .loss import *
from .neurals import *

class ModelOutput(OrderedDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key in sorted(self.keys()):
            setattr(self, key, self[key])

class VitConfig(object):
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

example_config = VitConfig({
    "image_size": 256,
    "image_channels": 3, 
    "num_labels": 10,
    "patch_size": 16,
    "d_model": 768, 
    "d_intermediate": 3072, 
    "nlayers": 6, 
    "nheads": 8, 
    "dropout_rate": 0.1, 
    "layernorm_eps": 1e-10,
    "initializer_range": 2e-2,
    "label_smoothing": 0.1,
    "activation": "gelu",
    "flatten": "normal",
    "interpolate": "bicubic",
    "device": "cuda",
})

class PatchEmbedding(nn.Module):
    def __init__(self, image_size, image_channels, patch_size, d_model, flatten):
        super(PatchEmbedding, self).__init__()

        self.flatten = flatten
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divided by patch_size")
        self.patch_num = (image_size // patch_size) * (image_size // patch_size)
        self.conv_proj = nn.Conv2d(image_channels, d_model, patch_size, patch_size)

    def normal_flatten(self, image):
        return image.view(image.size(0), image.size(1), -1).permute([0,2,1])

    def forward(self, image):
        image = self.conv_proj(image)
        feature = self.normal_flatten(image)
        return feature
    
class VitEmbedding(nn.Module):
    def __init__(self, config):
        super(VitEmbedding, self).__init__()
        
        self.patch_embedding = PatchEmbedding(config.image_size, config.image_channels, config.patch_size, config.d_model, config.flatten)
        self.position_embedding = nn.Parameter(torch.zeros((1, self.patch_embedding.patch_num, config.d_model)))
        # self.cls_embedding = nn.Parameter(torch.zeros((1, 1, config.d_model)))
        self.dropout = nn.Dropout(config.dropout_rate)
    
    def forward(self, image):
        embedding = self.patch_embedding(image)
        # expand_cls = self.cls_embedding.repeat([image.size(0),1,1])
        # embedding = torch.cat([expand_cls, embedding], 1)

        embedding = embedding + self.position_embedding
        embedding = self.dropout(embedding)
        return embedding
    
class VitSelfAttention(nn.Module):
    def __init__(self, d_model, nheads, dropout_rate):
        super(VitSelfAttention, self).__init__()

        if d_model % nheads != 0:
            raise ValueError("d_model must be divided by nheads")

        self.d_model = d_model
        self.nheads = nheads

        self.linear_query = nn.Linear(d_model, d_model)
        self.linear_key = nn.Linear(d_model, d_model)
        self.linear_value = nn.Linear(d_model, d_model)
        self.linear_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(-1)

    def attn_reshape(self, hidden_states):
        return hidden_states.view(hidden_states.size(0), hidden_states.size(1), self.nheads, -1).permute([0,2,1,3])
    
    def attn_unshape(self, hidden_states):
        return hidden_states.permute([0,2,1,3]).contiguous().view(hidden_states.size(0), hidden_states.size(2), -1)
    
    def forward(self, hidden_states):
        query = self.attn_reshape(self.linear_query(hidden_states))
        key = self.attn_reshape(self.linear_key(hidden_states))
        value = self.attn_reshape(self.linear_value(hidden_states))

        attn_output = query.matmul(key.permute([0,1,3,2]))
        attn_prob = self.softmax(attn_output / math.sqrt(self.d_model))
        attn_prob = self.dropout(attn_prob)

        output = attn_prob.matmul(value)
        output = self.attn_unshape(output)
        output = self.linear_proj(output)

        return output
    
class VitMLP(nn.Module):
    def __init__(self, d_model, d_intermediate, dropout_rate = 0.1, activation = "gelu"):
        super(VitMLP, self).__init__()

        self.activation = actv[activation]
        self.fc_in = nn.Linear(d_model, d_intermediate)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc_out = nn.Linear(d_intermediate, d_model)

    def forward(self, hidden_states):
        intermediate = self.activation(self.fc_in(hidden_states))
        output = self.dropout(self.fc_out(intermediate))
        return output

class VitPlugin(nn.Module):
    def __init__(self, d_model, d_intermediate, dropout_rate = 0.1, activation = "gelu", eps = 1e-10):
        super(VitPlugin, self).__init__()
        self.post_norm = nn.LayerNorm(d_model, eps = eps)
        self.activation = actv[activation]
        self.fc_in = nn.Linear(d_model, d_intermediate, bias=False)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc_out = nn.Linear(d_intermediate, d_model, bias=False)
        self.cache = dict({"K0": self.fc_in.weight.clone(), "V0": self.fc_out.weight.clone()})

    def enable_training(self):
        self.requires_grad_(True)

    def forward(self, hidden_states, return_inter = False, custom_inter = None):
        residual = hidden_states
        self.cache['x'] = hidden_states
        intermediate = self.activation(self.fc_in(hidden_states))
        if custom_inter is not None:
            intermediate[:,-1,:] = custom_inter
        z = self.dropout(self.fc_out(intermediate))
        self.cache['z'] = z
        output = self.post_norm(z + residual)
        if return_inter:
            output = (output, intermediate[:,-1,:])
        return output
    
class VitLayer(nn.Module):
    def __init__(self, config):
        super(VitLayer, self).__init__()

        self.self_attn = VitSelfAttention(config.d_model, config.nheads, config.dropout_rate)
        self.mlp = VitMLP(config.d_model, config.d_intermediate, config.dropout_rate, config.activation)

        self.layer_norm_1 = nn.LayerNorm(config.d_model, eps = config.layernorm_eps)
        self.layer_norm_2 = nn.LayerNorm(config.d_model, eps = config.layernorm_eps)

        self.plugin = None

    def forward(self, hidden_states, return_inter = False, custom_inter = None):
        hidden_states_norm = self.layer_norm_1(hidden_states)
        if self.plugin is not None:
            hidden_states_norm = self.plugin(hidden_states_norm, return_inter, custom_inter)
        attention_output = self.self_attn(hidden_states_norm)

        hidden_states = hidden_states + attention_output

        hidden_states_norm = self.layer_norm_2(hidden_states)
        linear_output = self.mlp(hidden_states_norm)

        hidden_states = hidden_states + linear_output

        return hidden_states
    
class VitEncoder(nn.Module):
    def __init__(self, config):
        super(VitEncoder, self).__init__()

        self.layers = nn.ModuleList([VitLayer(config) for _ in range(config.nlayers)])

    def forward(self, hidden_states, return_inter = False, custom_inter = None):
        for _, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, return_inter, custom_inter)
        return hidden_states
    
class VitPooler(nn.Module):
    def __init__(self, cls_idx = -1):
        super(VitPooler, self).__init__()
        self.cls_idx = cls_idx

    def forward(self, encoder_output):
        return encoder_output[:, self.cls_idx, :]
    
class VitModel(nn.Module):
    def __init__(self, config = example_config):
        super(VitModel, self).__init__()
        self.config = config

        self.embedding = VitEmbedding(config)

        self.encoder = VitEncoder(config)

        self.sampler = Sampler(config.image_size, config.interpolate)
        self.layer_norm = nn.LayerNorm(config.d_model, eps = config.layernorm_eps)
        self.pooler = VitPooler()
        self.classifier = nn.Linear(config.d_model, config.num_labels)
        # self.loss_func = LabelSmoothingLoss(config.label_smoothing, config.num_labels)
        self.plugin = None
        self.loss_func = CELoss()

        self.initializer_range = config.initializer_range

        self.weight_init()
        self.to(config.device)

    def weight_init(self):
        for _, p in self.named_parameters():
            nn.init.trunc_normal_(p, std = self.initializer_range)

    def init_plugin(self, layer = -1, use_act = True):
        activation = self.config.activation if use_act else "linear"
        layer = layer % self.config.nlayers
        self.plugin_layer = layer
        if layer == self.config.nlayers - 1:
            self.plugin = VitPlugin(
                self.config.d_model, 
                self.config.d_intermediate, 
                self.config.dropout_rate, 
                activation,
                self.config.layernorm_eps
            )
        else:
            self.encoder.layers[layer + 1].plugin = VitPlugin(
                self.config.d_model, 
                self.config.d_intermediate, 
                self.config.dropout_rate, 
                activation,
                self.config.layernorm_eps
            )

    @property
    def plugin_model(self):
        if self.plugin_layer == self.config.nlayers - 1:
            return self.plugin
        else:
            return self.encoder.layers[self.plugin_layer + 1].plugin

    def forward(self, image, label = None, return_batch_loss = False, return_inter = False, custom_inter = None):
        image = self.sampler(image)
        image_embedding = self.embedding(image)
        encoder_output = self.encoder(image_embedding, return_inter, custom_inter)
        encoder_output = self.layer_norm(encoder_output)
        if self.plugin is not None:
            encoder_output = self.plugin(encoder_output, return_inter, custom_inter)
        if return_inter:
            encoder_output, intermediate = encoder_output
        else:
            intermediate = None
        pooled_output = self.pooler(encoder_output)
        logits = self.classifier(pooled_output)

        label_predict = logits.max(1)[1]
        model_output = ModelOutput(logits = logits, label_predict = label_predict, intermediate = intermediate)

        if label is not None:
            loss = self.loss_func(logits, label, return_batch_loss)
            model_output.loss = loss

        return model_output
