
# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage

import models.configs as configs

# from .modeling_resnet import ResNetV2


logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, crt_local_mod=None, query_mod=None, key_mod=None, top_n=None, device='cuda'):

        # print('in Attention, device:', device)

        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
            
        query_layer = self.transpose_for_scores(mixed_query_layer) # 1 * 12 * search_patches * 64
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))                     # 1 * 12 * (search_patches+1) * (search_patches+1)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        
        relevance = None  
        # compute relevance between target query and search query
        if query_mod is not None:
            query_relevance = torch.matmul(query_mod[:, 1:, :], query_layer[:, :, 1:, :].transpose(-1, -2))       # 1 * 12 * target_patches * search_patches
            query_relevance = self.softmax(query_relevance / math.sqrt(query_mod.size(-1)))
            if crt_local_mod is not None:
                query_relevance = query_relevance * crt_local_mod                                                 # 1 * 12 * target_patches * search_patches
            keep = torch.argsort(torch.argsort(query_relevance, dim=-1)) >= (query_relevance.shape[-1] - top_n)   # 1 * 12 * target_patches * search_patches
            keep = torch.any(keep, dim=-2)                                                                        # 1 * 12 * search_patches
            keep = torch.ones_like(keep, dtype=torch.float32) * keep                                              # 1 * 12 * search_patches
            query_relevance = torch.cat((torch.ones(1, 12, 1, device=device), keep.clone()), dim=-1)              # 1 * 12 * (search_patches+1)
            attention_probs = query_relevance[:, :, :, None] * attention_probs
            relevance = query_relevance

        # compute relevance between target key and search key
        if key_mod is not None:
            key_relevance = torch.matmul(key_mod[:, 1:, :], key_layer[:, :, 1:, :].transpose(-1, -2))             # 1 * 12 * target_patches * search_patches
            key_relevance = self.softmax(key_relevance / math.sqrt(key_mod.size(-1)))
            if crt_local_mod is not None:
                key_relevance = key_relevance * crt_local_mod
            keep = torch.argsort(torch.argsort(key_relevance, dim=-1)) >= (key_relevance.shape[-1] - top_n)       # 1 * 12 * target_patches * search_patches
            keep = torch.any(keep, dim=-2)                                                                        # 1 * 12 * search_patches
            keep = torch.ones_like(keep, dtype=torch.float32) * keep                                              # 1 * 12 * search_patches
            key_relevance = torch.cat((torch.ones(1, 12, 1, device=device), keep.clone()), dim=-1)                # 1 * 12 * (search_patches+1)
            attention_probs = key_relevance[:, :, None, :] * attention_probs                                      # 1 * 12 * (search_patches+1) * (search_patches+1)
            relevance = key_relevance

        # compute match between query_relevance and key_relevance
        if query_mod is not None and key_mod is not None:
            relevance = torch.matmul(query_relevance.transpose(-1, -2), key_relevance)                            # 1 * 12 * (search_patches+1) * (search_patches+1)
            relevance = self.softmax(relevance / math.sqrt(query_mod.size(-2)))
            attention_probs = relevance * attention_probs
        
        # modulate attention only by crt_local_mod
        if crt_local_mod is not None and query_mod is None and key_mod is None:
            relevance = self.softmax(crt_local_mod).repeat(1, 12, 1)                                             # 1 * 12 * search_patches
            relevance = torch.cat((torch.ones(1, 12, 1, device=device), relevance.clone()), dim=-1)              # 1 * 12 * (search_patches+1)
            attention_probs = relevance[:, :, :, None] * attention_probs
        
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)

        return attention_output, weights, query_layer, key_layer, value_layer, context_layer, attention_output, relevance


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3, posembed=True):
        super(Embeddings, self).__init__()
        self.hybrid = None
        img_size = _pair(img_size)

        if config.patches.get("grid") is not None:
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            n_patches = (img_size[0] // 16) * (img_size[1] // 16)
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers,
                                         width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, config.hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        
        self.dropout = Dropout(config.transformer["dropout_rate"])
        self.posembed = posembed


    def forward(self, x_ls, device='cuda'):

        # print('in Embedding, device:', device)

        # x_ls is a list of inputs, with default length = 1
        B = x_ls[0].shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        if len(x_ls) == 1: # only search or only target
            all_x = x_ls[0]
            if self.hybrid:
                all_x = self.hybrid_model(all_x)
            all_x = self.patch_embeddings(all_x)
            all_x = all_x.flatten(2)
            all_x = all_x.transpose(-1, -2)
            all_x = torch.cat((cls_tokens, all_x), dim=1)
            
        else:             # concatenate target and search embeddings
            all_x = []
            for x in x_ls:
                if self.hybrid:
                    x = self.hybrid_model(x)
                x = self.patch_embeddings(x)
                x = x.flatten(2)
                x = x.transpose(-1, -2)
                all_x.append(x)
            all_x = torch.cat(all_x, dim=1)
            all_x = torch.cat((cls_tokens, all_x), dim=1)

        if self.posembed:
            if all_x.size(1) == self.position_embeddings.size(1):
                embeddings = all_x + self.position_embeddings
            else: 
                pe = self.position_embeddings[:, 1:]
                h, w = x_ls[0].shape[-2], x_ls[0].shape[-1]
                n_patch_h, n_patch_w = int(h // 16), int(w // 16)
                from torchvision import transforms as tf
                resize = tf.Resize(size=(n_patch_h, n_patch_w))
                resized_pe = torch.stack([resize(e.reshape(14, 14)[None]).squeeze() for e in pe.squeeze().T]).reshape(-1, n_patch_h*n_patch_w).T
                resized_pe = torch.cat([self.position_embeddings[:, 0][None], resized_pe[None]], axis=1)
                embeddings = all_x + resized_pe
        else:
            embeddings = all_x
        
        embeddings = self.dropout(embeddings)
        return embeddings


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x, crt_mod_method=None, crt_global_mod=None, crt_local_mod=None, query_mod=None, key_mod=None, top_n=None, device='cuda'):
        
        # print('in block, device:', device)
        
        h = x
#         if crt_global_mod is not None:
#             h[:, 1:] = h[:, 1:] * crt_global_mod[None, :, None]
        
        x = self.attention_norm(x)
        x, weights, query_layer, key_layer, value_layer, context_layer, attention_output, relevance = self.attn(x, crt_local_mod, query_mod, key_mod, top_n, device=device)
        
        if crt_global_mod is not None and crt_mod_method == 1:
            x[:, 1:] = x[:, 1:] * crt_global_mod[None, :, None]
        elif crt_global_mod is not None and crt_mod_method == 2:
            h[:, 1:] = h[:, 1:] * crt_global_mod[None, :, None]
        x = x + h
        if crt_global_mod is not None and crt_mod_method == 3:
            x[:, 1:] = x[:, 1:] * crt_global_mod[None, :, None]
            
        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        
        if crt_global_mod is not None and crt_mod_method == 4:
            x[:, 1:] = x[:, 1:] * crt_global_mod[None, :, None]
        elif crt_global_mod is not None and crt_mod_method == 5:
            h[:, 1:] = h[:, 1:] * crt_global_mod[None, :, None]
        x = x + h
        if crt_global_mod is not None and crt_mod_method == 6:
            x[:, 1:] = x[:, 1:] * crt_global_mod[None, :, None]
            
        return x, weights, query_layer, key_layer, value_layer, context_layer, attention_output, relevance


    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)

        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))
        
    def forward(self, hidden_states, crt_mod_method=None, crt_global_mod_layers=[], crt_global_mod=None, crt_local_mod=None, query_mod=None, key_mod=None, device='cuda'):
        
        # print('in Encoder, device:', device)
        
        hidden = []
        attn_weights, q_layers, k_layers, v_layers, context_layers, attention_outputs, relevances = [], [], [], [], [], [], []
        if query_mod is None and key_mod is None:
            for i, layer_block in enumerate(self.layer):
                if i not in crt_global_mod_layers:
                    crt_global_mod_current = None
                else:
                    crt_global_mod_current = crt_global_mod                
#                     print('CRT modulating l = {}'.format(i))
                hidden_states, weights, query_layer, key_layer, value_layer, context_layer, attention_output, relevance = layer_block(hidden_states, crt_mod_method, crt_global_mod_current, crt_local_mod, device=device)
                if self.vis:
                    hidden.append(hidden_states)
                    attn_weights.append(weights)
                    q_layers.append(query_layer)
                    k_layers.append(key_layer)
                    v_layers.append(value_layer)
                    context_layers.append(context_layer)
                    attention_outputs.append(attention_output)
                    relevances.append(relevance) 

        else:
            for i, layer_block in enumerate(self.layer):
                if query_mod is not None:
                    q_mod = query_mod[i - len(self.layer) + len(query_mod)] if i >= len(self.layer) - len(query_mod) else None  # modulate the last n layers with n = len(query_mod)
                else: 
                    q_mod = None
                if key_mod is not None:
                    k_mod = key_mod[i - len(self.layer) + len(key_mod)] if i >= len(self.layer) - len(key_mod) else None  # modulate the last n layers with n = len(query_mod)
                else:
                    k_mod = None
#                 top_n = 12 - i if i >= len(self.layer) - len(query_mod) else None
                top_n = 1
#                 top_n = 5 if i < 6 else 1
                
                if i not in crt_global_mod_layers:
                    crt_global_mod_current = None
                else:
                    crt_global_mod_current = crt_global_mod
                hidden_states, weights, query_layer, key_layer, value_layer, context_layer, attention_output, relevance = layer_block(hidden_states, crt_mod_method, crt_global_mod_current, crt_local_mod, q_mod, k_mod, top_n, device=device)

                if self.vis:
                    hidden.append(hidden_states)
                    attn_weights.append(weights)
                    q_layers.append(query_layer)
                    k_layers.append(key_layer)
                    v_layers.append(value_layer)
                    context_layers.append(context_layer)
                    attention_outputs.append(attention_output)
                    relevances.append(relevance) 
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights, q_layers, k_layers, v_layers, context_layers, attention_outputs, relevances, hidden


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis, posembed=True):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size, posembed=posembed)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids, crt_mod_method=None, crt_global_mod_layers=[], crt_global_mod=None, crt_local_mod=None, query_mod=None, key_mod=None, embedding_output=None, device='cuda'):
        
        # print('in Transformer, device:', device)
        
        if embedding_output is None:
            embedding_output = self.embeddings(input_ids, device=device)
        encoded, attn_weights, q_layers, k_layers, v_layers, context_layers, attention_outputs, relevances, hidden = self.encoder(embedding_output, crt_mod_method, crt_global_mod_layers, crt_global_mod, crt_local_mod, query_mod, key_mod, device=device)
        return encoded, attn_weights, q_layers, k_layers,v_layers, context_layers, attention_outputs, relevances, hidden


class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False, posembed=True):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size, vis, posembed)
        self.head = Linear(config.hidden_size, num_classes)

    def forward(self, x, labels=None, crt_mod_method=None, crt_global_mod_layers=[], crt_global_mod=None, crt_local_mod=None, query_mod=None, key_mod=None, embedding_output=None, device='cuda'):
        
        
        # print('in ViT, device:', device)
        
        x, attn_weights, q_layers, k_layers, v_layers, context_layers, attention_outputs, relevances, hidden = self.transformer(x, crt_mod_method, crt_global_mod_layers, crt_global_mod, crt_local_mod, query_mod, key_mod, embedding_output, device=device)
            
        logits = self.head(x[:, 0])

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
            return loss
        else:
            return x, attn_weights, q_layers, k_layers, v_layers, context_layers, attention_outputs, relevances, hidden


    def load_from(self, weights):
        with torch.no_grad():
            if self.zero_head:
                nn.init.zeros_(self.head.weight)
                nn.init.zeros_(self.head.bias)
            else:
                self.head.weight.copy_(np2th(weights["head/kernel"]).t())
                self.head.bias.copy_(np2th(weights["head/bias"]).t())

            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.transformer.embeddings.cls_token.copy_(np2th(weights["cls"]))
            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(weights["conv_root/kernel"], conv=True))
                gn_weight = np2th(weights["gn_root/scale"]).view(-1)
                gn_bias = np2th(weights["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=bname, n_unit=uname)


CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'testing': configs.get_testing(),
}
