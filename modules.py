# modules for EEG2image

import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from math import sqrt
from torch_geometric.nn import GATConv

class EEG_GAT(nn.Module):
    def __init__(self, in_channels=250, out_channels=250, num_channels=63):
        super(EEG_GAT, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = GATConv(in_channels=in_channels, out_channels=out_channels, heads=1)
        self.num_channels = num_channels
        # Create dense directed channel graph without self-loops and keep it as a buffer.
        edge_pairs = [(i, j) for i in range(self.num_channels) for j in range(self.num_channels) if i != j]
        edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        self.register_buffer('edge_index', edge_index)

    def forward(self, x):
        batch_size, _, num_channels, num_features = x.size()
        # Reshape x to (batch_size*num_channels, num_features) to pass through GATConv
        # print("x shape in EEG_GAT:", x.shape)
        # print("numberfeatures:", num_features)
        # x = x.view(batch_size*num_channels, num_features)
        x = x.reshape(batch_size * num_channels, num_features)
        x = self.conv1(x, self.edge_index)
        x = x.view(batch_size, num_channels, -1)
        x = x.unsqueeze(1)
        return x

class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn
    
class PositionalEmbedding(nn.Module):
	def __init__(self, d_model, max_len=5000):
		super(PositionalEmbedding, self).__init__()
		pe = torch.zeros(max_len, d_model).float()
		pe.require_grad = False
		position = torch.arange(0, max_len).float().unsqueeze(1)
		div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
		pe = pe.unsqueeze(0)
		self.register_buffer('pe', pe)
	def forward(self, x):
		return self.pe[:, :x.size(1)]


class SubjectEmbedding(nn.Module):
    def __init__(self, num_subjects, d_model):
        super(SubjectEmbedding, self).__init__()
        self.subject_embedding = nn.Embedding(num_subjects, d_model)
        self.shared_embedding = nn.Parameter(torch.randn(1, d_model))  # Shared token for unknown subjects
        self.mask_embedding = nn.Parameter(torch.randn(1, d_model))  # Mask token embedding

    def forward(self, subject_ids):
        if subject_ids[0] is None or torch.any(subject_ids >= self.subject_embedding.num_embeddings):
            batch_size = subject_ids.size(0)
            return self.shared_embedding.expand(batch_size, 1, -1)
        else:
            return self.subject_embedding(subject_ids).unsqueeze(1)


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term[:w[:, 1::2].shape[1]])

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()
    
class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1, joint_train=False, num_subjects=None):
        super(DataEmbedding, self).__init__()
        if joint_train and num_subjects is not None:
            self.value_embedding = nn.ModuleDict({
                str(subject_id): nn.Linear(c_in, d_model) for subject_id in range(num_subjects)
            })
        else:
            self.value_embedding = nn.Linear(c_in, d_model)  
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)
        self.subject_embedding = SubjectEmbedding(num_subjects, d_model) if num_subjects is not None else None
        self.mask_token = nn.Parameter(torch.randn(1, d_model))  # Mask token embedding
        self.joint_train = joint_train
        
    def forward(self, x, x_mark, subject_ids=None, mask=None):
        if self.joint_train:
            x = torch.stack([self.value_embedding[str(subject_id.item())](x[i]) for i, subject_id in enumerate(subject_ids)])
        else:
            x = self.value_embedding(x)

        if x_mark is not None:
            x = x + self.temporal_embedding(x_mark) + self.position_embedding(x)
            # print("x_mark")
        if mask is not None:
            x = x * (~mask.bool()) + self.mask_token * mask.float()
            # print("mask")
        if self.subject_embedding is not None:
            subject_emb = self.subject_embedding(subject_ids)  # (batch_size, 1, d_model)
            x = torch.cat([subject_emb, x], dim=1) 
        return self.dropout(x)
    


# def weights_init_tensor(module: nn.Module):
#     """Match the original weights_init_normal(m) behavior exactly."""
#     classname = module.__class__.__name__
#     with torch.no_grad():
#         if classname.find('Conv') != -1:
#             if hasattr(module, 'weight') and module.weight is not None:
#                 init.normal_(module.weight.data, 0.0, 0.02)
#         elif classname.find('Linear') != -1:
#             if hasattr(module, 'weight') and module.weight is not None:
#                 init.normal_(module.weight.data, 0.0, 0.02)
#         elif classname.find('BatchNorm') != -1:
#             if hasattr(module, 'weight') and module.weight is not None:
#                 init.normal_(module.weight.data, 1.0, 0.02)
#             if hasattr(module, 'bias') and module.bias is not None:
#                 init.constant_(module.bias.data, 0.0)
        

def weights_init_tensor(name: str, t: torch.Tensor):
    """Initialize tensor by name heuristic (safe default)."""
    lname = name.lower()
    with torch.no_grad():
        if lname.endswith("bias"):
            t.zero_()
            return
        # LayerNorm weights often end with .weight but should be 1.0
        if "norm" in lname or "layernorm" in lname:
            # only weights should be 1; bias handled above
            if lname.endswith("weight"):
                t.fill_(1.0)
            return
        # default for weights
        t.normal_(0.0, 0.02)


# def selective_reinit(model: nn.Module, init_groups: set):
#     """
#     Reinitialize modules by group, matching the original apply(weights_init_normal)
#     behavior exactly for Conv/Linear/BatchNorm only.
#     """
#     changed = []
#     for module_name, module in model.named_modules():
#         if module_name == "":
#             continue

#         direct_param_names = [
#             f"{module_name}.{param_name}" for param_name, _ in module.named_parameters(recurse=False)
#         ]
#         if not direct_param_names:
#             continue

#         matching_groups = [
#             ParameterGroupManager.get_group(param_name)
#             for param_name in direct_param_names
#             if ParameterGroupManager.get_group(param_name) in init_groups
#         ]
#         if not matching_groups:
#             continue

#         module_group = matching_groups[0]
#         if not ParameterGroupManager.INIT_SAFE.get(module_group, True):
#             continue

#         classname = module.__class__.__name__
#         if classname.find('Conv') != -1 or classname.find('Linear') != -1 or classname.find('BatchNorm') != -1:
#             weights_init_tensor(module)
#             changed.extend(direct_param_names)
#     return changed


class ParameterGroupManager:

    GROUP_PATTERNS = {
        "SUBJ":  [r"^subject_layer\.weights$", r"^subject_wise_linear\."],
        "ENC":   [r"^encoder\." , r"^embed\.", r"^nsam\.", r"^noise_aug\."],
        "GAT":   [r"^gatnn\."],
        "POS":   [r"^channel_pos_embedding\.", r"^pos_proj\.", r"\.pos_embed", r"\.position_embedding"],
        "CATTN": [r"^channel_attention\.", r"\.channel_attention\."],
        "NORM":  [r"^feature_norm(\.|$)", r"^feature_norm2(\.|$)", r"\.norm\d*\.", r"\.norm\d*$"],
        "CNN":   [r"^enc_eeg\."],
        "PROJ":  [r"^proj_eeg\."],
        "LOGIT": [r"^logit_scale$"],
        "DEC":   [r"^decoder", r"^mask_token"],  # decoder parameters (not used in fine-tuning, but appear in checkpoint)
    }

    LEGACY_ALIASES = {
        "W": {"SUBJ", "ENC", "GAT", "POS", "CATTN", "CNN", "PROJ", "LOGIT"},
        "N": {"NORM"},
        "S": {"SUBJ", "POS", "CATTN"},
        "T": set(),
        "B": {"SUBJ", "ENC", "GAT", "POS", "CATTN", "CNN", "PROJ"},
    }

    INIT_SAFE = {
        "SUBJ": False,  
        "ENC":  True,   
        "GAT":  True,
        "POS":  True,
        "CATTN": True,
        "NORM": False,   
        "CNN":  True,
        "PROJ": True,
        "LOGIT": True,  
        "DEC":  False,   
    }
   

    @staticmethod
    def set_init_safe(init_safe_groups: set):
        """
        Update INIT_SAFE dict based on specified groups.
        Groups in init_safe_groups are set to True, all others to False.
        """
        # Create a new INIT_SAFE with all False
        new_init_safe = {g: False for g in ParameterGroupManager.GROUP_PATTERNS.keys()}
        # Set specified groups to True
        for g in init_safe_groups:
            if g in new_init_safe:
                new_init_safe[g] = True
        ParameterGroupManager.INIT_SAFE = new_init_safe

    
    @staticmethod
    def selective_init(model: nn.Module, init_groups: set):
        changed = []
        for name, p in model.named_parameters():
            g = ParameterGroupManager.get_group(name)
            if g in init_groups:
                if not ParameterGroupManager.INIT_SAFE.get(g, True):
                    continue
                weights_init_tensor(name, p.data)
                changed.append(name)
        return changed

    @staticmethod
    def compile_patterns():
        compiled = {}
        for g, pats in ParameterGroupManager.GROUP_PATTERNS.items():
            compiled[g] = [re.compile(p) for p in pats]
        return compiled

    _COMPILED = {}  # Will be initialized after class definition

    @staticmethod
    def get_group(param_name: str) -> str:
        for g, regexes in ParameterGroupManager._COMPILED.items():
            for rx in regexes:
                if rx.search(param_name):
                    return g
        return "OTHER"

    @staticmethod
    def parse_groups(s: str):
        """
        Parse comma-separated groups.
        Special token 'ALL' expands to all known groups.
        """
        raw = [x.strip().upper() for x in s.split(",") if x.strip()]
        if not raw:
            return set()
        all_groups = set(ParameterGroupManager.GROUP_PATTERNS.keys())
        if "ALL" in raw:
            return all_groups

        expanded = set()
        recognized = False
        for token in raw:
            if token in all_groups:
                expanded.add(token)
                recognized = True
                continue
            if token in ParameterGroupManager.LEGACY_ALIASES:
                expanded.update(ParameterGroupManager.LEGACY_ALIASES[token])
                recognized = True
                continue

        if recognized:
            return expanded

        if not expanded and raw:
            return set(ParameterGroupManager.GROUP_PATTERNS.keys())
        return expanded

    @staticmethod
    def remap_ckpt_key(k: str) -> str:
        """
        Normalize ckpt keys into model-key candidates space.
        We will try multiple forms later; here only cheap normalization.
        """
        # remove dp wrappers
        for p in ("module.", "model.", "eeg_model."):
            if k.startswith(p):
                k = k[len(p):]
        # collapse encoder.encoder.
        if k.startswith("encoder.encoder."):
            k = "encoder." + k[len("encoder.encoder."):]
        return k

    @staticmethod
    def build_load_dict(model: nn.Module, ckpt_sd: dict, load_groups: set) -> dict:
        """
        Create a filtered & remapped state_dict that matches model.state_dict keys.
        Strategy:
          - normalize key
          - try direct match
          - if startswith encoder., also try dropping encoder. prefix (some models are encoder-only)
        """
        model_sd = model.state_dict()
        model_keys = set(model_sd.keys())

        out = {}
        for k, v in ckpt_sd.items():
            kk = ParameterGroupManager.remap_ckpt_key(k)

            candidates = [kk]
            if kk.startswith("encoder."):
                candidates.append(kk[len("encoder."):])
            if kk.startswith("encoder.embed."):
                candidates.append("encoder." + kk[len("encoder.embed."):])
                candidates.append(kk[len("encoder.embed."):])

            for cand in candidates:
                if cand in model_keys:
                    g = ParameterGroupManager.get_group(cand)
                    if g in load_groups:
                        out[cand] = v
                    break
        return out
