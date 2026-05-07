"""
EEG/MEG Encoder Models
"""
import math
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
from einops.layers.torch import Rearrange
from open_clip.loss import ClipLoss
from timm.models.vision_transformer import Block
from modules import EEG_GAT, Encoder, EncoderLayer, FullAttention, AttentionLayer, DataEmbedding

class Config:
    def __init__(self):
        self.task_name = 'classification'
        self.seq_len = 250
        self.pred_len = 250
        self.output_attention = False
        self.d_model = 250
        self.embed = 'timeF'
        self.freq = 'h'
        self.dropout = 0.25
        self.factor = 1
        self.n_heads = 4
        self.d_ff = 512
        self.e_layers = 1
        self.activation = 'gelu'
        self.enc_in = 63


class iTransformer(nn.Module):
    def __init__(self, configs, joint_train=False, num_subjects=10):
        super(iTransformer, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.enc_in = configs.enc_in
        # Embedding
        self.enc_embedding = DataEmbedding(configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout, joint_train=False, num_subjects=num_subjects)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

    def forward(self, x_enc, x_mark_enc, subject_ids=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc, subject_ids)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        enc_out = enc_out[:, :self.enc_in, :]
        return enc_out


class iTransformerDeep(nn.Module):
    """
    Multi-layer Transformer encoder with trainable sin-cos positional embeddings.
    """
    def __init__(self, configs, joint_train=False, num_subjects=10):
        super(iTransformerDeep, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.d_model = configs.d_model
        self.enc_in = configs.enc_in
        self.output_attention = configs.output_attention
        
        self.depth = configs.e_layers if configs.e_layers > 1 else 4
        
        # Patch Embedding
        self.patch_embed = nn.Conv1d(
            in_channels=configs.seq_len,
            out_channels=configs.d_model,
            kernel_size=3,
            padding=1,
            padding_mode='circular',
            bias=False
        )
        
        # Positional Embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, configs.enc_in + 1, configs.d_model),
            requires_grad=True
        )
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, configs.d_model))
        
        # Transformer Blocks
        mlp_ratio = configs.d_ff / configs.d_model if hasattr(configs, 'd_ff') else 4.0
        self.blocks = nn.ModuleList([
            Block(
                dim=configs.d_model,
                num_heads=configs.n_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                drop_path=configs.dropout,
                norm_layer=nn.LayerNorm
            ) for _ in range(self.depth)
        ])
        
        self.norm = nn.LayerNorm(configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)
        
        self.initialize_weights()
    
    def initialize_weights(self):
        pos_embed = self.get_sincos_encoding(self.enc_in + 1, self.d_model)
        self.pos_embed.data.copy_(pos_embed)
        
        w = self.patch_embed.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        
        torch.nn.init.normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            torch.nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    @staticmethod
    def get_sincos_encoding(num_positions, d_model):
        position = torch.arange(num_positions, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * 
                            -(math.log(10000.0) / d_model))
        
        pe = torch.zeros(num_positions, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
    
    def forward(self, x_enc, x_mark_enc=None, subject_ids=None):
        B, C, T = x_enc.shape
        
        x = x_enc.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        
        x = x + self.pos_embed[:, 1:, :]
        x = self.dropout(x)
        
        for blk in self.blocks:
            x = blk(x)
        
        x = self.norm(x)
        return x


class Subjectlayer(nn.Module):
    def __init__(self, configs, joint_train=False, num_subjects=10):
        super(Subjectlayer, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.enc_in = configs.enc_in
        self.enc_embedding = DataEmbedding(configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout, joint_train=False, num_subjects=num_subjects)

    def forward(self, x_enc, x_mark_enc, subject_ids=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc, subject_ids)
        enc_out = enc_out[:, :self.enc_in, :]
        return enc_out


class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40, num_channels=63):
        super().__init__()
        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (num_channels, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.shallownet(x)
        x = self.projection(x)
        return x


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FlattenHead(nn.Sequential):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        return x


class Enc_eeg(nn.Sequential):
    def __init__(self, emb_size=40, depth=3, n_classes=4, num_channels=63, seq_len=250, **kwargs):
        patch = PatchEmbedding(emb_size, num_channels=num_channels)
        
        flatten = FlattenHead()
        super().__init__(
            patch,
            flatten
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, num_channels, seq_len)
            out = self.forward(dummy)
            self.output_dim = int(out.shape[-1])



class Proj_eeg(nn.Sequential):
    def __init__(self, embedding_dim=1440, proj_dim=1024, drop_proj=0.5):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )


class Proj_img(nn.Sequential):
    def __init__(self, embedding_dim=1024, proj_dim=1024, drop_proj=0.3):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )

    def forward(self, x):
        return x

class NoiseAugmentation(nn.Module):
    def __init__(self, sigma=0.01):
        super().__init__()
        self.sigma = sigma
        
    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.sigma
            return x + noise
        return x


class EnhancedNSAM(nn.Module):
    """Neural Spectral Attention Module with frequency band analysis"""
    def __init__(self, num_channels: int = 63, seq_length: int = 250, sampling_rate: float = 250.0):
        super().__init__()
        self.num_channels = num_channels
        self.seq_length = seq_length
        self.sampling_rate = sampling_rate

        self.bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }

        self.channel_attention = nn.Sequential(
            nn.Linear(num_channels, num_channels),
            nn.GELU(),
            nn.Linear(num_channels, num_channels),
            nn.Sigmoid()
        )

        self.spectral_attention = nn.Sequential(
            nn.Linear(len(self.bands), len(self.bands)),
            nn.GELU(),
            nn.Linear(len(self.bands), len(self.bands)),
            nn.Softmax(dim=-1)
        )
        
        self.alpha = nn.Parameter(torch.zeros(1))
        self.norm = nn.LayerNorm(seq_length)

    def get_band_mask(self, freqs: torch.Tensor, band: str) -> torch.Tensor:
        low, high = self.bands[band]
        return ((freqs >= low) & (freqs <= high)).float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        batch_size = x.shape[0]
        
        X = torch.fft.rfft(x, dim=-1)
        freqs = torch.fft.rfftfreq(self.seq_length, 1/self.sampling_rate).to(x.device)

        band_features = {}
        band_powers = []
        
        for band in self.bands.keys():
            mask = self.get_band_mask(freqs, band).to(x.device)
            mask_sum = mask.sum() + 1e-8
            X_band = X * mask.unsqueeze(0).unsqueeze(0)
            band_features[band] = X_band
            power = torch.sum(torch.abs(X_band).pow(2), dim=-1) / mask_sum
            band_powers.append(power)

        band_powers = torch.stack(band_powers, dim=-1)
        
        channel_weights = self.channel_attention(band_powers.mean(dim=-1))
        channel_weights = channel_weights.unsqueeze(-1)

        band_powers = torch.log1p(band_powers)
        mean_bp = band_powers.mean(dim=-1, keepdim=True)
        std_bp = band_powers.std(dim=-1, keepdim=True) + 1e-8
        band_powers = (band_powers - mean_bp) / std_bp

        spectral_input = band_powers.mean(dim=1)
        s_mean = spectral_input.mean(dim=-1, keepdim=True)
        s_std = spectral_input.std(dim=-1, keepdim=True) + 1e-8
        spectral_input = (spectral_input - s_mean) / s_std

        spectral_weights = self.spectral_attention(spectral_input)

        X_combined = torch.zeros_like(X)
        for i, band in enumerate(self.bands.keys()):
            X_combined += (band_features[band] * 
                         channel_weights * 
                         spectral_weights[:, i:i+1].unsqueeze(1))

        output = torch.fft.irfft(X_combined, n=self.seq_length, dim=-1)
        output = self.norm(output)
        
        alpha = torch.sigmoid(self.alpha)
        output = alpha * output + (1 - alpha) * identity
        
        return output


class SubjectLayers(nn.Module):
    """Per-subject linear transformation layer"""
    def __init__(self, in_channels: int, out_channels: int, n_subjects: int, init_id: bool = False):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(n_subjects, in_channels, out_channels))
        if init_id:
            assert in_channels == out_channels
            self.weights.data[:] = torch.eye(in_channels)[None]
        self.weights.data *= 1 / in_channels**0.5
        
    def forward(self, x, subjects):
        _, C, D = self.weights.shape
        weights = self.weights.gather(0, subjects.view(-1, 1, 1).expand(-1, C, D))
        return torch.einsum("bct,bcd->bdt", x, weights)
        
    def __repr__(self):
        S, C, D = self.weights.shape
        return f"SubjectLayers({C}, {D}, {S})"


class GatedFusion(nn.Module):
    """Gated fusion: y = g * new + (1-g) * old."""
    def __init__(self, num_channels: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv1d(num_channels * 2, num_channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x_old: torch.Tensor, x_new: torch.Tensor) -> torch.Tensor:
        gate = self.gate(torch.cat([x_old, x_new], dim=1))
        return gate * x_new + (1 - gate) * x_old


class TemporalChannelAttention(nn.Module):
    """Channel attention with temporal context via depthwise temporal conv."""
    def __init__(self, num_channels: int, kernel_size: int = 9):
        super().__init__()
        self.temporal_conv = nn.Conv1d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=num_channels,
            bias=False
        )
        self.mlp = nn.Sequential(
            nn.Linear(num_channels, num_channels),
            nn.GELU(),
            nn.Linear(num_channels, num_channels),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        temporal_feat = self.temporal_conv(x).mean(dim=-1)
        return self.mlp(temporal_feat)


class PositionAwareChannelAttention(nn.Module):
    """Position-aware channel attention using additive attention."""
    def __init__(self, pos_dim: int, attn_dim: int = 32):
        super().__init__()
        self.temporal_proj = nn.Linear(1, attn_dim, bias=False)
        self.pos_proj = nn.Linear(pos_dim, attn_dim, bias=False)
        self.score = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, temporal_feat: torch.Tensor, pos_emb: torch.Tensor) -> torch.Tensor:
        t = self.temporal_proj(temporal_feat.unsqueeze(-1))
        p = self.pos_proj(pos_emb).unsqueeze(0)
        logits = self.score(torch.tanh(t + p)).squeeze(-1)
        return torch.softmax(logits, dim=1)



# ============================================================================
# Main Encoder Models
# ============================================================================

class NICE_MEG(nn.Module):
    """NICE variant for MEG with configurable channels and sequence length."""
    def __init__(self, num_channels=271, sequence_length=201, num_subjects=10, num_features=64, num_latents=1024, num_blocks=1):
        super(NICE_MEG, self).__init__()
        # Create custom config with MEG-specific parameters
        default_config = Config()
        default_config.seq_len = sequence_length
        default_config.pred_len = sequence_length
        default_config.d_model = sequence_length
        default_config.enc_in = num_channels
        
        self.subject_wise_linear = nn.ModuleList([nn.Linear(default_config.d_model, sequence_length) for _ in range(num_subjects)])
        
        self.enc_eeg = Enc_eeg(num_channels=num_channels, seq_len=sequence_length)
        self.proj_eeg = Proj_eeg(embedding_dim=self.enc_eeg.output_dim)
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = ClipLoss()
    
    def forward(self, x, subject_ids):
        if x.dim() == 4:
            x = x.squeeze(1)
        x = x.unsqueeze(1)
        eeg_embedding = self.enc_eeg(x)
        out = self.proj_eeg(eeg_embedding)
        return out


class NICE(nn.Module):
    """Baseline encoder without advanced modules"""
    def __init__(self, num_channels=63, sequence_length=250, num_subjects=10, num_features=64, num_latents=1440, num_blocks=1):
        super(NICE, self).__init__()
        default_config = Config()
        self.subject_wise_linear = nn.ModuleList([nn.Linear(default_config.d_model, sequence_length) for _ in range(num_subjects)])
        self.enc_eeg = Enc_eeg()
        self.proj_eeg = Proj_eeg()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = ClipLoss()
         
    def forward(self, x, subject_ids):
        if x.dim() == 4:
            x = x.squeeze(1)
        x = x.unsqueeze(1)
        eeg_embedding = self.enc_eeg(x)
        out = self.proj_eeg(eeg_embedding)
        return out


class ATMS(nn.Module):
    """Encoder with iTransformer (Attention-based Temporal Modeling)"""
    def __init__(self, num_channels=63, sequence_length=250, num_subjects=10, num_features=64, num_latents=1440, num_blocks=1):
        super(ATMS, self).__init__()
        default_config = Config()
        self.encoder = iTransformer(default_config)
        self.subject_wise_linear = nn.ModuleList([nn.Linear(default_config.d_model, sequence_length) for _ in range(num_subjects)])
        self.enc_eeg = Enc_eeg()
        self.proj_eeg = Proj_eeg()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = ClipLoss()
         
    def forward(self, x, subject_ids, text_features=None, img_features=None):
        if x.dim() == 4:
            x = x.squeeze(1)
        x = self.encoder(x, None, subject_ids)
        x = x.unsqueeze(1)
        eeg_embedding = self.enc_eeg(x)
        out = self.proj_eeg(eeg_embedding)
        return out


class MCRL(nn.Module):
    """Multi-modal Contrastive Representation Learning encoder"""
    def __init__(self, num_channels=63, sequence_length=250, num_subjects=10, num_features=64, num_latents=1024, num_blocks=1):
        super(MCRL, self).__init__()
        default_config = Config()

        self.subject_layer = SubjectLayers(
            in_channels=num_channels,
            out_channels=num_channels,
            n_subjects=num_subjects,
            init_id=True
        )
        self.embed = Subjectlayer(default_config)
        self.encoder = iTransformer(default_config)

        self.nsam = EnhancedNSAM(
            num_channels=num_channels,
            seq_length=sequence_length,
            sampling_rate=250.0
        )
        
        self.enc_eeg = Enc_eeg()
        self.proj_eeg = Proj_eeg()
        self.feature_norm = nn.LayerNorm([num_channels, sequence_length])
        
        self.noise_aug = NoiseAugmentation(sigma=0.01)
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = ClipLoss()

    def forward(self, x, subject_ids, text_features=None, img_features=None):
        x = x.squeeze(1)
        x = self.subject_layer(x, subject_ids)

        x_trans = self.encoder(x, None, subject_ids)
        x_processed = self.nsam(x_trans)

        x_normalized = self.feature_norm(x_processed)
        x_normalized = x_normalized.unsqueeze(1)
        eeg_features = self.enc_eeg(x_normalized)
        eeg_projected = self.proj_eeg(eeg_features)

        final_features = eeg_projected
        if self.training and (text_features is not None) and (img_features is not None):
            x_aligned = self.inter_mcr(eeg_projected, img_features, text_features)
            final_features = x_aligned + eeg_projected
        if self.training:
            final_features = self.noise_aug(final_features)
        
        return final_features


class ChannelPositionEmbedding(nn.Module):
    """Channel spatial position embedding using 10-10 EEG coordinates"""
    def __init__(self, emb_dim=4):
        super().__init__()

        # ---- (1) Head scalp spatial coordinates: 63 channels standard 10-10 coordinates ----
        # shape: (63, 3)
        coords = [
            [-0.50,  0.85,  0.10],  # Fp1
            [ 0.50,  0.85,  0.10],  # Fp2
            [-0.60,  0.70,  0.08],  # AF7
            [-0.30,  0.72,  0.12],  # AF3
            [ 0.00,  0.75,  0.15],  # AFz
            [ 0.30,  0.72,  0.12],  # AF4
            [ 0.60,  0.70,  0.08],  # AF8
            [-0.70,  0.50,  0.05],  # F7
            [-0.45,  0.50,  0.08],  # F5
            [-0.20,  0.50,  0.12],  # F3
            [-0.05,  0.48,  0.13],  # F1
            [ 0.05,  0.48,  0.13],  # F2
            [ 0.20,  0.50,  0.12],  # F4
            [ 0.45,  0.50,  0.08],  # F6
            [ 0.70,  0.50,  0.05],  # F8
            [-0.88,  0.25,  0.02],  # FT9
            [-0.72,  0.30,  0.03],  # FT7
            [-0.48,  0.28,  0.06],  # FC5
            [-0.28,  0.28,  0.09],  # FC3
            [-0.12,  0.28,  0.11],  # FC1
            [ 0.00,  0.30,  0.12],  # FCz
            [ 0.12,  0.28,  0.11],  # FC2
            [ 0.28,  0.28,  0.09],  # FC4
            [ 0.48,  0.28,  0.06],  # FC6
            [ 0.72,  0.30,  0.03],  # FT8
            [ 0.88,  0.25,  0.02],  # FT10
            [-0.80,  0.00,  0.00],  # T7
            [-0.60, -0.02, -0.02],  # C5
            [-0.40,  0.00,  0.00],  # C3
            [-0.20, -0.02,  0.01],  # C1
            [ 0.00,  0.00,  0.02],  # Cz
            [ 0.20, -0.02,  0.01],  # C2
            [ 0.40,  0.00,  0.00],  # C4
            [ 0.60, -0.02, -0.02],  # C6
            [ 0.80,  0.00,  0.00],  # T8
            [-0.90, -0.18, -0.05],  # TP9
            [-0.70, -0.18, -0.03],  # TP7
            [-0.50, -0.20,  0.00],  # CP5
            [-0.30, -0.20,  0.02],  # CP3
            [-0.12, -0.22,  0.03],  # CP1
            [ 0.00, -0.22,  0.04],  # CPz
            [ 0.12, -0.22,  0.03],  # CP2
            [ 0.30, -0.20,  0.02],  # CP4
            [ 0.50, -0.20,  0.00],  # CP6
            [ 0.70, -0.18, -0.03],  # TP8
            [ 0.90, -0.18, -0.05],  # TP10
            [-0.70, -0.40, -0.08],  # P7
            [-0.48, -0.38, -0.05],  # P5
            [-0.28, -0.40, -0.03],  # P3
            [-0.10, -0.42, -0.02],  # P1
            [ 0.10, -0.42, -0.02],  # Pz (approx)
            [ 0.28, -0.40, -0.03],  # P2
            [ 0.48, -0.38, -0.05],  # P4
            [ 0.70, -0.40, -0.08],  # P6
            [ 0.88, -0.42, -0.10],  # P8
            [-0.60, -0.60, -0.12],  # PO7
            [-0.35, -0.62, -0.08],  # PO3
            [-0.05, -0.68, -0.06],  # POz
            [ 0.25, -0.62, -0.08],  # PO4
            [ 0.55, -0.60, -0.12],  # PO8
            [-0.20, -0.85, -0.18],  # O1
            [ 0.00, -0.90, -0.20],  # Oz
            [ 0.20, -0.85, -0.18],  # O2
        ]

        self.register_buffer("coords", torch.tensor(coords, dtype=torch.float32))  # Non-trainable

        # ---- (2) Use MLP to map 3D coordinates to arbitrary dimension ----
        self.mlp = nn.Sequential(
            nn.Linear(4, emb_dim),
            nn.Tanh(),
            nn.Linear(emb_dim, emb_dim, bias=False),
        )

    def forward(self):

        coords = self.coords   
        r = torch.norm(coords, dim=1, keepdim=True)
        coords_aug = torch.cat([coords, r], dim=1)
        pos_emb = self.mlp(coords_aug) 
        return pos_emb   # shape (63, emb_dim)


class HYBRID(nn.Module):
    """Hybrid encoder with multiple attention mechanisms and spatial channel embeddings"""
    def __init__(self, num_channels=63, sequence_length=250, num_subjects=10, num_features=64, num_latents=1024, num_blocks=1, channel_emb_dim=40):
        super(HYBRID, self).__init__()
        # Create custom config with MEG-specific parameters
        default_config = Config()
        default_config.seq_len = sequence_length
        default_config.pred_len = sequence_length
        default_config.d_model = sequence_length
        default_config.enc_in = num_channels
        
        # self.embed = Subjectlayer(default_config)
        self.subject_layer = SubjectLayers(
            in_channels=num_channels,
            out_channels=num_channels,
            n_subjects=num_subjects,
            init_id=True
        )
        
        # Channel positional embedding using spatial coordinates
        self.channel_pos_embedding = ChannelPositionEmbedding(emb_dim=channel_emb_dim)
        self.pos_proj = nn.Linear(channel_emb_dim, 1)

        self.gatnn = EEG_GAT(in_channels=sequence_length, out_channels=sequence_length, num_channels=num_channels)
        self.encoder = iTransformer(default_config, joint_train=False, num_subjects=num_subjects)
        self.feature_norm = nn.LayerNorm([num_channels, sequence_length])
        self.feature_norm2 = nn.LayerNorm(sequence_length)

        self.channel_attention = nn.Sequential(
            nn.Linear(num_channels, num_channels),
            nn.GELU(),
            nn.Linear(num_channels, num_channels),
            nn.Sigmoid()
        )
        
        self.enc_eeg = Enc_eeg()
        self.proj_eeg = Proj_eeg()

        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # self.loss_func = ClipLoss()
    
    def forward(self, x, subject_ids):
        if x.dim() == 4:
            x = x.squeeze(1)
        
        x = self.subject_layer(x, subject_ids) # (B, 63, 250)

        x_for_gat = x.unsqueeze(1)
        gat_out = self.gatnn(x_for_gat)
        x_gat = gat_out[0] if isinstance(gat_out, tuple) else gat_out
        x_gat = x_gat.squeeze(1)
        x_gat = x_gat + x

        x_trans = self.encoder(x_gat, None, subject_ids)
        x_trans = x_trans + x_gat
        
        channel_weights = self.channel_attention(x.mean(dim=-1))
        channel_weights = channel_weights.unsqueeze(-1)
        x_weighted = x_trans * channel_weights 
        x_weighted = x_weighted + x_trans

        # Get spatial channel position embeddings
        channel_pos_emb = self.channel_pos_embedding()  # shape (63, channel_emb_dim)
        pos_bias = self.pos_proj(channel_pos_emb) # shape (1, 63, 1)
        x_weighted = x_weighted + pos_bias  # Broadcasting addition

        x_weighted = x_weighted.unsqueeze(1) 
        x_weighted = self.feature_norm(x_weighted)
        eeg_features = self.enc_eeg(x_weighted)
        eeg_projected = self.proj_eeg(eeg_features)
        
        return eeg_projected


# class ATMS_MEG(nn.Module):
#     """ATMS variant for MEG with configurable channels and sequence length."""
#     def __init__(self, num_channels=271, sequence_length=201, num_subjects=10, num_features=64, num_latents=1024, num_blocks=1):
#         super(ATMS_MEG, self).__init__()
#         # Create custom config with MEG-specific parameters
#         default_config = Config()
#         default_config.seq_len = sequence_length
#         default_config.pred_len = sequence_length
#         default_config.d_model = sequence_length
#         default_config.enc_in = num_channels
        
#         self.encoder = iTransformer(default_config, joint_train=False, num_subjects=num_subjects)
#         self.subject_wise_linear = nn.ModuleList([nn.Linear(default_config.d_model, sequence_length) for _ in range(num_subjects)])
        
#         self.enc_eeg = Enc_eeg(num_channels=num_channels, seq_len=sequence_length)
#         self.proj_eeg = Proj_eeg(embedding_dim=self.enc_eeg.output_dim)
        
#         self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
#         self.loss_func = ClipLoss()
    
#     def forward(self, x, subject_ids, text_features=None, img_features=None):
#         if x.dim() == 4:
#             x = x.squeeze(1)
        
#         x = self.encoder(x, None, subject_ids)
#         x = x.unsqueeze(1)
#         eeg_embedding = self.enc_eeg(x)
#         out = self.proj_eeg(eeg_embedding)
        
#         return out


# class HYBRID_MEG(nn.Module):
#     """HYBRID variant for MEG with configurable channels and no position embedding."""
#     def __init__(self, num_channels=271, sequence_length=201, num_subjects=10, num_features=64, num_latents=1024, num_blocks=1, channel_emb_dim=40):
#         super(HYBRID_MEG, self).__init__()
#         default_config = Config()
#         default_config.seq_len = sequence_length
#         default_config.pred_len = sequence_length
#         default_config.d_model = sequence_length
#         default_config.enc_in = num_channels

#         self.subject_layer = SubjectLayers(
#             in_channels=num_channels,
#             out_channels=num_channels,
#             n_subjects=num_subjects,
#             init_id=True
#         )

#         self.gatnn = EEG_GAT(in_channels=sequence_length, out_channels=sequence_length, num_channels=num_channels)
#         self.encoder = iTransformer(default_config, joint_train=False, num_subjects=num_subjects)
#         self.feature_norm = nn.LayerNorm([num_channels, sequence_length])
#         self.feature_norm2 = nn.LayerNorm(sequence_length)

#         self.channel_attention = nn.Sequential(
#             nn.Linear(num_channels, num_channels),
#             nn.GELU(),
#             nn.Linear(num_channels, num_channels),
#             nn.Sigmoid()
#         )

#         self.enc_eeg = Enc_eeg(num_channels=num_channels, seq_len=sequence_length)
#         self.proj_eeg = Proj_eeg(embedding_dim=self.enc_eeg.output_dim)

#     def forward(self, x, subject_ids):
#         if x.dim() == 4:
#             x = x.squeeze(1)

#         x = self.subject_layer(x, subject_ids)

#         x_for_gat = x.unsqueeze(1)
#         gat_out = self.gatnn(x_for_gat)
#         x_gat = gat_out[0] if isinstance(gat_out, tuple) else gat_out
#         x_gat = x_gat.squeeze(1)
#         x_gat = x_gat + x

#         x_trans = self.encoder(x_gat, None, subject_ids)
#         x_trans = x_trans + x_gat

#         channel_weights = self.channel_attention(x.mean(dim=-1))
#         channel_weights = channel_weights.unsqueeze(-1)
#         x_weighted = x_trans * channel_weights
#         x_weighted = x_weighted + x_trans

#         x_weighted = x_weighted.unsqueeze(1)
#         x_weighted = self.feature_norm(x_weighted)
#         eeg_features = self.enc_eeg(x_weighted)
#         eeg_projected = self.proj_eeg(eeg_features)

#         return eeg_projected
