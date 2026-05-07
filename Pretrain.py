"""
Mask-Reconstruction Pretraining for EEG Encoders
"""

from ast import arg
from mimetypes import init
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import argparse
import time
import datetime
import matplotlib.pyplot as plt
import timm.optim.optim_factory as optim_factory
from timm.models.vision_transformer import Block
from einops import rearrange

from train import NICE, ATMS, MCRL, HYBRID, Config
from modules import weights_init_tensor


class NativeScaler:
    state_dict_key = "amp_scaler"
    
    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()
    
    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                # Get grad norm without torch._six
                if parameters is None:
                    parameters = []
                parameters = [p for p in parameters if p.grad is not None]
                if len(parameters) == 0:
                    norm = torch.tensor(0.)
                else:
                    device = parameters[0].grad.device
                    total_norm = torch.norm(
                        torch.stack([torch.norm(p.grad.detach(), 2.0).to(device) for p in parameters]), 
                        2.0
                    )
                    norm = total_norm
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm
    
    def state_dict(self):
        return self._scaler.state_dict()
    
    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


# Helper function for adding weight decay
def add_weight_decay(model, weight_decay=0.05, skip_list=('bias', 'norm')):
    """Add weight decay to parameters except for bias and norm layers"""
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(skip_name in name for skip_name in skip_list):
            no_decay.append(param)
        else:
            decay.append(param)
    
    return [
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.},
    ]


def generate_masked_and_recon(model, eeg, subject_ids, mask):
    """Generate masked EEG and reconstructed EEG for a batch."""
    model.eval()
    with torch.no_grad():
        if not torch.is_tensor(mask):
            mask = model.random_masking(eeg.shape[0], mask, eeg.device)
        latent = model.forward_encoder(eeg, subject_ids)
        pred = model.forward_decoder(latent)

        target_patches = model.patchify(eeg)
        masked_patches = target_patches.clone()
        # Expand mask to match the last dimension of masked_patches
        mask_expanded = mask.unsqueeze(-1).expand(-1, -1, masked_patches.shape[-1]).bool()
        masked_patches[mask_expanded] = 0
        masked_eeg = model.unpatchify(masked_patches)

        recon_eeg = model.unpatchify(pred)
    return masked_eeg, recon_eeg, mask


def save_waveform_triplet_plot(output_dir, original, masked, recon, fs, channel_idx, mask=None, patch_size=5, trial_idx=0):
    """Save a 3x1 plot for original, masked, and reconstructed waveforms.
    
    Args:
        mask: Binary mask array (B, num_patches) where 1 = masked, 0 = kept
        trial_idx: Trial index for filename
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Align reconstructed data range to original data range
    # Get statistics from original (unmasked) data
    original_channel = original[channel_idx]
    recon_channel = recon[channel_idx]
    
    # Compute mean and std of original data
    orig_mean = np.mean(original_channel)
    orig_std = np.std(original_channel)
    
    # Normalize recon to original scale
    if orig_std > 0:
        recon_normalized = (recon_channel - np.mean(recon_channel)) / (np.std(recon_channel) + 1e-8)
        recon_channel_aligned = recon_normalized * orig_std + orig_mean
    else:
        recon_channel_aligned = recon_channel
    
    t = np.arange(original.shape[-1]) / fs

    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    
    # Original waveform
    axes[0].plot(t, original_channel)
    axes[0].set_title("Original")
    axes[0].set_ylim([np.min(original_channel), np.max(original_channel)])

    # Masked waveform - show original with transparency and highlight kept regions with bold line
    # First plot the full original signal with transparency
    axes[1].plot(t, original_channel, color='gray', alpha=0.5, linewidth=1, label='Original')
    
    if mask is not None:
        # mask shape: (1, num_patches), convert to samples
        mask_np = mask.cpu().numpy()[0]  # (num_patches,)
        # Expand mask back to sample level
        num_samples = masked.shape[-1]
        samples_per_patch = num_samples // len(mask_np)
        mask_expanded = np.repeat(mask_np, samples_per_patch)
        
        # Plot kept regions (mask==0) with bold line
        kept_data = masked[channel_idx].copy()
        kept_mask = (mask_expanded == 0)
        
        # Use masked array to plot only kept regions
        kept_masked = np.ma.array(kept_data, mask=~kept_mask)
        axes[1].plot(t, kept_masked, color='C0', linewidth=2, label='Kept')
    else:
        axes[1].plot(t, masked[channel_idx], color='C0', linewidth=2)
    
    axes[1].set_title("Masked")
    axes[1].set_ylim([np.min(original_channel), np.max(original_channel)])
    axes[1].legend(loc='upper right', fontsize=8)

    # Reconstructed waveform (aligned to original scale)
    # Plot original with transparency first
    axes[2].plot(t, original_channel, color='gray', alpha=0.5, linewidth=1, label='Original')
    # Then plot reconstructed signal
    axes[2].plot(t, recon_channel_aligned, color='C1', linewidth=2, label='Reconstructed')
    axes[2].set_title("Reconstructed")
    axes[2].set_ylim([np.min(original_channel), np.max(original_channel)])
    axes[2].legend(loc='upper right', fontsize=8)

    axes[2].set_xlabel("Time (s)")
    # for ax in axes:
    #     ax.set_ylabel("Amplitude")

    out_path = os.path.join(output_dir, f"waveform_triplet_trial{trial_idx:02d}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Waveform triplet (trial {trial_idx}) saved to {out_path}")


def get_1d_sincos_pos_embed(embed_dim, length, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_l = np.arange(length, dtype=np.float32)

    grid_l = grid_l.reshape([1, length])
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid_l)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

# EEG Dataset for MAE Pretraining
class EEGDatasetMAE(Dataset):
    """
    EEG dataset for MAE pretraining.
    Data structure follows MCRL data loading pattern.
    """
    def __init__(self, data_path, subjects, transform=None):
        super().__init__()
        self.data_path = data_path
        self.subjects = subjects
        self.transform = transform
        
        # Load all training data
        self.data = []
        self.subject_ids = []
        
        for sub_id in subjects:
            sub_path = os.path.join(data_path, f'sub-{sub_id:02d}', 'preprocessed_eeg_training.npy')
            if os.path.exists(sub_path):
                sub_data = np.load(sub_path, allow_pickle=True)
                sub_data = sub_data['preprocessed_eeg_data']
                # Average across trials and add channel dimension
                sub_data = np.mean(sub_data, axis=1)  # (n_samples, n_channels, n_timepoints)
                
                # Store data
                for i in range(len(sub_data)):
                    self.data.append(sub_data[i])
                    self.subject_ids.append(sub_id - 1)  # 0-indexed
                    
                print(f'Loaded subject {sub_id}: {len(sub_data)} samples')
        
        self.data = np.array(self.data)
        self.subject_ids = np.array(self.subject_ids)
        print(f'Total dataset size: {len(self.data)} samples from {len(subjects)} subjects')
        print(f'Data shape: {self.data.shape}')  # Should be (n_samples, 63, 250)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        eeg = self.data[idx]  # (63, 250)
        subject_id = self.subject_ids[idx]
        
        if self.transform:
            eeg = self.transform(eeg)
        
        eeg = torch.FloatTensor(eeg)
        
        return {
            'eeg': eeg,
            'subject_id': torch.LongTensor([subject_id])
        }

# MAE Model with EEG Encoders
class MAEforEEGEncoder(nn.Module):

    def __init__(self, 
                 encoder_type='NICE',
                 num_channels=63,
                 sequence_length=250,
                 num_subjects=10,
                 decoder_embed_dim=512,
                 decoder_depth=4,
                 decoder_num_heads=8,
                 mlp_ratio=4.,
                 mask_ratio=0.75,
                 patch_size=5,
                 mask_fill_type='zero'):
        super().__init__()
        self.encoder_type = encoder_type
        self.num_channels = num_channels
        self.sequence_length = sequence_length
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.mask_fill_type = mask_fill_type

        if encoder_type == 'NICE':
            self.encoder = NICE(
                num_channels=num_channels,
                sequence_length=sequence_length,
                num_subjects=num_subjects
            )
            encoder_output_dim = 1024
        elif encoder_type == 'ATMS':
            self.encoder = ATMS(
                num_channels=num_channels,
                sequence_length=sequence_length,
                num_subjects=num_subjects
            )
            encoder_output_dim = 1024
        elif encoder_type == 'MCRL':
            self.encoder = MCRL(
                num_channels=num_channels,
                sequence_length=sequence_length,
                num_subjects=num_subjects
            )
            encoder_output_dim = 1024
        elif encoder_type == 'HYBRID':
            self.encoder = HYBRID(
                num_channels=num_channels,
                sequence_length=sequence_length,
                num_subjects=num_subjects
            )
            encoder_output_dim = 1024
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

        self.encoder_output_dim = encoder_output_dim
        self.num_patches = sequence_length // patch_size

        self.decoder_embed = nn.Linear(encoder_output_dim, decoder_embed_dim, bias=True)
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, decoder_embed_dim), 
            requires_grad=False
        )
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, 
                  qkv_bias=True, norm_layer=nn.LayerNorm)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, num_channels * patch_size, bias=True)
        self.initialize_weights()

    def initialize_weights(self):
        decoder_pos_embed = get_1d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], 
            self.num_patches, 
            cls_token=False
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        # Initialize all parameters using weights_init_tensor
        for name, p in self.named_parameters():
            weights_init_tensor(name, p.data)

    def patchify(self, eeg):
        B, C, T = eeg.shape
        num_patches = T // self.patch_size
        eeg = eeg.reshape(B, C, num_patches, self.patch_size)
        eeg = eeg.permute(0, 2, 1, 3)
        eeg = eeg.reshape(B, num_patches, C * self.patch_size)
        return eeg

    def unpatchify(self, x):
        B, num_patches, _ = x.shape
        C = self.num_channels
        x = x.reshape(B, num_patches, C, self.patch_size)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(B, C, num_patches * self.patch_size)
        return x

    def forward_encoder(self, eeg, subject_ids):
        features = self.encoder(eeg, subject_ids)  # (B, 1024)
        features = features.unsqueeze(1).expand(-1, self.num_patches, -1)  # (B, num_patches, 1024)
        return features

    def forward_decoder(self, x):
        x = self.decoder_embed(x)
        x = x + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        return x

    def forward_loss(self, eeg, pred, mask):
        target = self.patchify(eeg)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        # Compute reconstruction loss over all patches (not masked-only).
        loss = loss.mean()
        return loss

    def random_masking(self, batch_size, mask_ratio, device):
        """Generate binary mask tensor with 1 for masked patches and 0 for kept patches."""
        len_keep = int(self.num_patches * (1 - mask_ratio))
        noise = torch.rand(batch_size, self.num_patches, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        mask = torch.ones([batch_size, self.num_patches], device=device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return mask

    def forward(self, eeg, subject_ids, mask_ratio=None, mask=None):
        # eeg: (B, C, T), mask: (B, num_patches), 1=mask, 0=keep
        if mask is None:
            if mask_ratio is None:
                mask_ratio = self.mask_ratio
            mask = self.random_masking(eeg.shape[0], mask_ratio, eeg.device)

        target_patches = self.patchify(eeg)
        masked_patches = target_patches.clone()
        mask_expanded = mask.unsqueeze(-1).expand(-1, -1, masked_patches.shape[-1]).bool()
        if self.mask_fill_type == 'zero':
            masked_patches[mask_expanded] = 0
        elif self.mask_fill_type == 'noise':
            noise = torch.randn_like(masked_patches)
            masked_patches[mask_expanded] = noise[mask_expanded]
        else:
            raise ValueError(f"Unknown mask_fill_type: {self.mask_fill_type}")
        eeg_masked = self.unpatchify(masked_patches)

        latent = self.forward_encoder(eeg_masked, subject_ids)  # (B, num_patches, 1024)
        pred = self.forward_decoder(latent)  # (B, num_patches, C*patch_size)
        loss = self.forward_loss(eeg, pred, mask)
        return loss, pred, mask


def train_one_epoch(model, dataloader, optimizer, device, epoch, loss_scaler, config):
    model.train()
    optimizer.zero_grad()
    
    total_loss = []
    
    for batch_idx, data_dict in enumerate(dataloader):
        eeg = data_dict['eeg'].to(device)  # (B, C, T)
        subject_ids = data_dict['subject_id'].squeeze(1).to(device)  # (B,)
        
        with torch.amp.autocast('cuda', enabled=True):
            loss, pred, mask = model(eeg, subject_ids, mask_ratio=config.mask_ratio)
        
        loss_value = loss.item()
        
        if not np.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)
        
        # Backward with gradient scaling
        loss_scaler(loss, optimizer, parameters=model.parameters(), 
                   clip_grad=config.clip_grad)
        optimizer.zero_grad()
        
        total_loss.append(loss_value)
        
        # if batch_idx % 50 == 0:
        #     print(f'Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss_value:.6f}')
    
    avg_loss = np.mean(total_loss)
    # print(f'[Epoch {epoch}] Average Loss: {avg_loss:.6f}')
    
    return avg_loss


def validate_one_epoch(model, dataloader, device, epoch, config):
    model.eval()
    total_loss = []

    with torch.no_grad():
        for data_dict in dataloader:
            eeg = data_dict['eeg'].to(device)  # (B, C, T)
            subject_ids = data_dict['subject_id'].squeeze(1).to(device)  # (B,)

            with torch.amp.autocast('cuda', enabled=True):
                loss, _, _ = model(eeg, subject_ids, mask_ratio=config.mask_ratio)

            total_loss.append(loss.item())

    avg_val_loss = np.mean(total_loss)
    # print(f'[Epoch {epoch}] Validation Loss: {avg_val_loss:.6f}')
    return avg_val_loss


def save_checkpoint(model, optimizer, loss_scaler, epoch, save_dir, encoder_type, subject=None):
    """Save model checkpoint"""
    os.makedirs(save_dir, exist_ok=True)
    
    if subject is not None:
        filename = f'mae_pretrain_{encoder_type}_sub{subject:02d}_epoch{epoch:03d}.pth'
    else:
        filename = f'mae_pretrain_{encoder_type}_epoch{epoch:03d}.pth'
    
    save_path = os.path.join(save_dir, filename)
    
    checkpoint = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scaler_state': loss_scaler.state_dict(),
        'encoder_type': encoder_type
    }
    
    torch.save(checkpoint, save_path)
    print(f'Checkpoint saved to {save_path}')
    
    return save_path


class MAEConfig:
    def __init__(self):
        # Data
        self.data_path = './Things_EEG2/Preprocessed_data_250Hz/'
        self.subjects = list(range(1, 11))  # Subjects 1-10
        self.encoder_type = 'HYBRID'  # NICE, ATMS, MCRL, HYBRID
        # Model
        self.num_channels = 63
        self.sequence_length = 250
        self.num_subjects = 10
        self.patch_size = 5  # Default: 250 / 5 = 50 patches
        self.mask_ratio = 0.3
        self.mask_fill_type = 'noise'
        
        # Decoder
        self.decoder_embed_dim = 512
        self.decoder_depth = 4
        self.decoder_num_heads = 8
        self.mlp_ratio = 4.0
        
        # Training
        self.batch_size = 128
        self.num_epochs = 100
        self.lr = 1e-4
        self.weight_decay = 0.05
        self.clip_grad = 1.0
        self.warmup_epochs = 5
        self.save_freq = 10
        self.resume = ''
        self.val_split = 0.2
        self.early_stopping_patience = 10
        
        self.output_path = './Things_EEG2/results/mae_eeg_pretrain/'
        self.checkpoint_dir = None  # Will be set dynamically in main() based on parameters
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Waveform saving
        self.save_waveforms = False
        self.waveform_channel = -1


def get_args_parser():
    parser = argparse.ArgumentParser('MAE EEG Pretraining', add_help=False)
    
    # Model
    parser.add_argument('--encoder_type', type=str, default='HYBRID',
                       choices=['NICE', 'ATMS', 'MCRL', 'HYBRID'],
                       help='Type of EEG encoder to use')
    parser.add_argument('--mask_ratio', type=float, default=0.3)
    parser.add_argument('--mask_fill_type', type=str, default='noise', choices=['zero', 'noise'],
                       help='How to fill masked patches before encoder: zero or noise')
    parser.add_argument('--patch_size', type=int, default=5)
    # parser.add_argument('--patch_size', type=int, default=2)
    parser.add_argument('--decoder_embed_dim', type=int, default=512)
    parser.add_argument('--decoder_depth', type=int, default=4)
    parser.add_argument('--decoder_num_heads', type=int, default=8)
    
    # Training
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='Validation split ratio (e.g., 0.2 means 8:2 train/val split)')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                       help='Stop training if val loss does not improve for N epochs')
    
    parser.add_argument('--data_path', type=str,
                       default='./Things_EEG2/Preprocessed_data_250Hz/')
    parser.add_argument('--subjects', type=str, default='1-10',
                       help='Subject range, e.g., "1-10" or "1,2,3"')
    
    parser.add_argument('--output_path', type=str,
                       default='./Things_EEG/results/mae_eeg_pretrain1/')
    parser.add_argument('--save_freq', type=int, default=10,
                       help='Save checkpoint every N epochs')
    
    parser.add_argument('--resume', type=str, default='',
                       help='Path to checkpoint to resume from')

    parser.add_argument('--save_waveforms', default = False, action='store_true',
                       help='Save original/masked/reconstructed waveforms for first subject')
    parser.add_argument('--waveform_channel', type=int, default=-1,
                       help='Channel index to plot (default: last channel)')
    
    return parser


def parse_subjects(subjects_str):
    """Parse subject string like '1-10' or '1,2,3' into list"""
    if '-' in subjects_str:
        start, end = subjects_str.split('-')
        return list(range(int(start), int(end) + 1))
    else:
        return [int(s) for s in subjects_str.split(',')]


def main(config):
    print('=' * 80)
    print('MAE Pretraining for EEG Encoders')
    print('=' * 80)
    print(f'Encoder Type: {config.encoder_type}')
    print(f'Subjects: {config.subjects}')
    print(f'Batch Size: {config.batch_size}')
    print(f'Epochs: {config.num_epochs}')
    print(f'Mask Ratio: {config.mask_ratio}')
    print(f'Output Path: {config.output_path}')
    print('=' * 80)
    
    device = torch.device(config.device)
    
    # Create output directory
    os.makedirs(config.output_path, exist_ok=True)
    
    # Set checkpoint_dir with parameters in name
    checkpoint_subdir = f'checkpoints_mr{config.mask_ratio}_embed{config.decoder_embed_dim}_depth{config.decoder_depth}'
    config.checkpoint_dir = os.path.join(config.output_path, checkpoint_subdir)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # Train each subject separately
    for subject_id in config.subjects:
        print('\n' + '=' * 80)
        print(f'Training Subject {subject_id}/{max(config.subjects)}')
        print('=' * 80)
        
        # Create dataset and dataloader for this subject only
        dataset = EEGDatasetMAE(
            data_path=config.data_path,
            subjects=[subject_id]  # Single subject
        )
    
        dataset_size = len(dataset)
        val_size = max(1, int(dataset_size * config.val_split))
        train_size = dataset_size - val_size
        if train_size == 0:
            train_size = dataset_size - 1
            val_size = 1

        generator = torch.Generator().manual_seed(42)
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        print(f'Dataset size: {len(dataset)}')
        print(f'Train/Val split: {train_size}/{val_size} (8:2)')
        print(f'Number of train batches: {len(train_loader)}')
        print(f'Number of val batches: {len(val_loader)}')
        
        # Create model
        model = MAEforEEGEncoder(
            encoder_type=config.encoder_type,
            num_channels=config.num_channels,
            sequence_length=config.sequence_length,
            num_subjects=config.num_subjects,
            decoder_embed_dim=config.decoder_embed_dim,
            decoder_depth=config.decoder_depth,
            decoder_num_heads=config.decoder_num_heads,
            mlp_ratio=config.mlp_ratio,
            mask_ratio=config.mask_ratio,
            patch_size=config.patch_size,
            mask_fill_type=config.mask_fill_type
        )
        model.to(device)
        
        print(f'Model created: {config.encoder_type}')
        print(f'Number of parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M')
        
        # Optimizer
        if optim_factory is not None and hasattr(optim_factory, 'add_weight_decay'):
            param_groups = optim_factory.add_weight_decay(model, config.weight_decay)
        else:
            param_groups = add_weight_decay(model, config.weight_decay)
        optimizer = torch.optim.AdamW(param_groups, lr=config.lr, betas=(0.9, 0.95))
        
        # Loss scaler
        loss_scaler = NativeScaler()
        
        start_epoch = 0

        # Prepare waveform samples for first subject only (5 trials)
        waveform_samples_list = []
        waveform_subject_ids = None
        waveform_originals = []
        waveform_maskeds = []
        waveform_masks = []
        if config.save_waveforms and subject_id == config.subjects[0]:
            num_trials = min(5, len(dataset))
            for trial_idx in range(num_trials):
                waveform_sample = dataset[trial_idx]['eeg'].unsqueeze(0).to(device)
                if waveform_subject_ids is None:
                    waveform_subject_ids = dataset[trial_idx]['subject_id'].to(device)
                
                waveform_original = waveform_sample.detach().cpu().squeeze(0).numpy()
                masked_initial, _, mask_initial = generate_masked_and_recon(
                    model, waveform_sample, waveform_subject_ids, config.mask_ratio
                )
                waveform_masked = masked_initial.detach().cpu().squeeze(0).numpy()
                
                waveform_samples_list.append(waveform_sample)
                waveform_originals.append(waveform_original)
                waveform_maskeds.append(waveform_masked)
                waveform_masks.append(mask_initial)
        
        # Resume from checkpoint if specified
        if hasattr(config, 'resume') and config.resume:
            print(f'Resuming from checkpoint: {config.resume}')
            checkpoint = torch.load(config.resume, map_location='cpu')
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            loss_scaler.load_state_dict(checkpoint['scaler_state'])
            start_epoch = checkpoint['epoch'] + 1
            print(f'Resumed from epoch {start_epoch}')
        
        # Training loop
        print(f'Starting training for Subject {subject_id}...')
        start_time = time.time()
    
        best_loss = float('inf')
        best_epoch = -1
        best_state = None
        early_stop_counter = 0
        for epoch in range(start_epoch, config.num_epochs):
            # Adjust learning rate (warmup + cosine decay)
            if epoch < config.warmup_epochs:
                lr = config.lr * (epoch + 1) / config.warmup_epochs
            else:
                lr = config.lr * 0.5 * (1 + np.cos(np.pi * (epoch - config.warmup_epochs) / 
                                                    (config.num_epochs - config.warmup_epochs)))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            # print(f'\nEpoch {epoch}/{config.num_epochs}, LR: {lr:.6f}')
            train_loss = train_one_epoch(
                model, train_loader, optimizer, device, epoch, loss_scaler, config
            )
            val_loss = validate_one_epoch(
                model, val_loader, device, epoch, config
            )
            print(f'[Epoch {epoch}] Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                early_stop_counter = 0
                # Save best model state
                best_state = {
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scaler_state': loss_scaler.state_dict(),
                    'encoder_type': config.encoder_type
                }
            else:
                early_stop_counter += 1

            if early_stop_counter >= config.early_stopping_patience:
                print(f'Early stopping triggered at epoch {epoch} (no val loss improvement for {config.early_stopping_patience} epochs).')
                break
        total_time = time.time() - start_time
        print(f'\nSubject {subject_id} training completed in {str(datetime.timedelta(seconds=int(total_time)))}')
        # Save only the best model for this subject
        if best_state is not None:
            best_path = os.path.join(config.checkpoint_dir, f'mae_pretrain_{config.encoder_type}_mr{config.mask_ratio}_embed{config.decoder_embed_dim}_depth{config.decoder_depth}_sub{subject_id:02d}_best.pth')
            torch.save(best_state, best_path)
            print(f'Subject {subject_id} best model (epoch {best_epoch}, best val loss {best_loss:.6f}) saved to {best_path}')

        # Save waveform plots for the first subject only (5 trials)
        if config.save_waveforms and subject_id == config.subjects[0]:
            output_dir = os.path.join(config.output_path, 'waveforms')
            for trial_idx in range(len(waveform_samples_list)):
                _, waveform_recon, _ = generate_masked_and_recon(
                    model, waveform_samples_list[trial_idx], waveform_subject_ids, config.mask_ratio
                )
                waveform_recon = waveform_recon.detach().cpu().squeeze(0).numpy()
                
                # Save individual plot for each trial
                save_waveform_triplet_plot(
                    output_dir,
                    waveform_originals[trial_idx],
                    waveform_maskeds[trial_idx],
                    waveform_recon,
                    fs=250,
                    channel_idx=config.waveform_channel,
                    mask=waveform_masks[trial_idx],
                    patch_size=config.patch_size,
                    trial_idx=trial_idx
                )
    
    print('\n' + '=' * 80)
    print('All subjects training completed!')
    print('=' * 80)


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    
    # Create config from args
    config = MAEConfig()
    
    # Update config with args
    if args.encoder_type:
        config.encoder_type = args.encoder_type
    if args.mask_ratio:
        config.mask_ratio = args.mask_ratio
    if hasattr(args, 'mask_fill_type') and args.mask_fill_type:
        config.mask_fill_type = args.mask_fill_type
    if args.patch_size:
        config.patch_size = args.patch_size
    if args.decoder_embed_dim:
        config.decoder_embed_dim = args.decoder_embed_dim
    if args.decoder_depth:
        config.decoder_depth = args.decoder_depth
    if args.decoder_num_heads:
        config.decoder_num_heads = args.decoder_num_heads
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.num_epochs:
        config.num_epochs = args.num_epochs
    if args.lr:
        config.lr = args.lr
    if args.weight_decay:
        config.weight_decay = args.weight_decay
    if args.warmup_epochs:
        config.warmup_epochs = args.warmup_epochs
    if hasattr(args, 'val_split') and args.val_split:
        config.val_split = args.val_split
    if hasattr(args, 'early_stopping_patience') and args.early_stopping_patience:
        config.early_stopping_patience = args.early_stopping_patience
    if args.data_path:
        config.data_path = args.data_path
    if args.subjects:
        config.subjects = parse_subjects(args.subjects)
    if args.output_path:
        config.output_path = args.output_path
    if hasattr(args, 'save_freq') and args.save_freq:
        config.save_freq = args.save_freq
    if hasattr(args, 'resume') and args.resume:
        config.resume = args.resume
    if hasattr(args, 'save_waveforms') and args.save_waveforms:
        config.save_waveforms = True
    if hasattr(args, 'waveform_channel'):
        config.waveform_channel = args.waveform_channel
    
    # Always set checkpoint_dir dynamically based on MAE parameters
    checkpoint_subdir = f'checkpoints_mr{config.mask_ratio}_embed{config.decoder_embed_dim}_depth{config.decoder_depth}'
    config.checkpoint_dir = os.path.join(config.output_path, checkpoint_subdir)
    
    main(config)
