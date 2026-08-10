"""ConvNeXt V2 encoder, FCMAE pre-training wrapper, and classification head.

The architecture (LayerNorm, GRN, ConvNeXt V2 block, hierarchical encoder) follows
Woo et al., "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders"
(CVPR 2023) and is implemented from scratch -- no `timm`, no pre-trained weights.

The *pre-training framework* is a simplification of the paper's FCMAE; see the
"Faithful vs. Simplified" table in the README for exactly what differs and why.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Depth/width configurations from the ConvNeXt V2 paper (Table 7).
CONFIGS = {
    "atto":  {"depths": [2, 2, 6, 2], "dims": [40, 80, 160, 320]},
    "femto": {"depths": [2, 2, 6, 2], "dims": [48, 96, 192, 384]},
    "pico":  {"depths": [2, 2, 6, 2], "dims": [64, 128, 256, 512]},
    "nano":  {"depths": [2, 2, 8, 2], "dims": [80, 160, 320, 640]},
}


class LayerNorm(nn.Module):
    """LayerNorm supporting both channels_last (N, H, W, C) and channels_first (N, C, H, W)."""

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class GRN(nn.Module):
    """Global Response Normalization -- the component ConvNeXt V2 adds over V1.

    Aggregates each channel with an L2 norm over the spatial dims, divides by the
    mean channel response, and re-calibrates. This restores inter-channel competition
    and is what prevents the feature collapse V1 exhibits under masked pre-training.
    Input/output are channels_last (N, H, W, C).
    """

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class Block(nn.Module):
    """ConvNeXt V2 block: 7x7 depthwise conv -> LN -> 1x1 expand -> GELU -> GRN -> 1x1 project."""

    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # -> channels_last
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # -> channels_first
        return shortcut + x


class ConvNeXtV2Encoder(nn.Module):
    """Four-stage hierarchical ConvNeXt V2 encoder.

    Note on naming: the configuration used throughout this project is `atto`
    (depths [2,2,6,2], dims [40,80,160,320], ~3.7M params). The original course
    report called this "Nano"; that was a mislabel -- Nano is depths [2,2,8,2],
    dims [80,160,320,640]. The network itself is unchanged.
    """

    def __init__(self, in_chans=3, variant="atto", depths=None, dims=None):
        super().__init__()
        cfg = CONFIGS[variant]
        depths = depths or cfg["depths"]
        dims = dims or cfg["dims"]
        self.dims = dims

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            self.downsample_layers.append(
                nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                )
            )

        self.stages = nn.ModuleList(
            nn.Sequential(*[Block(dim=dims[i]) for _ in range(depths[i])]) for i in range(4)
        )
        self.final_dim = dims[-1]

    def forward(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return x


class FCMAE(nn.Module):
    """Masked autoencoder wrapper used for self-supervised pre-training.

    SIMPLIFICATION vs. the paper (kept as-submitted so the reported numbers stay valid):
      * masking is per-pixel i.i.d., not 32x32 patch-level;
      * the encoder uses dense convolutions, not sparse submanifold convolutions,
        so information can leak across masked regions;
      * the reconstruction loss is computed over the whole image, not only the
        masked regions;
      * the decoder is a 4-stage transposed-conv stack, not the paper's single
        lightweight block.
    See README "Faithful vs. Simplified".
    """

    def __init__(self, encoder: ConvNeXtV2Encoder):
        super().__init__()
        self.encoder = encoder
        d = encoder.dims
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(d[3], d[2], kernel_size=2, stride=2), Block(d[2]),
            nn.ConvTranspose2d(d[2], d[1], kernel_size=2, stride=2), Block(d[1]),
            nn.ConvTranspose2d(d[1], d[0], kernel_size=2, stride=2), Block(d[0]),
            nn.ConvTranspose2d(d[0], 3, kernel_size=4, stride=4),
        )

    def forward(self, x, mask_ratio=0.6):
        n, _, h, w = x.shape
        keep = (torch.rand(n, 1, h, w, device=x.device) > mask_ratio).float()
        x_masked = x * keep
        reconstruction = self.decoder(self.encoder(x_masked))
        return reconstruction, x, keep


class Classifier(nn.Module):
    """Encoder + LayerNorm + global average pool + dropout + linear head."""

    def __init__(self, encoder: ConvNeXtV2Encoder, num_classes: int, dropout_rate=0.5):
        super().__init__()
        self.encoder = encoder
        self.norm = LayerNorm(encoder.final_dim, eps=1e-6, data_format="channels_first")
        self.dropout = nn.Dropout(p=dropout_rate)
        self.head = nn.Linear(encoder.final_dim, num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.norm(x)
        x = x.mean([-2, -1])
        x = self.dropout(x)
        return self.head(x)

    def freeze_encoder(self):
        """Freeze the backbone for linear probing (the head and its norm stay trainable)."""
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()
        return self


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
