"""
Created on Thu July 2 13:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code for vision models that perform image classification tasks within
NeuronProofreader pipelines.

"""

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuron_proofreader.utils.ml_util import FeedForwardNet

BLOCK_REGISTRY = {}


# --- Convolutional Neural Networks ---
class NewCNN3D(nn.Module):
    """
    Convolutional neural network for 3D images.
    """

    def __init__(
        self,
        input_shape,
        base_channels=16,
        block_type=None,
        center_pool_sigma=0.4,
        channel_multiplier=2,
        depth=5,
        dropout=0.1,
        learnable_center_sigma=True,
        max_channels=256,
        num_single_blocks=2,
        output_dim=1,
        pool_stage_idxs=(2, -1),
        use_double=True,
    ):
        # Call parent class
        nn.Module.__init__(self)

        # Set block type
        if isinstance(block_type, str):
            block_name = block_type
            block_type = BLOCK_REGISTRY[block_name]
        else:
            block_type = block_type or ConvBlock3D
            block_name = self._block_name(block_type)

        # Save model config
        self.config = {
            "input_shape": tuple(input_shape),
            "base_channels": base_channels,
            "block_type": block_name,
            "center_pool_sigma": center_pool_sigma,
            "channel_multiplier": channel_multiplier,
            "depth": depth,
            "dropout": dropout,
            "learnable_center_sigma": learnable_center_sigma,
            "max_channels": max_channels,
            "num_single_blocks": num_single_blocks,
            "output_dim": output_dim,
            "pool_stage_idxs": tuple(pool_stage_idxs),
            "use_double": use_double,
        }

        # Encoder
        self.dropout = dropout
        self.encode = Encoder3D(
            input_shape[0], base_channels, depth,
            block_type=block_type,
            channel_multiplier=channel_multiplier,
            max_channels=max_channels,
            num_single_blocks=num_single_blocks,
            use_double=use_double,
        )

        # Output
        self.pool_stage_idxs = pool_stage_idxs
        stage_channels = [self.encode.blocks[i].out_channels for i in pool_stage_idxs]
        self.center_pools = nn.ModuleList([
            CenterWeightedPool3D(sigma=center_pool_sigma, learnable=learnable_center_sigma)
            for _ in stage_channels
        ])

        total_dim = sum(c * 2 for c in stage_channels)
        self.output = FeedForwardNet(total_dim, output_dim, 3)
        self.apply(self.init_weights)

    @staticmethod
    def _block_name(block_type):
        """
        Reverse-lookup a block class/partial against BLOCK_REGISTRY.
        """
        target = block_type.func if isinstance(block_type, partial) else block_type
        for name, cls in BLOCK_REGISTRY.items():
            if cls is target:
                return name
        raise ValueError(
            f"block_type {target} isn't in BLOCK_REGISTRY, so it can't be "
            f"saved/reloaded from config. Add it to BLOCK_REGISTRY first."
        )

    def save(self, path):
        """
        Saves config (architecture hyperparameters) and weights together.

        Parameters
        ----------
        path : str
            Destination path, e.g. "model.pt".
        """
        torch.save({"config": self.config, "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path, map_location=None):
        """
        Reconstructs a NewCNN3D from a checkpoint saved with `save`.

        Parameters
        ----------
        path : str
            Path to a checkpoint written by `save`.
        map_location : str or torch.device, optional
            Passed through to `torch.load`.

        Returns
        -------
        NewCNN3D
            Model with architecture and weights matching the checkpoint.
        """
        ckpt = torch.load(path, map_location=map_location)
        model = cls(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        return model

    @staticmethod
    def init_weights(m):
        """
        Initializes the weights and biases of a given PyTorch layer.

        Parameters
        ----------
        m : nn.Module
            PyTorch layer or module.
        """
        if isinstance(m, nn.Conv3d):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Passes the given input through this neural network.

        Parameters
        ----------
        x : torch.Tensor
            Input vector of features.

        Returns
        -------
        x : torch.Tensor
            Output of the neural network.
        """
        stages = self.encode(x)
        feats = []
        for idx, pool in zip(self.pool_stage_idxs, self.center_pools):
            s = stages[idx]
            center = pool(s)
            mx = F.adaptive_max_pool3d(s, 1).flatten(1)
            feats.append(torch.cat([center, mx], dim=1))
        return self.output(torch.cat(feats, dim=1))


class Encoder3D(nn.Module):
    """
    Sequence of convolution blocks with growing (capped) channel width.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        depth,
        block_type=None,
        channel_multiplier=2,
        max_channels=256,
        num_single_blocks=2,
        stem_depth=0,
        stem_dilations=None,
        use_double=True,
    ):
        """
        Instantiates an Encoder3D object.

        Parameters
        ----------
        in_channels : int
            Number of channels input to the first block.
        out_channels : int
            Number of channels output by the first block.
        depth : int
            Number of conv blocks in the encoder.
        channel_multiplier : float, optional
            Multiplicative channel growth factor per layer. Default is 2.
        use_double : bool, optional
            True if blocks use double convolution. Default is True.
        max_channels : int, optional
            Cap on channel growth across layers. Default is 128.
        """
        # Call parent class
        super().__init__()

        # Create convolutional blocks
        blocks = list()
        block_type = block_type or ConvBlock3D
        for i in range(depth):
            # Add block
            use_double_i = i > num_single_blocks
            block = block_type(in_channels, out_channels, use_double=use_double_i)
            blocks.append(block)

            # Update channel dimensions
            in_channels = block.out_channels
            out_channels = int(min(out_channels * channel_multiplier, max_channels))

        self.blocks = nn.ModuleList(blocks)
        self.out_channels = self.blocks[-1].out_channels

    def forward(self, x):
        stages = []
        for block in self.blocks:
            x = block(x)
            stages.append(x)
        return stages


# --- Convolutional Blocks ---
class ConvUnit3D(nn.Sequential):
    """
    Conv -> GroupNorm -> (optional) GELU
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        activation=True,
        max_groups=8,
    ):
        # Initializations
        padding = kernel_size // 2
        n_groups = self.get_num_groups(out_channels, max_groups)
        layers = [
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(n_groups, out_channels),
        ]

        if activation:
            layers.append(nn.GELU())

        super().__init__(*layers)

    @staticmethod
    def get_num_groups(num_channels, max_groups=8):
        for g in reversed(range(1, max_groups + 1)):
            if num_channels % g == 0:
                return g
        return 1


def register_block(name):
    def wrap(cls):
        BLOCK_REGISTRY[name] = cls
        return cls
    return wrap


@register_block("conv")
class ConvBlock3D(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        use_double=True,
    ):
        super().__init__()

        self.out_channels = out_channels

        layers = [
            ConvUnit3D(
                in_channels,
                out_channels,
                kernel_size,
            )
        ]

        if use_double:
            layers.append(
                ConvUnit3D(
                    out_channels,
                    out_channels,
                    kernel_size,
                )
            )

        self.conv = nn.Sequential(*layers)
        self.pool = nn.MaxPool3d(2)

    def forward(self, x):
        x = self.conv(x)
        return self.pool(x)


@register_block("res")
class ResConvBlock3D(nn.Module):
    """
    3D convolutional block with a residual connection around the conv unit(s),
    followed by downsampling. The skip path projects channels via a 1x1 conv
    when in_channels != out_channels (or is an identity otherwise), so the
    residual add is always shape-compatible.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        use_double=True,
    ):
        """
        Instantiates a ResConvBlock3D object.

        Parameters
        ----------
        in_channels : int
            Number of input channels to this block.
        out_channels : int
            Number of output channels from this block.
        kernel_size : int, optional
            Size of kernel used on convolutional layers. Default is 3.
        use_double : bool, optional
            Indication of whether to apply a second conv+norm before the
            residual add. Default is True.
        """
        # Call parent class
        super().__init__()

        self.out_channels = out_channels

        layers = [
            ConvUnit3D(
                in_channels,
                out_channels,
                kernel_size,
            )
        ]

        if use_double:
            layers.append(
                ConvUnit3D(
                    out_channels,
                    out_channels,
                    kernel_size,
                    activation=False,
                )
            )

        self.main = nn.Sequential(*layers)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            )
        )

        self.act = nn.GELU()
        self.pool = nn.MaxPool3d(2)

    def forward(self, x):
        x = self.main(x) + self.skip(x)
        x = self.act(x)
        return self.pool(x)


@register_block("se_res")
class SEResConvBlock3D(ResConvBlock3D):
    """
    ResConvBlock3D with squeeze-excitation applied to the residual branch
    before the skip add.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        use_double=True,
        reduction=8,
    ):
        super().__init__(in_channels, out_channels, kernel_size, use_double)
        self.se = SqueezeExcite3D(out_channels, reduction)

    def forward(self, x):
        x = self.se(self.main(x)) + self.skip(x)
        x = self.act(x)
        return self.pool(x)


class SqueezeExcite3D(nn.Module):
    """
    Channel attention: squeeze via global average pooling, excite via a
    small bottleneck MLP gate.
    """

    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c = x.shape[:2]
        gate = F.adaptive_avg_pool3d(x, 1).view(b, c)
        gate = self.fc(gate).view(b, c, 1, 1, 1)
        return x * gate


@register_block("convnext")
class ConvNeXtDownBlock3D(nn.Module):
    """
    Adapts ConvNeXtBlock3D (fixed channels, no downsampling) to the
    Encoder3D block interface via a 1x1 channel projection and pooling.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        use_double=True,
        expansion=4,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.proj = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        )
        self.block = ConvNeXtBlock3D(out_channels, expansion=expansion)
        self.pool = nn.MaxPool3d(2)

    def forward(self, x):
        x = self.proj(x)
        x = self.block(x)
        return self.pool(x)


class ConvNeXtBlock3D(nn.Module):

    def __init__(self, channels, expansion=4):
        # Call parent class
        super().__init__()

        # Create block
        hidden = expansion * channels
        self.block = nn.Sequential(
            # Depthwise convolution
            nn.Conv3d(
                channels,
                channels,
                kernel_size=7,
                padding=3,
                groups=channels,
                bias=False,
            ),
            nn.GroupNorm(1, channels),

            # Pointwise expansion
            nn.Conv3d(channels, hidden, kernel_size=1),
            nn.GELU(),

            # Pointwise projection
            nn.Conv3d(hidden, channels, kernel_size=1),
        )

    def forward(self, x):
        return x + self.block(x)


class CenterWeightedPool3D(nn.Module):
    """
    Global pooling with a Gaussian weight centered on the patch, so voxels
    near the center (e.g. the candidate merge/split point) contribute more
    than voxels near the border. Weights are per-spatial-position only
    (shared across channels) and normalize to sum to 1.
    """

    def __init__(self, sigma=0.4, learnable=True):
        """
        Parameters
        ----------
        sigma : float, optional
            Gaussian std, as a fraction of the half-extent along each axis
            (coords run -1..1, so 0.4 means weight falls to ~1/e at 40% of
            the way to the edge). Smaller = tighter focus on center.
            Default is 0.4.
        learnable : bool, optional
            If True, sigma is a learned scalar so the model can widen or
            narrow the focus during training. Default is True.
        """
        super().__init__()
        self._dist_cache = {}
        init = torch.log(torch.tensor(float(sigma)))
        if learnable:
            self.log_sigma = nn.Parameter(init)
        else:
            self.register_buffer("log_sigma", init, persistent=False)

    def _dist_sq(self, shape, device, dtype):
        key = (shape, device)
        if key not in self._dist_cache:
            coords = [torch.linspace(-1, 1, s, device=device, dtype=dtype) for s in shape]
            grid = torch.stack(torch.meshgrid(*coords, indexing="ij"), dim=0)
            self._dist_cache[key] = (grid ** 2).sum(0)
        return self._dist_cache[key]

    def forward(self, x):
        b, c, d, h, w = x.shape
        dist_sq = self._dist_sq((d, h, w), x.device, x.dtype)
        sigma = self.log_sigma.exp().clamp(min=1e-3)
        weights = torch.exp(-dist_sq / (2 * sigma ** 2))
        weights = (weights / weights.sum()).view(1, 1, d, h, w)
        return (x * weights).sum(dim=(2, 3, 4))


# --- Vision Transformers ---
class ViT3D(nn.Module):

    def __init__(
        self,
        input_shape,
        patch_size=8,
        embed_dim=128,
        depth=6,
        num_heads=4,
        mlp_ratio=4.0,
        dropout=0.1,
        output_dim=1,
    ):
        # Call parent class
        super().__init__()
        in_channels, d, h, w = input_shape
        assert d % patch_size == 0 and h % patch_size == 0 and w % patch_size == 0, (
            f"input_shape spatial dims must be divisible by patch_size={patch_size}"
        )
        num_patches = (d // patch_size) * (h // patch_size) * (w // patch_size)

        self.patch_embed = PatchEmbed3D(in_channels, embed_dim, patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = FeedForwardNet(embed_dim, output_dim, 3)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        b = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.encoder(x)
        return self.head(self.norm(x[:, 0]))


class PatchEmbed3D(nn.Module):
    """
    Splits a volume into non-overlapping cubic patches, linearly embeds
    each as a token.
    """

    def __init__(self, in_channels, embed_dim, patch_size=4):
        super().__init__()
        self.proj = nn.Conv3d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)
