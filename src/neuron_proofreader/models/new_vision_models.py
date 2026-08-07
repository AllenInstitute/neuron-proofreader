"""
Created on Thu July 2 13:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code for vision models that perform image classification tasks within
NeuronProofreader pipelines.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuron_proofreader.utils.ml_util import FeedForwardNet


# --- Convolutional Neural Networks ---
class CNN3D(nn.Module):
    """
    Convolutional neural network for 3D images.
    """

    def __init__(
        self,
        input_shape,
        base_channels=32,
        channel_multiplier=2,
        depth=5,
        dropout=0.1,
        max_channels=256,
        output_dim=1,
        pool_stage_idxs=(-3, -2, -1),
        use_double=True,
        use_se=True,
    ):
        # Call parent class
        nn.Module.__init__(self)

        # Save model config
        self.config = {
            "input_shape": tuple(input_shape),
            "base_channels": base_channels,
            "channel_multiplier": channel_multiplier,
            "depth": depth,
            "dropout": dropout,
            "max_channels": max_channels,
            "output_dim": output_dim,
            "pool_stage_idxs": tuple(pool_stage_idxs),
            "use_double": use_double,
            "use_se": use_se,
        }

        # Encoder
        self.drop = nn.Dropout(dropout)
        self.encode = Encoder3D(
            input_shape[0],
            base_channels,
            depth,
            channel_multiplier=channel_multiplier,
            max_channels=max_channels,
            use_double=use_double,
            use_se=use_se,
        )

        # Output
        self.pool_stage_idxs = pool_stage_idxs
        total_dim = sum(
            self.encode.blocks[i].out_channels for i in pool_stage_idxs
        )
        self.output = FeedForwardNet(total_dim, output_dim, 3)
        self.apply(self.init_weights)

    def save(self, path):
        """
        Saves config (architecture hyperparameters) and weights together.

        Parameters
        ----------
        path : str
            Destination path, e.g. "model.pt".
        """
        torch.save(
            {"config": self.config, "state_dict": self.state_dict()}, path
        )

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
        config = ckpt["config"]
        config.pop("num_single_conv_blocks", None)
        config.pop("num_single_blocks", None)
        config.pop("center_pool_sigma", None)
        model = cls(**config)
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
        feats = [
            F.adaptive_avg_pool3d(stages[i], 1).flatten(1)
            for i in self.pool_stage_idxs
        ]
        x = self.drop(torch.cat(feats, dim=1))
        return self.output(x)


class Encoder3D(nn.Module):
    """
    Sequence of convolution blocks with growing (capped) channel width.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        depth,
        channel_multiplier=2,
        max_channels=256,
        stem_depth=0,
        stem_dilations=None,
        use_double=True,
        use_se=True,
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
        for _ in range(depth):
            # Add block
            block = ConvBlock3D(
                in_channels,
                out_channels,
                use_double=use_double,
                use_se=use_se,
            )
            blocks.append(block)

            # Update channel dimensions
            in_channels = block.out_channels
            out_channels = int(
                min(out_channels * channel_multiplier, max_channels)
            )

        self.blocks = nn.ModuleList(blocks)
        self.out_channels = self.blocks[-1].out_channels

    def forward(self, x):
        stages = []
        for block in self.blocks:
            x = block(x)
            stages.append(x)
        return stages


# --- Convolutional Blocks ---
class SEBlock3D(nn.Module):
    """
    Squeeze-and-Excitation block: recalibrates channel responses by learning
    which feature maps are most informative for the current task.
    """

    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c = x.shape[:2]
        w = x.mean(dim=(2, 3, 4))
        w = self.fc(w).view(b, c, 1, 1, 1)
        return x * w


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


class ConvBlock3D(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        use_double=True,
        use_se=True,
    ):
        # Call parent class
        super().__init__()

        # Create convolutional layers
        layers = [ConvUnit3D(in_channels, out_channels, kernel_size)]
        if use_double:
            layers.append(ConvUnit3D(out_channels, out_channels, kernel_size))

        # Instance attributes
        self.out_channels = out_channels
        self.conv = nn.Sequential(*layers)
        self.se = SEBlock3D(out_channels) if use_se else nn.Identity()
        self.pool = nn.Conv3d(
            out_channels, out_channels, kernel_size=2, stride=2, bias=False
        )

    def forward(self, x):
        x = self.se(self.conv(x))
        if min(x.shape[2:]) > 2:
            x = self.pool(x)
        return x


class CenterWeightedPool3D(nn.Module):
    """
    Global pooling with a Gaussian weight centered on the patch, so voxels
    near the center (e.g. the candidate merge/split point) contribute more
    than voxels near the border. Weights are per-spatial-position only
    (shared across channels) and normalize to sum to 1.
    """

    def __init__(self, sigma=0.4):
        """
        Parameters
        ----------
        sigma : float, optional
            Gaussian std, as a fraction of the half-extent along each axis
            (coords run -1..1, so 0.4 means weight falls to ~1/e at 40% of
            the way to the edge). Smaller = tighter focus on center.
            Default is 0.4.
        """
        super().__init__()
        self._dist_cache = {}
        init = torch.log(torch.tensor(float(sigma)))
        self.log_sigma = nn.Parameter(init)

    def _dist_sq(self, shape, device, dtype):
        key = (shape, device, dtype)
        if key not in self._dist_cache:
            coords = [
                torch.linspace(-1, 1, s, device=device, dtype=dtype)
                for s in shape
            ]
            grid = torch.stack(torch.meshgrid(*coords, indexing="ij"), dim=0)
            self._dist_cache[key] = (grid**2).sum(0)
        return self._dist_cache[key]

    def forward(self, x):
        b, c, d, h, w = x.shape
        dist_sq = self._dist_sq((d, h, w), x.device, x.dtype)
        sigma = self.log_sigma.exp().clamp(min=1e-3)
        weights = torch.exp(-dist_sq / (2 * sigma**2))
        weights = (weights / weights.sum()).view(1, 1, d, h, w)
        return (x * weights).sum(dim=(2, 3, 4))


# --- Vision Transformers ---
class ViT3D(nn.Module):

    def __init__(
        self,
        input_shape,
        patch_size=16,
        embed_dim=256,
        depth=6,
        num_heads=4,
        mlp_ratio=4.0,
        dropout=0.1,
        output_dim=1,
        stem_channels=64,
        stem_depth=4,
    ):
        super().__init__()
        in_channels, d, h, w = input_shape
        assert (
            d % patch_size == 0 and h % patch_size == 0 and w % patch_size == 0
        ), f"input_shape spatial dims must be divisible by patch_size={patch_size}"

        # Save model config
        self.config = {
            "input_shape": tuple(input_shape),
            "patch_size": patch_size,
            "embed_dim": embed_dim,
            "depth": depth,
            "num_heads": num_heads,
            "mlp_ratio": mlp_ratio,
            "dropout": dropout,
            "output_dim": output_dim,
            "stem_channels": stem_channels,
            "stem_depth": stem_depth,
        }

        self.embed_dim = embed_dim
        self.num_d = d // patch_size
        self.num_h = h // patch_size
        self.num_w = w // patch_size

        # Patch embedding with conv stem
        self.patch_embed = PatchEmbed3D(
            in_channels,
            embed_dim,
            patch_size,
            stem_channels=stem_channels,
            stem_depth=stem_depth,
        )

        # CLS token with its own learned positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_pos = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Fixed sinusoidal positional embeddings in spherical coordinates
        patch_pe = self._build_spherical_pe(
            self.num_d, self.num_h, self.num_w, embed_dim
        )
        self.register_buffer("patch_pos_embed", patch_pe, persistent=False)
        self.pos_drop = nn.Dropout(dropout)

        # Transformer
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

        # Center-weighted pooling over spatial patch tokens
        self.center_pool = CenterWeightedPool3D()

        # Head receives CLS token, center-pooled patch tokens, and max-pooled patch tokens
        self.head = FeedForwardNet(3 * embed_dim, output_dim, 3)

        self.apply(self.init_weights)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.cls_pos, std=0.02)

    def save(self, path):
        torch.save(
            {"config": self.config, "state_dict": self.state_dict()}, path
        )

    @classmethod
    def load(cls, path, map_location=None):
        ckpt = torch.load(path, map_location=map_location)
        model = cls(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        return model

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv3d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def _build_spherical_pe(num_d, num_h, num_w, embed_dim):
        """
        Fixed sinusoidal positional encoding in spherical coordinates centered
        on the patch grid. Each of r, theta, phi is independently encoded with
        the same frequency bank and the results are summed.
        Returns (1, num_d * num_h * num_w, embed_dim).
        """
        d_c = torch.arange(num_d, dtype=torch.float32) - (num_d - 1) / 2
        h_c = torch.arange(num_h, dtype=torch.float32) - (num_h - 1) / 2
        w_c = torch.arange(num_w, dtype=torch.float32) - (num_w - 1) / 2
        dg, hg, wg = torch.meshgrid(d_c, h_c, w_c, indexing="ij")

        r = (dg**2 + hg**2 + wg**2).sqrt().clamp(min=1e-6)
        theta = (dg / r).clamp(-1 + 1e-6, 1 - 1e-6).acos()  # [0, pi]
        phi = wg.atan2(hg) + torch.pi  # [0, 2*pi]

        # Scale r to [0, 2*pi] so coordinates share the same frequency base
        r_scaled = r / r.amax() * (2 * torch.pi)

        half = embed_dim // 2
        denom = 10000.0 ** (
            2 * torch.arange(half, dtype=torch.float32) / embed_dim
        )

        def encode(v):
            args = v.reshape(-1, 1) / denom
            return torch.cat([args.sin(), args.cos()], dim=1)  # (N, embed_dim)

        pe = encode(r_scaled) + encode(theta) + encode(phi)
        return pe.unsqueeze(0)  # (1, num_patches, embed_dim)

    def forward(self, x):
        b = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_drop(
            x + torch.cat([self.cls_pos, self.patch_pos_embed], dim=1)
        )
        x = self.norm(self.encoder(x))
        patches = x[:, 1:].transpose(1, 2).view(
            b, self.embed_dim, self.num_d, self.num_h, self.num_w
        )
        max_pool = patches.amax(dim=(2, 3, 4))
        feat = torch.cat([x[:, 0], self.center_pool(patches), max_pool], dim=1)
        return self.head(feat)


class PatchEmbed3D(nn.Module):
    """
    Optional conv stem for local feature extraction followed by a strided
    conv that maps spatial patches to token embeddings.
    """

    def __init__(
        self,
        in_channels,
        embed_dim,
        patch_size=4,
        stem_channels=32,
        stem_depth=2,
    ):
        super().__init__()
        layers, c = [], in_channels
        for _ in range(stem_depth):
            layers.append(ConvUnit3D(c, stem_channels, kernel_size=3))
            c = stem_channels
        self.stem = nn.Sequential(*layers) if layers else nn.Identity()
        self.proj = nn.Conv3d(
            c, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        return self.proj(self.stem(x)).flatten(2).transpose(1, 2)
