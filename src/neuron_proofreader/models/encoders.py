"""
Created on Tue Nov 24 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Encoders shared by the proofreading models: the image patch encoder and
its multimodal combination with arborist's ArboristEncoder. The merge and
split models differ only in the classification head on top of
MultimodalEncoder. Each encoder exposes ``embed_dim`` and a ``config`` of
its constructor arguments so a checkpoint can rebuild it.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from arborist.models.arborist import ArboristEncoder
from neuron_proofreader.models.vision_models import CNN3D, CenterWeightedPool3D


class VisionEncoder(nn.Module):
    """
    Image encoder for the single-task proofreading models: a CNN3D backbone
    without an output head, a task-specific 1x1 conv and center-weighted pool
    on its final stage, and a projection of the concatenated pooled features
    to the image embedding.

    Parameters
    ----------
    input_shape : Tuple[int]
        Image patch shape (C, D, H, W).
    embed_dim : int, optional
        Image embedding dimension. Default is 96.
    dropout : float, optional
        Dropout after the projection. Default is 0.1.
    **cnn_kwargs
        Forwarded to CNN3D (e.g. base_channels, depth, block_type,
        max_channels, pool_stage_idxs).
    """

    def __init__(self, input_shape, embed_dim=96, dropout=0.1, **cnn_kwargs):
        """Instantiates a VisionEncoder object."""
        super().__init__()
        self.config = {
            "input_shape": tuple(input_shape),
            "embed_dim": embed_dim,
            "dropout": dropout,
            **cnn_kwargs,
        }
        self.embed_dim = embed_dim
        self.backbone = CNN3D(input_shape, use_output_head=False, **cnn_kwargs)

        final_idx = self.backbone.pool_stage_idxs[-1]
        final_channels = self.backbone.encode.blocks[final_idx].out_channels
        sigma = self.backbone.config["center_pool_sigma"]
        learnable_sigma = self.backbone.config["learnable_center_sigma"]
        self.task_conv = nn.Conv3d(final_channels, final_channels, 1)
        self.task_pool = CenterWeightedPool3D(sigma, learnable_sigma)
        self.proj = nn.Sequential(
            nn.Linear(
                self.backbone.feature_dim + 2 * final_channels, embed_dim
            ),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, img):
        """
        Parameters
        ----------
        img : torch.Tensor
            Shape (B, C, D, H, W).

        Returns
        -------
        torch.Tensor
            Shape (B, embed_dim), float32 regardless of the backbone dtype so
            the tree encoders and heads see fp32 when the image path runs in
            fp16 (see the models' "half_vision").
        """
        # Run the backbone once
        img = img.to(next(self.backbone.parameters()).dtype)
        stages = self.backbone.encode(img)

        # Multi-scale pooling of the backbone stages (center-weighted + max)
        shared = []
        stage_pools = zip(
            self.backbone.pool_stage_idxs, self.backbone.center_pools
        )
        for idx, pool in stage_pools:
            shared.append(_center_and_max(stages[idx], pool))
        shared = self.backbone.drop(torch.cat(shared, dim=1))

        # Task-specific conv and pooling on the final stage
        task = self.task_conv(stages[self.backbone.pool_stage_idxs[-1]])
        task = _center_and_max(task, self.task_pool)

        return self.proj(torch.cat([shared, task], dim=1)).float()


class MultimodalEncoder(nn.Module):
    """
    Embeds a site from its image patch and skeleton subgraph by
    concatenating the VisionEncoder and ArboristEncoder embeddings.

    Parameters
    ----------
    patch_shape : Tuple[int]
        Image patch shape (C, D, H, W).
    img_embed_dim : int, optional
        Image embedding dimension. Default is 96.
    arborist_latent_dim : int, optional
        Arborist latent dimension. Default is 64.
    arborist_kwargs : dict, optional
        Forwarded to Arborist. Default is None.
    dropout : float, optional
        Dropout in the vision projection. Default is 0.1.
    **vision_backbone_kwargs
        Forwarded to CNN3D (e.g. base_channels, depth, block_type).
    """

    def __init__(
        self,
        patch_shape,
        img_embed_dim=96,
        arborist_latent_dim=64,
        arborist_kwargs=None,
        dropout=0.1,
        **vision_backbone_kwargs,
    ):
        """Instantiates a MultimodalEncoder object."""
        super().__init__()
        self.vision = VisionEncoder(
            patch_shape,
            img_embed_dim,
            dropout,
            **vision_backbone_kwargs,
        )
        self.arborist = ArboristEncoder(
            curve_latent_dim=arborist_latent_dim, **(arborist_kwargs or {})
        )
        self.embed_dim = self.vision.embed_dim + self.arborist.embed_dim

    def forward(self, img, tree_samples):
        """
        Parameters
        ----------
        img : torch.Tensor
            Shape (B, C, D, H, W).
        tree_samples : List[TreeSample]
            Rooted subgraphs, one per site.

        Returns
        -------
        torch.Tensor
            Shape (B, embed_dim), on the image's device.
        """
        z_img = self.vision(img)
        z_tree = self.arborist(tree_samples).to(z_img.device)
        return torch.cat([z_img, z_tree], dim=1)

    def half_vision(self):
        """
        Runs the image path in fp16, leaving Arborist in fp32. The backbone
        is memory-bandwidth bound, so halving its activations is ~1.8x faster
        on an L4; Arborist builds fp32 tensors from numpy inside its forward.
        """
        self.vision.half()
        return self


def _center_and_max(x, center_pool):
    """Concatenates center-weighted and max pooling of a 3D feature map."""
    return torch.cat(
        [center_pool(x), F.adaptive_max_pool3d(x, 1).flatten(1)], dim=1
    )
