"""
Created on Tue Nov 24 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Multimodal merge detection model: image patch plus skeleton subgraph of a
candidate site.
"""

import json
import os

import torch
import torch.nn as nn

from arborist.models.arborist import Arborist
from neuron_proofreader.models.vision_models import VisionEncoder
from neuron_proofreader.utils.ml_util import FeedForwardNet

# Legacy JointProofreader checkpoints: merge-path keys renamed, split keys dropped
_LEGACY_KEYS = {
    "vision.": "vision.backbone.",
    "merge_task_conv.": "vision.task_conv.",
    "merge_task_pool.": "vision.task_pool.",
    "merge_vision_proj.": "vision.proj.",
    "merge_head.": "head.",
    "arborist.": "arborist.",
}
_LEGACY_CONFIG_KEYS = ("gnn_hidden_dim", "split_heads", "task")


class MergeModel(nn.Module):
    """
    Merge detector that fuses a VisionEncoder embedding of the site's image
    patch with Arborist's encoding of its skeleton subgraph (CLS token plus
    the mean of the root-adjacent curve embeddings) and classifies with a
    FeedForwardNet.

    Parameters
    ----------
    vision_input_shape : Tuple[int]
        Image patch shape (C, D, H, W).
    img_embed_dim : int, optional
        Image embedding dimension. Default is 96.
    arborist_latent_dim : int, optional
        Arborist latent dimension. Default is 64.
    arborist_kwargs : dict, optional
        Forwarded to Arborist. Default is None.
    dropout : float, optional
        Dropout in the vision projection and before the head. Default is 0.1.
    **vision_backbone_kwargs
        Forwarded to CNN3D (e.g. base_channels, depth, block_type).
    """

    def __init__(
        self,
        vision_input_shape,
        img_embed_dim=96,
        arborist_latent_dim=64,
        arborist_kwargs=None,
        dropout=0.1,
        **vision_backbone_kwargs,
    ):
        super().__init__()
        arborist_kwargs = arborist_kwargs or {}
        self.config = {
            "vision_input_shape": tuple(vision_input_shape),
            "img_embed_dim": img_embed_dim,
            "arborist_latent_dim": arborist_latent_dim,
            "arborist_kwargs": arborist_kwargs,
            "dropout": dropout,
            **vision_backbone_kwargs,
        }
        self.vision = VisionEncoder(
            vision_input_shape, img_embed_dim, dropout, **vision_backbone_kwargs
        )
        self.arborist = Arborist(latent_dim=arborist_latent_dim, **arborist_kwargs)
        self.drop = nn.Dropout(dropout)
        self.head = FeedForwardNet(img_embed_dim + 2 * arborist_latent_dim, 1, 3)

    def forward(self, x):
        """
        Parameters
        ----------
        x : dict
            Batch with "img" (B, C, D, H, W) and "tree_sample"
            (List[TreeSample]).

        Returns
        -------
        torch.Tensor
            Logits of shape (B, 1).
        """
        z_img = self.vision(x["img"])
        z_tree = self._encode_trees(x["tree_sample"], z_img.device)
        return self.head(self.drop(torch.cat([z_img, z_tree], dim=1)))

    @torch._dynamo.disable
    def _encode_trees(self, tree_samples, device):
        with torch.amp.autocast("cuda", enabled=False):
            z_trees = []
            for s in tree_samples:
                z_tree, z_curves = self.arborist.encode(s)
                idx = list(s.root_curve_indices or range(len(z_curves)))
                z_trees.append(torch.cat([z_tree, z_curves[idx].mean(dim=0)]))
        return torch.stack(z_trees).to(device)

    def half_vision(self):
        """
        Runs the image path in fp16, leaving Arborist and the head in fp32.
        The backbone is memory-bandwidth bound, so halving its activations
        is ~1.8x faster on an L4; Arborist stays fp32 because it builds fp32
        tensors from numpy inside the forward pass.
        """
        self.vision.half()
        return self

    def save(self, path):
        torch.save({"config": self.config, "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path, map_location=None, config=None):
        """
        Reconstructs a MergeModel from a checkpoint written by ``save``, or
        from a raw state_dict plus a config dict or model_config.json path.
        Also reads the merge path out of legacy JointProofreader checkpoints.

        Parameters
        ----------
        path : str
            Path to the checkpoint.
        map_location : str or torch.device, optional
            Passed through to torch.load. Default is None.
        config : dict or str, optional
            Constructor kwargs, or a path to them, for a raw state_dict.
            Default is None.

        Returns
        -------
        MergeModel
        """
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
        if "config" in ckpt and "state_dict" in ckpt:
            config, state_dict = ckpt["config"], ckpt["state_dict"]
        else:
            if config is None:
                config = os.path.join(os.path.dirname(path), "model_config.json")
            if isinstance(config, str):
                with open(config) as f:
                    config = json.load(f)
            state_dict = ckpt
        model = cls(**_normalize_config(config, _LEGACY_CONFIG_KEYS))
        model.load_state_dict(_upgrade_state_dict(state_dict, _LEGACY_KEYS, legacy_marker="merge_head."))
        return model


def _normalize_config(config, drop_keys=()):
    """
    Restores the tuple-valued entries that JSON turns into lists and drops
    keys from other models' configs. Conv3d and the pooling stages index the
    tuples positionally, so a list would silently build a different backbone.
    """
    config = {k: v for k, v in config.items() if k not in drop_keys}
    for key in ("vision_input_shape", "pool_stage_idxs"):
        if config.get(key) is not None:
            config[key] = tuple(config[key])
    return config


def _upgrade_state_dict(state_dict, legacy_keys, legacy_marker):
    """
    Renames the keys of a legacy JointProofreader state_dict onto the
    single-task model (prefixes in ``legacy_keys``; others are dropped).
    A state_dict without ``legacy_marker`` is returned unchanged.
    """
    if not any(k.startswith(legacy_marker) for k in state_dict):
        return state_dict
    upgraded = {}
    for key, value in state_dict.items():
        for old, new in legacy_keys.items():
            if key.startswith(old):
                upgraded[new + key[len(old):]] = value
                break
    return upgraded
