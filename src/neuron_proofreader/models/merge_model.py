"""
Created on Tue Nov 24 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Merge detection model: MultimodalEncoder with a FeedForwardNet head.
"""

import json
import os

import torch
import torch.nn as nn

from neuron_proofreader.models.encoders import MultimodalEncoder
from neuron_proofreader.utils.ml_util import FeedForwardNet

# Legacy JointProofreader checkpoints: merge-path keys renamed, split keys dropped
_LEGACY_KEYS = {
    "vision.": "encoder.vision.backbone.",
    "merge_task_conv.": "encoder.vision.task_conv.",
    "merge_task_pool.": "encoder.vision.task_pool.",
    "merge_vision_proj.": "encoder.vision.proj.",
    "arborist.": "encoder.arborist.arborist.",
    "merge_head.": "head.",
}
_LEGACY_CONFIG_KEYS = ("gnn_hidden_dim", "split_heads", "task")


class MergeModel(nn.Module):
    """
    Merge detector: a MultimodalEncoder embedding of the candidate site
    classified by a FeedForwardNet. Constructor arguments are those of
    MultimodalEncoder.
    """

    def __init__(self, vision_input_shape, dropout=0.1, **encoder_kwargs):
        super().__init__()
        self.config = {
            "vision_input_shape": tuple(vision_input_shape),
            "dropout": dropout,
            **encoder_kwargs,
        }
        self.encoder = MultimodalEncoder(vision_input_shape, dropout=dropout, **encoder_kwargs)
        self.drop = nn.Dropout(dropout)
        self.head = FeedForwardNet(self.encoder.embed_dim, 1, 3)

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
        return self.head(self.drop(self.encoder(x["img"], x["tree_sample"])))

    def half_vision(self):
        """Runs the image path in fp16; see MultimodalEncoder.half_vision."""
        self.encoder.half_vision()
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
        model = cls(**normalize_config(config, _LEGACY_CONFIG_KEYS))
        model.load_state_dict(_upgrade_state_dict(state_dict))
        return model


def normalize_config(config, drop_keys=()):
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


def _upgrade_state_dict(state_dict):
    """
    Renames the keys of a legacy JointProofreader state_dict onto MergeModel
    and drops the split path. Other state_dicts are returned unchanged.
    """
    if not any(k.startswith("merge_head.") for k in state_dict):
        return state_dict
    upgraded = {}
    for key, value in state_dict.items():
        for old, new in _LEGACY_KEYS.items():
            if key.startswith(old):
                upgraded[new + key[len(old):]] = value
                break
    return upgraded
