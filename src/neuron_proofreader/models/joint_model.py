"""
Created on Fri Sep 25 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Joint model for simultaneous split and merge correction.

"""

import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from arborist.models.arborist import Arborist
from neuron_proofreader.models.new_gnn_models import (
    ProposalTreeEncoder,
    _build_hetero_gat,
    proposal_length_feature,
)
from neuron_proofreader.models.new_vision_models import CNN3D, CenterWeightedPool3D
from neuron_proofreader.models.proofreading_models import ArboristVisionMergeDetector
from neuron_proofreader.utils.ml_util import FeedForwardNet

# The GAT layers are lazy, so a checkpoint written before the first forward
# pass holds UninitializedParameters that weights_only loading rejects.
torch.serialization.add_safe_globals([torch.nn.parameter.UninitializedParameter])


class JointProofreader(nn.Module):
    """
    Joint model for split and merge proofreading.

    Architecture
    ------------
    Shared (low-level)
        CNN3D backbone without an output head — produces raw pooled features
        consumed by both task projections. The curve encoder inside Arborist
        is also shared between the merge and split tree encoders so that
        low-level skeletal geometry is learned once.

    Merge head
        merge_vision_proj: projects shared backbone features → img_embed_dim.
        arborist: encodes the full proposal subgraph → z_graph (graph-level).
        merge_head MLP: classifies from cat(z_img, z_tree).

    Split head
        split_vision_proj: projects shared backbone features → img_embed_dim.
        split_tree_encoder (ProposalTreeEncoder): encodes root curves →
            z_tree (curve-level, distinct pooling from merge).
        branch_relay: single learned vector broadcast to all branch nodes.
        split_fusion MLP: fuses image + tree + proposal length → gnn_hidden_dim.
        gat1, gat2, gat3: three rounds of heterogeneous GAT message passing.
        split_head: linear classifier on proposal node embeddings.
    """

    def __init__(
        self,
        vision_input_shape,
        img_embed_dim=96,
        arborist_latent_dim=64,
        arborist_kwargs=None,
        gnn_hidden_dim=96,
        split_heads=4,
        dropout=0.1,
        **vision_backbone_kwargs,
    ):
        """
        Instantiates a JointProofreader object.

        Parameters
        ----------
        vision_input_shape : Tuple[int]
            Shape of one image patch: (C, D, H, W). Both tasks share this
            format — 2 channels (raw image + segmentation mask).
        img_embed_dim : int, optional
            Output dimension of each task's vision projection. Default is 64.
        arborist_latent_dim : int, optional
            Latent dimension for both Arborist encoders (merge graph-level and
            split curve-level). Default is 64.
        arborist_kwargs : dict or None, optional
            Extra keyword arguments forwarded to Arborist.__init__. The same
            kwargs are used for both encoders since they share curve encoder
            weights.
        gnn_hidden_dim : int, optional
            Hidden dimension for split node embeddings entering the GAT.
            Default is 64.
        split_heads : int, optional
            Number of GAT attention heads in the split head. Default is 4.
        dropout : float, optional
            Dropout probability used in vision projections and before the
            merge classification head. Default is 0.1.
        **vision_backbone_kwargs
            Forwarded to CNN3D (e.g. base_channels, depth, block_type).
        """
        super().__init__()

        _arborist_kwargs = arborist_kwargs or {}
        self.config = {
            "vision_input_shape": tuple(vision_input_shape),
            "img_embed_dim": img_embed_dim,
            "arborist_latent_dim": arborist_latent_dim,
            "arborist_kwargs": _arborist_kwargs,
            "gnn_hidden_dim": gnn_hidden_dim,
            "split_heads": split_heads,
            "dropout": dropout,
            **vision_backbone_kwargs,
        }

        # Shared low-level vision backbone — no output head, raw pooled features.
        self.vision = CNN3D(
            vision_input_shape,
            use_output_head=False,
            **vision_backbone_kwargs,
        )

        # Task-specific conv heads on the final encoder stage, followed by
        # task-specific pooling. 1x1 convs mix channels without additional
        # spatial processing (spatial processing is done by the shared encoder).
        final_idx = self.vision.pool_stage_idxs[-1]
        final_channels = self.vision.encode.blocks[final_idx].out_channels
        sigma = self.vision.config.get("center_pool_sigma", 0.4)
        learnable_sigma = self.vision.config.get("learnable_center_sigma", True)

        self.merge_task_conv = nn.Conv3d(final_channels, final_channels, kernel_size=1)
        self.split_task_conv = nn.Conv3d(final_channels, final_channels, kernel_size=1)
        self.merge_task_pool = CenterWeightedPool3D(sigma=sigma, learnable=learnable_sigma)
        self.split_task_pool = CenterWeightedPool3D(sigma=sigma, learnable=learnable_sigma)

        # Total image feature dim: shared pooled features + task-specific pooled features.
        task_feat_dim = final_channels * 2  # center + adaptive max
        total_img_feat_dim = self.vision.feature_dim + task_feat_dim

        # Task-specific projections from combined image features to img_embed_dim.
        self.merge_vision_proj = nn.Sequential(
            nn.Linear(total_img_feat_dim, img_embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.split_vision_proj = nn.Sequential(
            nn.Linear(total_img_feat_dim, img_embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Merge encoder: graph-level Arborist embedding + MLP classifier.
        self.arborist = Arborist(latent_dim=arborist_latent_dim, **_arborist_kwargs)
        self.drop = nn.Dropout(dropout)
        self.merge_head = FeedForwardNet(img_embed_dim + arborist_latent_dim, 1, 3)

        # Split encoder: curve-level ProposalTreeEncoder + HGAT classifier.
        # Shares the low-level curve encoder with self.arborist; the graph
        # transformer is separate so each task can develop its own high-level
        # representation.
        self.split_tree_encoder = ProposalTreeEncoder(
            latent_dim=arborist_latent_dim, **_arborist_kwargs
        )
        self.split_tree_encoder.arborist.curve_encoder = self.arborist.curve_encoder

        self.branch_relay = nn.Parameter(torch.zeros(gnn_hidden_dim))
        nn.init.normal_(self.branch_relay, std=0.02)
        self.split_fusion = FeedForwardNet(
            img_embed_dim + arborist_latent_dim + 1, gnn_hidden_dim, 2
        )
        self.gat1 = _build_hetero_gat(gnn_hidden_dim, split_heads)
        self.gat2 = _build_hetero_gat(gnn_hidden_dim * split_heads, split_heads)
        self.gat3 = _build_hetero_gat(gnn_hidden_dim * split_heads ** 2, split_heads)
        self.split_head = nn.Linear(gnn_hidden_dim * split_heads ** 3, 1)

    def forward(self, x, task):
        """
        Parameters
        ----------
        x : dict
            Batch dictionary. For "merge": must contain "img" and
            "tree_sample". For "split": must contain "img", "tree_samples",
            and "edge_index_dict".
        task : str
            "merge" or "split".

        Returns
        -------
        torch.Tensor
            Logits with shape (N, 1).
        """
        if task == "merge":
            return self._forward_merge(x)
        return self._forward_split(x)

    def _encode_image(self, x_img, task_conv, task_pool, vision_proj):
        """
        Run the shared encoder once, then apply the task-specific 1x1 conv and
        pool on the final stage. Returns the projected image embedding.
        """
        stages = self.vision.encode(x_img)

        # Shared multi-scale pooling (mirrors CNN3D.forward with use_output_head=False).
        shared_feats = []
        for idx, pool in zip(self.vision.pool_stage_idxs, self.vision.center_pools):
            s = stages[idx]
            center = pool(s)
            mx = F.adaptive_max_pool3d(s, 1).flatten(1)
            shared_feats.append(torch.cat([center, mx], dim=1))
        shared = self.vision.drop(torch.cat(shared_feats, dim=1))

        # Task-specific conv + pool on the final encoder stage.
        final = stages[self.vision.pool_stage_idxs[-1]]
        task_out = task_conv(final)
        center = task_pool(task_out)
        mx = F.adaptive_max_pool3d(task_out, 1).flatten(1)
        task = torch.cat([center, mx], dim=1)

        return vision_proj(torch.cat([shared, task], dim=1))

    def _forward_merge(self, x):
        x_img = x["img"]
        if not torch.is_autocast_enabled():
            x_img = x_img.to(next(self.vision.parameters()).dtype)
        z_img = self._encode_image(
            x_img, self.merge_task_conv, self.merge_task_pool, self.merge_vision_proj
        )
        z_tree = self._encode_merge_tree(x["tree_sample"], z_img.device)
        return self.merge_head(self.drop(torch.cat([z_img, z_tree], dim=1)))

    def _forward_split(self, x):
        x_img = x["img"]
        if not torch.is_autocast_enabled():
            x_img = x_img.to(next(self.vision.parameters()).dtype)
        z_img = self._encode_image(
            x_img, self.split_task_conv, self.split_task_pool, self.split_vision_proj
        )

        z_tree = self.split_tree_encoder(x["tree_samples"]).to(z_img.dtype)
        length = proposal_length_feature(x["tree_samples"], z_img.device, z_img.dtype)

        n_branches = x["x_dict"]["branch"].shape[0]
        x_dict = {
            "proposal": self.split_fusion(torch.cat([z_img, z_tree, length], dim=1)),
            "branch": self.branch_relay.to(z_img.dtype).unsqueeze(0).expand(
                n_branches, -1
            ),
        }
        x_dict = self.gat1(x_dict, x["edge_index_dict"])
        x_dict = self.gat2(x_dict, x["edge_index_dict"])
        x_dict = self.gat3(x_dict, x["edge_index_dict"])
        return self.split_head(x_dict["proposal"])

    @torch._dynamo.disable
    def _encode_merge_tree(self, tree_samples, device):
        with torch.amp.autocast("cuda", enabled=False):
            z_trees = [self.arborist.encode(s)[0] for s in tree_samples]
        return torch.stack(z_trees).to(device)

    def save(self, path):
        torch.save({"config": self.config, "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path, map_location=None, config=None):
        """
        Reconstructs a JointProofreader from a checkpoint.

        Supports two formats:
        - Dict with "config" and "state_dict" keys (saved via `save()`).
        - Raw state_dict, which is what JointTrainer writes. In this case
          `config` must be provided as a dict of constructor kwargs, or as a
          path to a model_config.json.

        Parameters
        ----------
        path : str
            Path to a checkpoint written by `save` or a raw state_dict.
        map_location : str or torch.device, optional
            Passed through to `torch.load`.
        config : dict or str, optional
            Constructor kwargs (or path to model_config.json) used when the
            checkpoint is a raw state_dict with no embedded config.

        Returns
        -------
        JointProofreader
            Model with architecture and weights matching the checkpoint.
        """
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
        if "config" in ckpt and "state_dict" in ckpt:
            model = cls(**ckpt["config"])
            model.load_state_dict(ckpt["state_dict"])
            return model

        if config is None:
            config = os.path.join(os.path.dirname(path), "model_config.json")
        if isinstance(config, str):
            with open(config) as f:
                config = json.load(f)
        model = cls(**cls._normalize_config(config))
        model.load_state_dict(ckpt)
        return model

    @staticmethod
    def _normalize_config(config):
        """
        Restores tuple-valued entries that JSON round-tripping turns into
        lists. Conv3d and the pooling stages index these positionally, so a
        list would silently build a different backbone.
        """
        config = dict(config)
        for key in ("vision_input_shape", "pool_stage_idxs"):
            if key in config and config[key] is not None:
                config[key] = tuple(config[key])
        return config

    def as_merge_model(self):
        """
        Returns a view of this model that is callable as `model(x)` for merge
        detection, which is the interface MLMergeProofreader expects.
        """
        return JointTaskView(self, "merge")

    def as_split_model(self):
        """
        Returns a view of this model that is callable as `model(x)` for split
        correction, which is the interface LearnedSplitProofreader expects.
        """
        return JointTaskView(self, "split")

    def drop_split_head(self):
        """
        Deletes the split-only submodules in place, leaving a model that can
        serve merge detection alone.

        The curve encoder is shared with the split tree encoder, so it stays
        on self.arborist; only modules the merge path never touches are
        removed.

        Returns
        -------
        JointProofreader
            This model, for chaining.
        """
        for name in (
            "split_vision_proj", "split_task_conv", "split_task_pool",
            "split_tree_encoder", "branch_relay", "split_fusion",
            "gat1", "gat2", "gat3", "split_head",
        ):
            if hasattr(self, name):
                delattr(self, name)
        return self

    @classmethod
    def load_merge_head(cls, path, map_location=None, config=None):
        """
        Loads a checkpoint and returns a standalone merge-only model, callable
        as `model(x)`.

        This reconstructs the full JointProofreader, drops the split
        submodules, and returns a merge-bound view, so the forward pass is the
        one the checkpoint was trained with. It does not copy the weights into
        an ArboristVisionMergeDetector: that model has no task-specific
        conv/pool stage, and its vision_proj is sized for the shared pooled
        features only, so it cannot represent the merge path.

        Parameters
        ----------
        path : str
            Path to a checkpoint written by `save` or a raw state_dict.
        map_location : str or torch.device, optional
            Passed through to `torch.load`.
        config : dict or str, optional
            Constructor kwargs (or path to model_config.json), needed only for
            a raw state_dict with no embedded config.

        Returns
        -------
        JointTaskView
            Merge-only model; call it as `model(x)`.
        """
        model = cls.load(path, map_location=map_location, config=config)
        return model.drop_split_head().as_merge_model()


class JointTaskView(nn.Module):
    """
    Binds a JointProofreader to one task so that it can be called with a
    single argument. The inference pipelines are written against single-task
    models and call `model(x)`; the joint model needs `model(x, task)`.
    """

    def __init__(self, model, task):
        super().__init__()
        self.model = model
        self.task = task

    def forward(self, x):
        return self.model(x, self.task)
