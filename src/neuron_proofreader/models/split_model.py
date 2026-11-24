"""
Created on Tue Nov 24 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Multimodal split correction model: image patch plus proposal-rooted skeleton
subgraph per proposal, with heterogeneous GAT message passing between
proposals and branches.
"""

import torch
import torch.nn as nn

from neuron_proofreader.models.gnn_models import (
    ProposalTreeEncoder,
    _build_hetero_gat,
    proposal_length_feature,
)
from neuron_proofreader.models.merge_model import _normalize_config, _upgrade_state_dict
from neuron_proofreader.models.vision_models import VisionEncoder
from neuron_proofreader.utils.ml_util import FeedForwardNet

# GATv2Conv layers are lazy, so checkpoints may hold uninitialized parameters
torch.serialization.add_safe_globals([torch.nn.parameter.UninitializedParameter])

# Legacy JointProofreader checkpoints: split-path keys renamed, merge keys dropped
_LEGACY_KEYS = {
    "vision.": "vision.backbone.",
    "split_task_conv.": "vision.task_conv.",
    "split_task_pool.": "vision.task_pool.",
    "split_vision_proj.": "vision.proj.",
    "split_tree_encoder.": "tree_encoder.",
    "branch_relay": "branch_relay",
    "split_fusion.": "fusion.",
    "gat1.": "gat1.",
    "gat2.": "gat2.",
    "gat3.": "gat3.",
    "split_head.": "head.",
}
_LEGACY_CONFIG_KEYS = ("task",)


class SplitModel(nn.Module):
    """
    Split corrector that fuses a VisionEncoder embedding of each proposal's
    image patch with a ProposalTreeEncoder embedding of its skeleton and the
    proposal length, then runs three rounds of heterogeneous GAT message
    passing over proposal and branch nodes before a linear classifier.

    Parameters
    ----------
    vision_input_shape : Tuple[int]
        Image patch shape (C, D, H, W).
    img_embed_dim : int, optional
        Image embedding dimension. Default is 96.
    arborist_latent_dim : int, optional
        Latent dimension of the proposal tree encoder. Default is 64.
    arborist_kwargs : dict, optional
        Forwarded to Arborist inside ProposalTreeEncoder. Default is None.
    gnn_hidden_dim : int, optional
        Node embedding dimension entering the GAT. Default is 96.
    split_heads : int, optional
        GAT attention heads. Default is 4.
    dropout : float, optional
        Dropout in the vision projection. Default is 0.1.
    **vision_backbone_kwargs
        Forwarded to CNN3D (e.g. base_channels, depth, block_type).
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
        super().__init__()
        arborist_kwargs = arborist_kwargs or {}
        self.config = {
            "vision_input_shape": tuple(vision_input_shape),
            "img_embed_dim": img_embed_dim,
            "arborist_latent_dim": arborist_latent_dim,
            "arborist_kwargs": arborist_kwargs,
            "gnn_hidden_dim": gnn_hidden_dim,
            "split_heads": split_heads,
            "dropout": dropout,
            **vision_backbone_kwargs,
        }
        self.vision = VisionEncoder(
            vision_input_shape, img_embed_dim, dropout, **vision_backbone_kwargs
        )
        self.tree_encoder = ProposalTreeEncoder(latent_dim=arborist_latent_dim, **arborist_kwargs)
        self.branch_relay = nn.Parameter(torch.zeros(gnn_hidden_dim))
        nn.init.normal_(self.branch_relay, std=0.02)
        self.fusion = FeedForwardNet(img_embed_dim + arborist_latent_dim + 1, gnn_hidden_dim, 2)
        self.gat1 = _build_hetero_gat(gnn_hidden_dim, split_heads)
        self.gat2 = _build_hetero_gat(gnn_hidden_dim * split_heads, split_heads)
        self.gat3 = _build_hetero_gat(gnn_hidden_dim * split_heads ** 2, split_heads)
        self.head = nn.Linear(gnn_hidden_dim * split_heads ** 3, 1)

    def forward(self, x):
        """
        Parameters
        ----------
        x : dict
            Batch with "img" (N, C, D, H, W), "tree_samples"
            (List[TreeSample], one per proposal), "edge_index_dict" and
            "n_branches".

        Returns
        -------
        torch.Tensor
            Logits of shape (N, 1), one per proposal.
        """
        z_img = self.vision(x["img"])
        z_tree = self.tree_encoder(x["tree_samples"]).to(z_img.dtype)
        length = proposal_length_feature(x["tree_samples"], z_img.device, z_img.dtype)
        x_dict = {
            "proposal": self.fusion(torch.cat([z_img, z_tree, length], dim=1)),
            "branch": self.branch_relay.to(z_img.dtype).unsqueeze(0).expand(x["n_branches"], -1),
        }
        x_dict = self.gat1(x_dict, x["edge_index_dict"])
        x_dict = self.gat2(x_dict, x["edge_index_dict"])
        x_dict = self.gat3(x_dict, x["edge_index_dict"])
        return self.head(x_dict["proposal"])

    def half_vision(self):
        """Runs the image path in fp16, leaving the tree encoder, GAT and head in fp32."""
        self.vision.half()
        return self

    def save(self, path):
        torch.save({"config": self.config, "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path, map_location=None):
        """
        Reconstructs a SplitModel from a checkpoint written by ``save``. Also
        reads the split path out of legacy JointProofreader checkpoints.

        Parameters
        ----------
        path : str
            Path to the checkpoint.
        map_location : str or torch.device, optional
            Passed through to torch.load. Default is None.

        Returns
        -------
        SplitModel
        """
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
        model = cls(**_normalize_config(ckpt["config"], _LEGACY_CONFIG_KEYS))
        model.load_state_dict(_upgrade_state_dict(ckpt["state_dict"], _LEGACY_KEYS, legacy_marker="split_head."))
        return model
