"""
Created on Tue Nov 24 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Split correction model: MultimodalEncoder with a heterogeneous GAT head
over proposals and branches.
"""

import torch
import torch.nn as nn

from arborist.models.arborist import upgrade_state_dict
from neuron_proofreader.models.encoders import MultimodalEncoder
from neuron_proofreader.models.gnn_models import SplitHead, proposal_length_feature
from neuron_proofreader.models.merge_model import normalize_config

# GATv2Conv layers are lazy, so checkpoints may hold uninitialized parameters
torch.serialization.add_safe_globals([torch.nn.parameter.UninitializedParameter])


class SplitModel(nn.Module):
    """
    Split corrector: MultimodalEncoder embeddings of each proposal, plus the
    proposal length, classified by a SplitHead. Keyword arguments other than
    ``gnn_hidden_dim`` and ``split_heads`` are those of MultimodalEncoder.

    Parameters
    ----------
    vision_input_shape : Tuple[int]
        Image patch shape (C, D, H, W).
    gnn_hidden_dim : int, optional
        Node embedding dimension entering the GAT. Default is 96.
    split_heads : int, optional
        GAT attention heads. Default is 4.
    """

    def __init__(self, vision_input_shape, gnn_hidden_dim=96, split_heads=4, **encoder_kwargs):
        super().__init__()
        self.config = {
            "vision_input_shape": tuple(vision_input_shape),
            "gnn_hidden_dim": gnn_hidden_dim,
            "split_heads": split_heads,
            **encoder_kwargs,
        }
        self.encoder = MultimodalEncoder(vision_input_shape, **encoder_kwargs)
        self.head = SplitHead(self.encoder.embed_dim + 1, gnn_hidden_dim, split_heads)

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
        z = self.encoder(x["img"], x["tree_samples"])
        length = proposal_length_feature(x["tree_samples"], z.device, z.dtype)
        return self.head(torch.cat([z, length], dim=1), x["edge_index_dict"], x["n_branches"])

    def half_vision(self):
        """Runs the image path in fp16; see MultimodalEncoder.half_vision."""
        self.encoder.half_vision()
        return self

    def save(self, path):
        torch.save({"config": self.config, "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path, map_location=None):
        """
        Reconstructs a SplitModel from a checkpoint written by ``save``.

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
        model = cls(**normalize_config(ckpt["config"]))
        model.load_state_dict(upgrade_state_dict(ckpt["state_dict"]))
        return model
