"""
Created on Tue Nov 24 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Split correction model: MultimodalEncoder with a heterogeneous GAT head
over proposals and branches.
"""

import torch
import torch.nn as nn

from neuron_proofreader.models.encoders import MultimodalEncoder
from neuron_proofreader.models.gnn_models import (
    SplitHead,
    proposal_length_feature,
)

# GATv2Conv layers are lazy, so checkpoints may hold uninitialized parameters
torch.serialization.add_safe_globals(
    [torch.nn.parameter.UninitializedParameter]
)


class SplitModel(nn.Module):
    """
    Split corrector: MultimodalEncoder embeddings of each proposal, plus the
    proposal length, classified by a SplitHead. Keyword arguments other than
    ``gnn_hidden_dim``, ``gnn_n_layers`` and ``split_heads`` are those of
    MultimodalEncoder.

    Parameters
    ----------
    vision_input_shape : Tuple[int]
        Image patch shape (C, D, H, W).
    gnn_hidden_dim : int, optional
        Node embedding dimension entering the GAT. Default is 96.
    gnn_n_layers : int, optional
        Message passing layers. Default is 3.
    split_heads : int, optional
        GAT attention heads. Default is 4.
    """

    def __init__(
        self,
        vision_input_shape,
        gnn_hidden_dim=96,
        gnn_n_layers=3,
        split_heads=4,
        **encoder_kwargs,
    ):
        """Instantiates a SplitModel object."""
        super().__init__()
        self.config = {
            "vision_input_shape": tuple(vision_input_shape),
            "gnn_hidden_dim": gnn_hidden_dim,
            "gnn_n_layers": gnn_n_layers,
            "split_heads": split_heads,
            **encoder_kwargs,
        }
        self.encoder = MultimodalEncoder(vision_input_shape, **encoder_kwargs)
        self.head = SplitHead(
            self.encoder.embed_dim + 1,
            hidden_dim=gnn_hidden_dim,
            heads=split_heads,
            n_layers=gnn_n_layers,
        )

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
        z = torch.cat([z, length], dim=1)
        return self.head(z, x["edge_index_dict"], x["n_branches"])

    def half_vision(self):
        """Runs the image path in fp16; see MultimodalEncoder.half_vision."""
        self.encoder.half_vision()
        return self

    def save(self, path):
        """Saves the config and weights to a checkpoint."""
        state = {"config": self.config, "state_dict": self.state_dict()}
        torch.save(state, path)

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
        model = cls(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        return model
