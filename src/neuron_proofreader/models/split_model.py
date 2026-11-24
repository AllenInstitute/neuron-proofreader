"""
Created on Tue Nov 24 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Split correction model: MultimodalEncoder with a heterogeneous GAT head
over proposals and branches.
"""

import torch

from neuron_proofreader.models.gnn_models import (
    SplitHead,
    proposal_length_feature,
)
from neuron_proofreader.models.proofreading_models import ProofreadingModel

# GATv2Conv layers are lazy, so checkpoints may hold uninitialized parameters
torch.serialization.add_safe_globals(
    [torch.nn.parameter.UninitializedParameter]
)


class SplitModel(ProofreadingModel):
    """
    Split corrector: MultimodalEncoder embeddings of each proposal, plus the
    proposal length, classified by a SplitHead. Keyword arguments other than
    ``gnn_hidden_dim``, ``gnn_n_layers`` and ``split_heads`` are those of
    MultimodalEncoder.

    Parameters
    ----------
    patch_shape : Tuple[int]
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
        patch_shape,
        gnn_hidden_dim=96,
        gnn_n_layers=3,
        split_heads=4,
        **encoder_kwargs,
    ):
        """Instantiates a SplitModel object."""
        super().__init__(patch_shape, **encoder_kwargs)
        self.config.update(
            gnn_hidden_dim=gnn_hidden_dim,
            gnn_n_layers=gnn_n_layers,
            split_heads=split_heads,
        )
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
