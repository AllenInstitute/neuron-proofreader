"""
Created on Fri Aug 15 12:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Single-task proofreading models: a shared MultimodalEncoder with a merge
or split head.
"""

from torch_geometric import nn as nn_geometric

import torch
import torch.nn as nn

from neuron_proofreader.models.encoders import MultimodalEncoder
from neuron_proofreader.utils.ml_util import FeedForwardNet

# GATv2Conv layers are lazy, so checkpoints may hold uninitialized parameters
torch.serialization.add_safe_globals(
    [torch.nn.parameter.UninitializedParameter]
)


class ProofreadingModel(nn.Module):
    """
    Base class for the single-task proofreading models: a MultimodalEncoder
    plus the config, fp16 and checkpoint handling they share. Subclasses add
    a head and ``forward``.

    Parameters
    ----------
    patch_shape : Tuple[int]
        Image patch shape (C, D, H, W).
    **encoder_kwargs
        Forwarded to MultimodalEncoder.
    """

    def __init__(self, patch_shape, **encoder_kwargs):
        """Instantiates a ProofreadingModel object."""
        super().__init__()
        self.config = {"patch_shape": tuple(patch_shape), **encoder_kwargs}
        self.encoder = MultimodalEncoder(patch_shape, **encoder_kwargs)

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
        Reconstructs a model from a checkpoint written by ``save``.

        Parameters
        ----------
        path : str
            Path to the checkpoint.
        map_location : str or torch.device, optional
            Passed through to torch.load. Default is None.

        Returns
        -------
        ProofreadingModel
        """
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
        model = cls(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        return model


class MergeModel(ProofreadingModel):
    """
    Merge detector: a MultimodalEncoder embedding of the candidate site
    classified by a FeedForwardNet. Keyword arguments other than
    ``dropout`` are those of MultimodalEncoder.

    Parameters
    ----------
    patch_shape : Tuple[int]
        Image patch shape (C, D, H, W).
    dropout : float, optional
        Dropout in the vision projection and before the head. Default is
        0.1.
    """

    def __init__(self, patch_shape, dropout=0.1, **encoder_kwargs):
        """Instantiates a MergeModel object."""
        super().__init__(patch_shape, dropout=dropout, **encoder_kwargs)
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


# --- Split head
def proposal_length_feature(tree_samples, device, dtype=torch.float32):
    """
    Log-scaled proposal length, one scalar per sample.

    Parameters
    ----------
    tree_samples : List[TreeSample]
        Samples produced by ProposalTreeFeatureExtractor.

    Returns
    -------
    torch.Tensor
        Shape (n_samples, 1), values log1p(length_in_microns).
    """
    lengths = torch.tensor(
        [s.proposal_length for s in tree_samples], device=device, dtype=dtype
    )
    return torch.log1p(lengths).unsqueeze(1)


RELATIONS = [
    ("branch", "to", "branch"),
    ("proposal", "to", "proposal"),
    ("branch", "to", "proposal"),
    ("proposal", "to", "branch"),
]


def _build_hetero_gat(out_dim, heads):
    """
    Builds a HeteroConv with GATv2Conv for every proposal/branch relation.
    Same-type relations use a single input-dim spec; cross-type relations use
    a pair so GATv2Conv can handle bipartite input.
    """
    gat_dict = {}
    for rel in RELATIONS:
        src, _, dst = rel
        if src == dst:
            conv = nn_geometric.GATv2Conv(
                -1, out_dim, dropout=0.1, heads=heads
            )
        else:
            conv = nn_geometric.GATv2Conv(
                (-1, -1),
                out_dim,
                add_self_loops=False,
                dropout=0.1,
                heads=heads,
            )
        gat_dict[rel] = conv
    return nn_geometric.HeteroConv(gat_dict)


class SplitHead(nn.Module):
    """
    Classifies proposals with GAT message passing over proposal and branch
    nodes.

    Parameters
    ----------
    in_dim : int
        Proposal embedding dimension.
    hidden_dim : int, optional
        Node embedding dimension entering the GAT. Default is 96.
    heads : int, optional
        GAT attention heads. Default is 4.
    n_layers : int, optional
        Message passing layers. Default is 3.
    """

    def __init__(self, in_dim, hidden_dim=96, heads=4, n_layers=3):
        """Instantiates a SplitHead object."""
        super().__init__()
        self.fusion = FeedForwardNet(in_dim, hidden_dim, 2)
        self.branch_relay = nn.Parameter(torch.zeros(hidden_dim))
        nn.init.normal_(self.branch_relay, std=0.02)
        self.gats = nn.ModuleList([
            _build_hetero_gat(hidden_dim * heads ** i, heads)
            for i in range(n_layers)
        ])
        self.output = nn.Linear(hidden_dim * heads ** n_layers, 1)

    def forward(self, z, edge_index_dict, n_branches):
        """
        Parameters
        ----------
        z : torch.Tensor
            Proposal embeddings, shape (N, in_dim).
        edge_index_dict : dict
            Edge indices per (src, "to", dst) relation in RELATIONS.
        n_branches : int
            Number of branch nodes.

        Returns
        -------
        torch.Tensor
            Logits of shape (N, 1).
        """
        x_dict = {
            "proposal": self.fusion(z),
            "branch": self.branch_relay.to(z.dtype).expand(n_branches, -1),
        }
        for gat in self.gats:
            x_dict = gat(x_dict, edge_index_dict)
        return self.output(x_dict["proposal"])
