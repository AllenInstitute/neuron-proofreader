"""
Created on Fri Sep 25 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Heterogeneous GAT classification head of the split model.

"""

from torch_geometric import nn as nn_geometric

import torch
import torch.nn as nn

from neuron_proofreader.utils.ml_util import FeedForwardNet


def proposal_length_feature(tree_samples, device, dtype=torch.float32):
    """
    Log-scaled proposal length, one scalar per sample. Length is a property
    of the candidate edge that neither the rooted-subgraph curves nor the
    resized image patch expose, so it is fed to models explicitly.

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
