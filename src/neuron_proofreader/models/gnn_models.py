"""
Created on Fri Sep 25 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Tree encoder and heterogeneous GAT layers used by the split head.

"""

from torch_geometric import nn as nn_geometric

import torch
import torch.nn as nn

from arborist.models.arborist import Arborist
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


class ProposalTreeEncoder(nn.Module):
    """
    Encodes proposal-rooted TreeSamples into fixed-size embeddings using
    Arborist. The two curves incident to the root (one per side of the
    proposal) are mean-pooled after the GraphTransformer has contextualized
    them against the rest of the subgraph.

    Parameters
    ----------
    latent_dim : int, optional
        Arborist latent dimension and output embedding size. Default is 64.
    **arborist_kwargs
        Forwarded to the Arborist constructor.
    """

    def __init__(
        self,
        latent_dim=64,
        **arborist_kwargs,
    ):
        super().__init__()
        self.config = {"latent_dim": latent_dim, **arborist_kwargs}
        self.arborist = Arborist(latent_dim=latent_dim, **arborist_kwargs)

    @torch._dynamo.disable
    def forward(self, tree_samples):
        """
        Parameters
        ----------
        tree_samples : List[TreeSample]
            One sample per proposal, in proposal-index order.

        Returns
        -------
        torch.Tensor
            Shape (n_proposals, latent_dim).
        """
        with torch.amp.autocast("cuda", enabled=False):
            zs = []
            for sample in tree_samples:
                _, z_curves = self.arborist.encode(sample)
                idx = sample.root_curve_indices or range(len(z_curves))
                zs.append(z_curves[list(idx)].mean(dim=0))
        return torch.stack(zs)

    def save(self, path):
        torch.save({"config": self.config, "state_dict": self.state_dict()}, path)

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
            conv = nn_geometric.GATv2Conv(-1, out_dim, dropout=0.1, heads=heads)
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
