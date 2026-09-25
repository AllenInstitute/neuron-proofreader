"""
Created on Fri Sep 25 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Split-correction GNN models using the updated CNN3D vision backbone.

"""

from torch_geometric import nn as nn_geometric

import torch
import torch.nn as nn

from neuron_proofreader.models.new_vision_models import CNN3D
from neuron_proofreader.split_proofreading import split_feature_extraction
from neuron_proofreader.utils.ml_util import FeedForwardNet


class NewVisionHGAT(nn.Module):
    """
    Heterogeneous graph attention network for split correction.

    Uses the updated CNN3D backbone from new_vision_models for image patch
    encoding. Branch and proposal nodes are embedded from geometry features;
    proposal nodes are then augmented with per-patch image embeddings before
    two rounds of heterogeneous message passing classify each proposal.
    """

    _RELATIONS = [
        ("branch", "to", "branch"),
        ("proposal", "to", "proposal"),
        ("branch", "to", "proposal"),
        ("proposal", "to", "branch"),
    ]

    def __init__(
        self,
        patch_shape,
        geometry_embed_dim=64,
        img_embed_dim=64,
        heads=4,
    ):
        """
        Instantiates a NewVisionHGAT object.

        Parameters
        ----------
        patch_shape : Tuple[int]
            Spatial shape of image patches (D, H, W) — no channel dimension.
        geometry_embed_dim : int, optional
            Hidden dimension for branch and proposal geometry embeddings.
            Default is 64.
        img_embed_dim : int, optional
            Output dimension of the image patch encoder. Default is 64.
        heads : int, optional
            Number of attention heads per GAT layer. Default is 4.
        """
        super().__init__()

        feats = split_feature_extraction.get_feature_dict()

        # Node feature embeddings: branch→geometry_embed_dim,
        # proposal→geometry_embed_dim//2 (img features fill the other half).
        self.node_embedding = nn.ModuleDict({
            "branch": FeedForwardNet(feats["branch"], geometry_embed_dim, 3),
            "proposal": FeedForwardNet(
                feats["proposal"], geometry_embed_dim // 2, 3
            ),
        })

        # Image patch embedding: 2-channel input (raw image + segmentation mask).
        self.patch_embedding = CNN3D(
            (2,) + tuple(patch_shape),
            output_dim=img_embed_dim,
            use_output_head=True,
        )

        # Heterogeneous GAT layers.
        # After node embedding + image fusion, proposal nodes have dim
        # geometry_embed_dim//2 + img_embed_dim; branch nodes have dim
        # geometry_embed_dim.  GATv2Conv handles mismatched dims via lazy
        # (-1) input initialisation.
        self.gat1 = _build_hetero_gat(geometry_embed_dim, heads)
        self.gat2 = _build_hetero_gat(geometry_embed_dim * heads, heads)
        self.output = nn.Linear(geometry_embed_dim * heads ** 2, 1)

        self._init_weights()

    def _init_weights(self):
        for module in [self.node_embedding, self.output]:
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.kaiming_normal_(p)
                else:
                    nn.init.zeros_(p)

    def forward(self, input_dict):
        """
        Parameters
        ----------
        input_dict : dict
            Must contain:
              "img"            – (N_proposals, 2, D, H, W) image patches
              "x_dict"         – dict mapping node type → feature tensor
              "edge_index_dict" – dict mapping edge type → (2, E) index tensor

        Returns
        -------
        torch.Tensor
            Per-proposal logits with shape (N_proposals, 1).
        """
        x_dict = input_dict["x_dict"]
        x_img = input_dict["img"]
        edge_index_dict = input_dict["edge_index_dict"]

        # Patches may arrive as float16 from the dataloader; convolutions need
        # them in the parameter dtype when not running inside autocast.
        if not torch.is_autocast_enabled():
            x_img = x_img.to(next(self.patch_embedding.parameters()).dtype)
        x_img = self.patch_embedding(x_img)

        # Embed node geometry features, then fuse image embeddings into proposals.
        for key, f in self.node_embedding.items():
            x_dict[key] = f(x_dict[key])
        x_dict["proposal"] = torch.cat((x_dict["proposal"], x_img), dim=1)

        # Two rounds of heterogeneous message passing.
        x_dict = self.gat1(x_dict, edge_index_dict)
        x_dict = self.gat2(x_dict, edge_index_dict)
        return self.output(x_dict["proposal"])


def _build_hetero_gat(out_dim, heads):
    """
    Builds a HeteroConv with GATv2Conv for every relation in NewVisionHGAT.
    Same-type relations use a single input-dim spec; cross-type relations use
    a pair so GATv2Conv can handle bipartite input.
    """
    gat_dict = {}
    for rel in NewVisionHGAT._RELATIONS:
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
