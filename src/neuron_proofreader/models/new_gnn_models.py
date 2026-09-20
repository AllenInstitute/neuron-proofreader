"""
Created on Fri April 11 11:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Graph neural network architectures that classify edge proposals.

"""

from torch import nn
from torch_geometric import nn as nn_geometric

import torch
import torch.nn.init as init

from neuron_proofreader.models.new_vision_models import CNN3D
from neuron_proofreader.models.simple_models import FeedForwardNet
from neuron_proofreader.split_proofreading import split_feature_extraction


# --- Models ---
class NewVisionHGAT(torch.nn.Module):
    """
    Heterogeneous graph attention network that processes multimodal features
    such as image patches and feature vectors.
    """

    # Class attributes
    relations = [
        ("branch", "to", "branch"),
        ("proposal", "to", "proposal"),
        ("branch", "to", "proposal"),
        ("proposal", "to", "branch"),
    ]

    def __init__(
        self,
        patch_shape,
        concat_heads=False,
        geometry_embed_dim=64,
        heads=4,
        hidden_dim=128,
        img_embed_dim=64,
        n_layers=2,
    ):
        """
        Instantiates a NewVisionHGAT object.

        Parameters
        ----------
        patch_shape : Tuple[int]
            Shape of input image patch.
        concat_heads : bool, optional
            If True, attention heads are concatenated and projected back to
            "hidden_dim" after each message passing layer, preserving head
            diversity. If False, heads are averaged. Default is False.
        geometry_embed_dim : int, optional
            Dimension of the proposal (geometric) feature embedding. Default
            is 64.
        heads : int, optional
            Number of attention heads. Default is 4.
        hidden_dim : int, optional
            Dimension of node embeddings during message passing. Default is
            128.
        img_embed_dim : int, optional
            Dimension of the image patch embedding. Default is 64.
        n_layers : int, optional
            Number of message passing layers. Default is 2.
        """
        # Call parent class
        super().__init__()

        # Store config for checkpointing
        self.config = {
            "patch_shape": patch_shape,
            "concat_heads": concat_heads,
            "geometry_embed_dim": geometry_embed_dim,
            "heads": heads,
            "hidden_dim": hidden_dim,
            "img_embed_dim": img_embed_dim,
            "n_layers": n_layers,
        }

        # Initial embeddings. The fusion layer maps the concatenated image
        # and geometric embeddings onto "hidden_dim", so the two embedding
        # dimensions can be set independently of it. Note: branch features
        # are low-dimensional, so the branch encoder uses a single MLP block
        # rather than the deeper encoder used for proposals.
        feature_dims = split_feature_extraction.get_feature_dict()
        self.branch_encoder = FeedForwardNet(
            feature_dims["branch"], hidden_dim, 1
        )
        self.proposal_encoder = FeedForwardNet(
            feature_dims["proposal"], geometry_embed_dim, 3
        )
        self.img_encoder = init_img_encoder(patch_shape, img_embed_dim)
        self.fusion = FeedForwardNet(
            img_embed_dim + geometry_embed_dim, hidden_dim, 2
        )

        # Message passing layers. The final layer only computes messages
        # into proposal nodes since the output head does not read branch
        # embeddings — convs writing to branch nodes in the last layer would
        # be dead weight.
        self.concat_heads = concat_heads
        last_relations = [
            r for r in NewVisionHGAT.relations if r[2] == "proposal"
        ]
        layer_relations = [
            NewVisionHGAT.relations if i < n_layers - 1 else last_relations
            for i in range(n_layers)
        ]
        self.gats = nn.ModuleList(
            [
                self.init_gat(hidden_dim, heads, concat_heads, relations)
                for relations in layer_relations
            ]
        )
        if concat_heads:
            self.head_projs = nn.ModuleList(
                [
                    nn_geometric.HeteroDictLinear(
                        hidden_dim * heads,
                        hidden_dim,
                        types=tuple({r[2] for r in relations}),
                    )
                    for relations in layer_relations
                ]
            )
        else:
            self.head_projs = nn.ModuleList(
                [nn.Identity() for _ in range(n_layers)]
            )
        self.output = nn.Linear(hidden_dim, 1)

        # Initialize weights
        self.init_weights()

    def init_gat(self, hidden_dim, heads, concat_heads, relations):
        gat_dict = dict()
        for relation in relations:
            node_type_1, _, node_type_2 = relation
            is_same = node_type_1 == node_type_2
            init_gat = init_gat_same if is_same else init_gat_mixed
            gat_dict[relation] = init_gat(hidden_dim, heads, concat_heads)
        return nn_geometric.HeteroConv(gat_dict)

    def init_weights(self):
        norm_types = (nn.GroupNorm, nn.BatchNorm3d, nn.LayerNorm)
        for layer in [self.proposal_encoder, self.img_encoder, self.output]:
            for module in layer.modules():
                if isinstance(module, norm_types):
                    continue
                for param in module.parameters(recurse=False):
                    if len(param.shape) > 1:
                        init.kaiming_normal_(param)
                    else:
                        init.zeros_(param)

    def forward(self, input_dict):
        x_dict = input_dict["x_dict"]
        x_img = input_dict["img"]
        edge_index_dict = input_dict["edge_index_dict"]

        # Initial embedding
        x_img = self.img_encoder(x_img)
        x_dict["branch"] = self.branch_encoder(x_dict["branch"])
        x_dict["proposal"] = self.proposal_encoder(x_dict["proposal"])

        # Combine image and geometric embeddings, then fuse
        x_dict["proposal"] = torch.cat((x_dict["proposal"], x_img), dim=1)
        x_dict["proposal"] = self.fusion(x_dict["proposal"])

        # Message passing with residual connections. The final layer only
        # updates proposal embeddings, so node types absent from a layer's
        # output are carried through unchanged.
        for gat, proj in zip(self.gats, self.head_projs):
            x_dict_new = gat(x_dict, edge_index_dict)
            if self.concat_heads:
                x_dict_new = proj(x_dict_new)
            x_dict = {
                key: torch.relu(x_dict_new[key]) + x_dict[key]
                if key in x_dict_new
                else x_dict[key]
                for key in x_dict
            }
        return self.output(x_dict["proposal"])


# --- Helpers ---
def init_gat_same(hidden_dim, heads, concat_heads=False):
    return nn_geometric.GATv2Conv(
        -1, hidden_dim, heads=heads, concat=concat_heads, dropout=0.1
    )


def init_gat_mixed(hidden_dim, heads, concat_heads=False):
    return nn_geometric.GATv2Conv(
        (hidden_dim, hidden_dim),
        hidden_dim,
        heads=heads,
        concat=concat_heads,
        add_self_loops=False,
        dropout=0.1,
    )


def init_img_encoder(patch_shape, output_dim, vision_type="CNN3D"):
    """
    Builds the initial image patch embedding layer using a Convolutional
    Neural Network (CNN).

    Parameters
    ----------
    output_dim : int
        Output dimension of the embedding.
    """
    if vision_type == "CNN3D":
        input_shape = (2,) + patch_shape
        model = CNN3D(
            input_shape,
            base_channels=32,
            depth=4,
            max_channels=256,
            output_dim=output_dim,
            use_double=True,
        )
    else:
        assert "Invalid model type"
    return model
