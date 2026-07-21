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

from neuron_proofreader.models.new_vision_models import NewCNN3D
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
        heads=4,
        hidden_dim=128,
        n_layers=2,
    ):
        # Call parent class
        super().__init__()

        # Initial embeddings
        self.branch_encoder = init_node_encoder(hidden_dim, "branch")
        self.proposal_encoder = init_node_encoder(hidden_dim, "proposal")
        self.img_encoder = init_img_encoder(patch_shape, hidden_dim // 2)
        self.fusion = FeedForwardNet(hidden_dim, hidden_dim, 2)

        # Message passing layers
        self.gats = nn.ModuleList(
            [self.init_gat(hidden_dim, heads) for _ in range(n_layers)]
        )
        self.output = nn.Linear(hidden_dim, 1)

        # Initialize weights
        self.init_weights()

    def init_gat(self, hidden_dim, heads):
        gat_dict = dict()
        for relation in NewVisionHGAT.relations:
            node_type_1, _, node_type_2 = relation
            is_same = node_type_1 == node_type_2
            init_gat = init_gat_same if is_same else init_gat_mixed
            gat_dict[relation] = init_gat(hidden_dim, heads)
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

        # Message passing with residual connections
        for gat in self.gats:
            x_dict_new = gat(x_dict, edge_index_dict)
            x_dict = {
                key: torch.relu(x_dict_new[key]) + x_dict[key]
                for key in x_dict
            }
        return self.output(x_dict["proposal"])


# --- Helpers ---
def init_gat_same(hidden_dim, heads):
    return nn_geometric.GATv2Conv(
        -1, hidden_dim, heads=heads, concat=False, dropout=0.1
    )


def init_gat_mixed(hidden_dim, heads):
    return nn_geometric.GATv2Conv(
        (hidden_dim, hidden_dim),
        hidden_dim,
        heads=heads,
        concat=False,
        add_self_loops=False,
        dropout=0.1,
    )


def init_node_encoder(output_dim, node_type):
    """
    Builds a node embedding layer using a feed forward network for each
    node type.

    Parameters
    ----------
    output_dim : int
        Output dimension for the embeddings. Note that the proposal output
        dimension must be divided by 2 to account for the image patch
        features.
    """
    feature_dims = split_feature_extraction.get_feature_dict()
    output_dim = output_dim // (2 if node_type == "proposal" else 1)
    return FeedForwardNet(feature_dims[node_type], output_dim, 3)


def init_img_encoder(patch_shape, output_dim, vision_type="NewCNN3D"):
    """
    Builds the initial image patch embedding layer using a Convolutional
    Neural Network (CNN).

    Parameters
    ----------
    output_dim : int
        Output dimension of the embedding.
    """
    if vision_type == "NewCNN3D":
        input_shape = (2,) + patch_shape
        model = NewCNN3D(
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
