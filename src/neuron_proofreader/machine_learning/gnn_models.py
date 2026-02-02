"""
Created on Fri April 11 11:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Graph neural network architectures that classify edge proposals.

"""

from torch import nn
from torch_geometric import nn as nn_geometric

import ast
import torch
import torch.nn.init as init

from neuron_proofreader.machine_learning.vision_models import CNN3D
from neuron_proofreader.split_proofreading import split_feature_extraction
from neuron_proofreader.utils.ml_util import FeedForwardNet


# --- Models ---
class VisionHGAT(torch.nn.Module):
    """
    Heterogeneous graph attention network that processes multimodal features
    such as image patches and feature vectors.
    """
    # Class attributes
    relations = [
        str(("branch", "to", "branch")),
        str(("proposal", "to", "proposal")),
        str(("branch", "to", "proposal")),
        str(("proposal", "to", "branch")),
    ]

    def __init__(self, patch_shape, heads=2, hidden_dim=128, n_layers=2):
        # Call parent class
        super().__init__()

        # Initial embeddings
        self.node_embedding = init_node_embedding(hidden_dim)
        self.patch_embedding = init_patch_embedding(patch_shape, hidden_dim // 2)

        # Message passing layers
        self.gat1 = self.init_gat(hidden_dim, hidden_dim, heads)
        self.gat2 = self.init_gat(hidden_dim * heads, hidden_dim, heads)
        self.output = nn.Linear(hidden_dim * heads ** 2, 1)

        # Initialize weights
        self.init_weights()

    def init_gat(self, hidden_dim, edge_dim, heads):
        gat_dict = dict()
        for relation in VisionHGAT.relations:
            # Parse relation string
            relation = ast.literal_eval(relation)
            node_type_1, _, node_type_2 = relation
            is_same = node_type_1 == node_type_2

            # Initialize layer
            init_gat = init_gat_same if is_same else init_gat_mixed
            gat_dict[relation] = init_gat(hidden_dim, edge_dim, heads)
        return nn_geometric.HeteroConv(gat_dict)

    def init_weights(self):
        """
        Initializes linear layers.
        """
        for layer in [self.node_embedding, self.patch_embedding, self.output]:
            for param in layer.parameters():
                if len(param.shape) > 1:
                    init.kaiming_normal_(param)
                else:
                    init.zeros_(param)

    def forward(self, input_dict):
        # Extract inputs
        x_dict = input_dict["x_dict"]
        x_img = input_dict["img"]
        edge_index_dict = input_dict["edge_index_dict"]

        before = list()
        for key, x in x_dict.items():
            before.append(f"{key}: {x.size()}")

        # Node embeddings
        x_img = self.patch_embedding(x_img)
        for key, f in self.node_embedding.items():
            x_dict[key] = f(x_dict[key])
        x_dict["proposal"] = torch.cat((x_dict["proposal"], x_img), dim=1)

        # Message passing
        try:
            x_dict = self.gat1(x_dict, edge_index_dict)
            x_dict = self.gat2(x_dict, edge_index_dict)
        except:
            print("Before...")
            print("\n".join(before))
            print("After..."
            for key, x in x_dict.items():
                print(key, x.size())
            stop
        return self.output(x_dict["proposal"])


# --- Helpers ---
def _filter_empty(edge_index_dict):
    return {
        k: v for k, v in edge_index_dict.items()
        if v.numel() > 0
    }


def init_gat_same(hidden_dim, edge_dim, heads):
    gat = nn_geometric.GATv2Conv(
        -1, hidden_dim, dropout=0.1, edge_dim=edge_dim, heads=heads
    )
    return gat


def init_gat_mixed(hidden_dim, edge_dim, heads):
    gat = nn_geometric.GATv2Conv(
        (hidden_dim, hidden_dim),
        hidden_dim,
        add_self_loops=False,
        dropout=0.1,
        edge_dim=edge_dim,
        heads=heads,
    )
    return gat


def init_node_embedding(output_dim):
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
    # Get feature dimensions
    node_input_dims = split_feature_extraction.get_feature_dict()
    dim_b = node_input_dims["branch"]
    dim_p = node_input_dims["proposal"]

    # Set node embedding layer
    node_embedding = nn.ModuleDict({
        "branch": FeedForwardNet(dim_b, output_dim, 3),
        "proposal": FeedForwardNet(dim_p, output_dim // 2, 3),
    })
    return node_embedding


def init_patch_embedding(patch_shape, output_dim):
    """
    Builds the initial image patch embedding layer using a Convolutional
    Neural Network (CNN).

    Parameters
    ----------
    output_dim : int
        Output dimension of the embedding.
    """
    patch_embedding = CNN3D(
        patch_shape,
        output_dim=output_dim,
        n_conv_layers=6,
        n_feat_channels=24,
    )
    return patch_embedding
