"""
Created on Sat April 12 11:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Graph neural network architectures that learn to classify edge proposals.

"""

from torch import nn
from torch_geometric import nn as nn_geometric

import ast
import torch
import torch.nn.init as init

from neuron_proofreader.machine_learning.vision_models import CNN3D
from neuron_proofreader.split_proofreading import feature_extraction
from neuron_proofreader.utils.ml_util import FeedForwardNet


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

    def __init__(self, patch_shape, heads_1=2, heads_2=2, hidden_dim=128):
        # Call parent class
        super().__init__()

        # Initial embeddings
        self._init_node_embedding(hidden_dim)
        self._init_patch_embedding(patch_shape, hidden_dim // 2)

        # Message passing layers
        self.gat1 = self.init_gat(hidden_dim, hidden_dim, heads_1)
        self.gat2 = self.init_gat(hidden_dim * heads_1, hidden_dim, heads_2)
        self.output = nn.Linear(hidden_dim * heads_1 * heads_2, 1)

        # Initialize weights
        self.init_weights()

    def _init_node_embedding(self, output_dim):
        """
        Builds the initial node embedding layer using a Multi-Layer Perceptron
        (MLP) for each type of node.

        Parameters
        ----------
        output_dim : int
            Output dimension for the embeddings. Note that the proposal output
            dimension must be divided by 2 to account for the image patch
            features.
        """
        # Get feature dimensions
        node_input_dims = feature_extraction.get_node_dict()
        dim_b = node_input_dims["branch"]
        dim_p = node_input_dims["proposal"]

        # Set node embedding layer
        self.node_embedding = nn.ModuleDict({
            "branch": FeedForwardNet(dim_b, output_dim, 3),
            "proposal": FeedForwardNet(dim_p, output_dim // 2, 3),
        })

    def _init_patch_embedding(self, patch_shape, output_dim):
        """
        Builds the initial image patch embedding layer using a Convolutional
        Neural Network (CNN).

        Parameters
        ----------
        output_dim : int
            Output dimension for the embeddings.
        """
        self.patch_embedding = CNN3D(
            patch_shape,
            output_dim=output_dim,
            n_conv_layers=6,
            n_feat_channels=24,
        )

    def init_gat(self, hidden_dim, edge_dim, heads):
        gat_dict = dict()
        for relation in VisionHGAT.relations:
            # Parse relation string
            relation = ast.literal_eval(relation)
            node_type_1, edge_type, node_type_2 = relation
            is_same = node_type_1 == node_type_2

            # Initialize layer
            init_gat = init_gat_same if is_same else init_gat_mixed
            gat_dict[relation] = init_gat(hidden_dim, edge_dim, heads)
        return nn_geometric.HeteroConv(gat_dict, aggr="sum")

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
        edge_index_dict = input_dict["edge_index_dict"]

        # Node embeddings
        x_patch = self.patch_embedding(x_dict.pop("img"))
        for key, f in self.node_embedding.items():
            x_dict[key] = f(x_dict[key])
        x_dict["proposal"] = torch.cat((x_dict["proposal"], x_patch), dim=1)

        # Message passing
        x_dict = self.gat1(x_dict, edge_index_dict)
        x_dict = self.gat2(x_dict, edge_index_dict)
        return self.output(x_dict["proposal"])


# --- Helpers ---
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
