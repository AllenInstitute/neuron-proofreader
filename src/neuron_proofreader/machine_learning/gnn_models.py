"""
Created on Sat April 12 11:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Graph neural network architectures that learn to classify edge proposals.

"""

from torch import nn
from torch_geometric import nn as nn_geometric

import ast
import re
import torch
import torch.nn.init as init

from neuron_proofreader.machine_learning.vision_models import CNN3D
from neuron_proofreader.split_proofreading import feature_generation


class VisionHGAT(torch.nn.Module):
    """
    Heterogeneous graph attention network that processes multimodal features
    such as image patches and feature vectors.
    """
    # Class attributes
    relations = [
        str(("proposal", "edge", "proposal")),
        str(("branch", "edge", "proposal")),
        str(("branch", "edge", "branch")),
    ]

    def __init__(self, heads_1=2, heads_2=2, hidden_dim=128):
        # Call parent class
        super().__init__()

        # Initial embeddings
        self._init_node_embedding(hidden_dim)
        self._init_edge_embedding(hidden_dim)
        self._init_patch_embedding(hidden_dim // 2)

        # Message passing layers
        self.gat1 = self.init_gat(hidden_dim, hidden_dim, heads_1)
        self.gat2 = self.init_gat(hidden_dim * heads_1, hidden_dim, heads_2)
        self.output = nn.Linear(hidden_dim * heads_1 * heads_2, 1)

        # Initialize weights
        self.init_weights()

    # --- Class methods ---
    @classmethod
    def get_relations(cls):
        """
        Gets a list of relations expected by this architecture.

        Returns
        -------
        List[Tuple[str]]
            List of relations.
        """
        return cls.relations

    # --- Constructor Helpers ---
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
        node_input_dims = feature_generation.get_node_dict()
        input_dim_b = node_input_dims["branch"]
        input_dim_p = node_input_dims["proposal"]

        # Set node embedding layer
        self.node_embedding = nn.ModuleDict({
            "branch": init_mlp(input_dim_b, output_dim),
            "proposal": init_mlp(input_dim_p, output_dim // 2),
        })

    def _init_edge_embedding(self, output_dim):
        """
        Builds the initial edge embedding layer using a Multi-Layer Perceptron
        (MLP) for each type of node.

        Parameters
        ----------
        output_dim : int
            Output dimension for the embeddings.
        """
        self.edge_embedding = nn.ModuleDict()
        edge_input_dims = feature_generation.get_edge_dict()
        for key, input_dim in edge_input_dims.items():
            self.edge_embedding[str(key)] = init_mlp(input_dim, output_dim)

    def _init_patch_embedding(self, output_dim):
        """
        Builds the initial image patch embedding layer using a Convolutional
        Neural Network (CNN).

        Parameters
        ----------
        output_dim : int
            Output dimension for the embeddings.
        """
        self.patch_embedding = CNN3D(
            self.patch_shape,
            output_dim=output_dim,
            n_conv_layers=6,
            n_feat_channels=16,
            use_double_conv=True
        )

    def init_gat(self, hidden_dim, edge_dim, heads):
        gat_dict = dict()
        for relation in self.get_relations():
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

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        # Node embeddings
        x_patch = self.patch_embedding(x_dict.pop("patch"))
        for key, f in self.node_embedding.items():
            x_dict[key] = f(x_dict[key])
        x_dict["proposal"] = torch.cat((x_dict["proposal"], x_patch), dim=1)

        # Edge embeddings
        for key, f in self.edge_embedding.items():
            attr_key = ast.literal_eval(key)
            edge_attr_dict[attr_key] = f(edge_attr_dict[attr_key])

        # Message passing
        x_dict = self.gat1(
            x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict
        )
        x_dict = self.gat2(
            x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict
        )
        return self.output(x_dict["proposal"])


# --- Helpers ---
def init_gat_same(hidden_dim, edge_dim, heads):
    gat = nn_geometric.GATv2Conv(
        -1, hidden_dim, dropout=0.25, edge_dim=edge_dim, heads=heads
    )
    return gat


def init_gat_mixed(hidden_dim, edge_dim, heads):
    gat = nn_geometric.GATv2Conv(
        (hidden_dim, hidden_dim),
        hidden_dim,
        add_self_loops=False,
        dropout=0.25,
        edge_dim=edge_dim,
        heads=heads,
    )
    return gat


def init_mlp(input_dim, output_dim):
    """
    Initializes a multi-layer perceptron (MLP).

    Parametersƒ
    ----------
    input_dim : int
        Dimension of input feature vector.
    output_dim : int
        Dimension of embedded feature vector.

    Returns
    -------
    nn.Sequential
        ...
    """
    mlp = nn.Sequential(
        nn.Linear(input_dim, 2 * output_dim),
        nn.LeakyReLU(),
        nn.Dropout(p=0.25),
        nn.Linear(2 * output_dim, output_dim),
    )
    return mlp


def reformat_edge_key(key):
    if type(key) is str:
        return tuple([re.sub(r'\W+', '', s) for s in key.split(",")])
    else:
        return key
