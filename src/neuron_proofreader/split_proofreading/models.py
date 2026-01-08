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

from neuron_proofreader.split_proofreading import feature_extraction


class HGAT(torch.nn.Module):
    """
    Heterogeneous graph attention network that classifies proposals.
    """
    # Class attributes
    relation_types = [
        ("proposal", "edge", "proposal"),
        ("branch", "edge", "proposal"),
        ("branch", "edge", "branch"),
    ]

    def __init__(
        self,
        hidden_dim=64,
        dropout=0.2,
        heads_1=2,
        heads_2=2,
    ):
        """
        Constructs a heterogeneous graph neural network.
        """
        super().__init__()
        # Nonlinear activation
        self.dropout = dropout
        self.leaky_relu = nn.LeakyReLU()

        # Initial Embedding
        self.input_nodes = nn.ModuleDict()
        node_dict = feature_extraction.get_node_dict()
        for key, d in node_dict.items():
            self.input_nodes[key] = nn.Linear(d, hidden_dim)

        self.input_edges = dict()
        edge_dict = feature_extraction.get_edge_dict()
        for key, d in edge_dict.items():
            self.input_edges[key] = nn.Linear(d, hidden_dim)

        # Layer dimensions
        hidden_dim_1 = hidden_dim
        hidden_dim_2 = hidden_dim_1 * heads_2
        output_dim = hidden_dim_1 * heads_1 * heads_2

        # Message passing layers
        self.gat1 = self.init_gat_layer(hidden_dim_1, hidden_dim_1, heads_1)
        self.gat2 = self.init_gat_layer(hidden_dim_2, hidden_dim_1, heads_2)
        self.output = nn.Linear(output_dim, 1)

        # Initialize weights
        self.init_weights()

    # --- Class methods ---
    @classmethod
    def get_relation_types(cls):
        return cls.relation_types

    # --- Architecture ---
    def init_gat_layer(self, hidden_dim, edge_dim, heads):
        gat_dict = dict()
        for r in self.get_relation_types():
            is_same = True if r[0] == r[2] else False
            init_gat = self.init_gat_same if is_same else self.init_gat_mixed
            gat_dict[r] = init_gat(hidden_dim, edge_dim, heads)
        return nn_geometric.HeteroConv(gat_dict, aggr="sum")

    def init_gat_same(self, hidden_dim, edge_dim, heads):
        gat_layer = nn_geometric.GATv2Conv(
            -1,
            hidden_dim,
            dropout=self.dropout,
            edge_dim=edge_dim,
            heads=heads,
        )
        return gat_layer

    def init_gat_mixed(self, hidden_dim, edge_dim, heads):
        gat_layer = nn_geometric.GATv2Conv(
            (hidden_dim, hidden_dim),
            hidden_dim,
            add_self_loops=False,
            edge_dim=edge_dim,
            heads=heads,
        )
        return gat_layer

    def init_weights(self):
        """
        Initializes linear layers.
        """
        for layer in [self.input_nodes, self.output]:
            for param in layer.parameters():
                if len(param.shape) > 1:
                    init.kaiming_normal_(param)
                else:
                    init.zeros_(param)

    def activation(self, x_dict):
        """
        Applies nonlinear activation

        Parameters
        ----------
        x_dict : dict
            Dictionary that maps node/edge types to feature matrices.

        Returns
        -------
        dict
            Feature matrices with activation applied.
        """
        x_dict = {key: self.leaky_relu(x) for key, x in x_dict.items()}
        return x_dict

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        # Input - Nodes
        x_dict = {key: f(x_dict[key]) for key, f in self.input_nodes.items()}
        x_dict = self.activation(x_dict)

        # Input - Edges
        for key, f in self.input_edges.items():
            key = reformat_edge_key(key)
            edge_attr_dict[key] = f(edge_attr_dict[key])
        edge_attr_dict = self.activation(edge_attr_dict)

        # Message passing layers
        x_dict = self.gat1(
            x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict
        )
        x_dict = self.gat2(
            x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict
        )

        # Output
        x_dict = self.output(x_dict["proposal"])
        return x_dict


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
        node_input_dims = feature_extraction.get_node_dict()
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
        edge_input_dims = feature_extraction.get_edge_dict()
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
        self.patch_embedding = ConvNet(output_dim)

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


class ConvNet(nn.Module):
    """
    Convolutional neural network that classifies edge proposals given an image
    patch.
    """

    def __init__(self, output_dim):
        """
        Constructs a ConvNet object.
        """
        # Call parent class
        nn.Module.__init__(self)

        # Layer 1
        self.conv1a = self._init_conv_layer(2, 16, 3)
        self.conv1b = self._init_conv_layer(16, 16, 3)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Layer 2
        self.conv2 = self._init_conv_layer(16, 32, 3)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Layer 3
        self.conv3 = self._init_conv_layer(32, 64, 3)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Layer 4
        self.conv4 = self._init_conv_layer(64, 64, 3)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Output layer
        self.output = init_mlp(4096, output_dim)

        # Initialize weights
        self.apply(self.init_weights)

    def _init_conv_layer(self, in_channels, out_channels, kernel_size):
        """
        Initializes a convolutional layer.

        Parameters
        ----------
        in_channels : int
            Number of channels that are input to this convolutional layer.
        out_channels : int
            Number of channels that are output from this convolutional layer.
        kernel_size : int
            Size of kernel used on convolutional layers.

        Returns
        -------
        torch.nn.Sequential
            Sequence of operations that define this layer.
        """
        conv_layer = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding="same",
            ),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(p=0.25),
        )
        return conv_layer

    @staticmethod
    def init_weights(m):
        """
        Initializes the weights and biases of a given PyTorch layer.

        Parameters
        ----------
        m : nn.Module
            PyTorch layer or module.
        """
        if isinstance(m, nn.Conv3d):
            init.kaiming_normal_(
                m.weight, mode="fan_in", nonlinearity="leaky_relu"
            )
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm3d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Passes an input vector "x" through this neural network.

        Parameters
        ----------
        x : torch.Tensor
            Input vector of features.

        Returns
        -------
        torch.Tensor
            Output of neural network.
        """
        # Convolutional Layers
        x = self.pool1(self.conv1b(self.conv1a(x)))
        x = self.conv2(x)
        x = self.pool2(x)

        # Layer 3
        x = self.conv3(x)
        x = self.pool3(x)

        # Layer 4
        x = self.conv4(x)
        x = self.pool4(x)

        # Output layer
        x = self.output(x.view(x.size(0), -1))
        return x


# --- Helpers ---
def init_model(is_multimodal, heads_1=2, heads_2=4):
    """
    Initialize a HGAT or MultiModalHGAT model based on modality flag.

    Parameters
    ----------
    is_multimodal : bool
        If True, initialize a MultiModalHGAT model; otherwise, use HGAT.
    heads_1 : int, optional
        Number of attention heads in the first GAT layer. Default is 2.
    heads_2 : int, optional
        Number of attention heads in the second GAT layer. Default is 2.

    Returns
    -------
    torch.nn.Module
        An initialized instance of HGAT or MultiModalHGAT.
    """
    node_dict = feature_extraction.get_node_dict()
    edge_dict = feature_extraction.get_edge_dict()
    if is_multimodal:
        model = VisionHGAT(
            node_dict, edge_dict, heads_1=heads_1, heads_2=heads_2
        )
    else:
        model = HGAT(node_dict, edge_dict, heads_1=heads_1, heads_2=heads_2)
    return model


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
