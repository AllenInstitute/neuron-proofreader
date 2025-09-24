"""
Created on Sat April 12 11:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Custom datasets for training graph neural networks.

"""

from random import sample

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import HeteroData as HeteroGraphData

from neuron_proofreader.split_proofreading.feature_generation import (
    get_matrix,
    get_patches_matrix,
)
from neuron_proofreader.utils import ml_util

DTYPE = torch.float32


# Wrapper
def init(features, graph, gt_accepts=set()):
    """
    Initializes a dataset that can be used to train a graph neural network.

    Parameters
    ----------
    features : dict
        Dictionary that contains different types of feature vectors for nodes,
        edges, and proposals.
    graph : networkx.Graph
        Graph used by gnn to classify proposals.
    gt_accepts : Set[Frozenset[int]]
        Ground truth set of proposals are accepts. Default is an empty set.

    Returns
    -------
    HeteroGraphDataset
        Custom dataset.
    """
    # Extract features -- rewrite as build_feature_matrix
    idxs, x_dict = dict(), dict()
    x_dict["branches"], _, idxs["branches"] = get_matrix(features["branches"])
    x_dict["proposals"], y_proposals, idxs["proposals"] = get_matrix(
        features["proposals"], gt_accepts
    )
    x_dict["nodes"] = features["nodes"]

    # Build patch matrix
    if "patches" in features:
        x_dict["patch"] = get_patches_matrix(
            features["patches"], idxs["proposals"]["id_to_idx"]
        )

    # Initialize dataset
    proposals = list(features["proposals"].keys())
    if "patches" in features:
        Dataset = HeteroGraphMultiModalDataset
    else:
        Dataset = HeteroGraphDataset

    dataset = Dataset(
        graph,
        proposals,
        x_dict,
        y_proposals,
        idxs,
    )
    return dataset


# Datasets
class HeteroGraphDataset:
    """
    Custom dataset for heterogenous graphs.
    """

    def __init__(
        self,
        graph,
        proposals,
        x_dict,
        y_proposals,
        idxs,
    ):
        """
        Constructs a HeteroGraphDataset object.

        Parameters
        ----------
        graph : networkx.Graph
            Graph used by gnn to classify proposals.
        proposals : list
            List of proposals to be classified.
        x_nodes : numpy.ndarray
            Feature matrix generated from nodes in "graph".
        x_branches : numpy.ndarray
            Feature matrix generated from branches in "graph".
        x_proposals : numpy.ndarray
            Feature matrix generated from "proposals" in "graph".
        y_proposals : numpy.ndarray
            Ground truth of proposals.
        idx_to_id : dict
            Dictionary that maps an edge id in "graph" to its index
            in either x_branches or x_proposals.
        """
        # Conversion idxs
        self.idxs_branches = idxs["branches"]
        self.idxs_proposals = idxs["proposals"]
        self.graph = graph
        self.proposals = proposals

        # Types
        self.node_types = ["branch", "proposal"]
        self.edge_types = [
            ("proposal", "edge", "proposal"),
            ("branch", "edge", "branch"),
            ("branch", "edge", "proposal"),
        ]

        # Features
        self.init_nodes(x_dict, y_proposals)
        self.init_edges()
        self.check_missing_edge_types()
        self.init_edge_attrs(x_dict["nodes"])
        self.n_edge_attrs = n_edge_features(x_dict["nodes"])

    def init_nodes(self, x_dict, y_proposals):
        self.data = HeteroGraphData()
        self.data["branch"].x = torch.tensor(x_dict["branches"], dtype=DTYPE)
        self.data["proposal"].x = torch.tensor(x_dict["proposals"], dtype=DTYPE)
        self.data["proposal"].y = torch.tensor(y_proposals, dtype=DTYPE)

    def init_edges(self):
        """
        Initializes edge index for a graph dataset.
        """
        # Compute edges
        proposal_edges = self.proposal_to_proposal()
        branch_edges = self.branch_to_branch()
        branch_proposal_edges = self.branch_to_proposal()

        # Store edges
        self.data["proposal", "edge", "proposal"].edge_index = proposal_edges
        self.data["branch", "edge", "branch"].edge_index = branch_edges
        self.data["branch", "edge", "proposal"].edge_index = branch_proposal_edges

    def init_edge_attrs(self, x_nodes):
        """
        Initializes edge attributes.
        """
        # Proposal edges
        edge_type = ("proposal", "edge", "proposal")
        self.set_edge_attrs(x_nodes, edge_type, self.idxs_proposals)

        # Branch edges
        edge_type = ("branch", "edge", "branch")
        self.set_edge_attrs(x_nodes, edge_type, self.idxs_branches)

        # Branch-Proposal edges
        edge_type = ("branch", "edge", "proposal")
        self.set_hetero_edge_attrs(
            x_nodes, edge_type, self.idxs_branches, self.idxs_proposals
        )

    def check_missing_edge_types(self):
        for node_type in ["branch", "proposal"]:
            edge_type = (node_type, "edge", node_type)
            if len(self.data[edge_type].edge_index) == 0:
                # Add dummy features
                dtype = self.data[node_type].x.dtype
                if node_type == "branch":
                    d = self.n_branch_features()
                else:
                    d = self.n_proposal_features()

                zeros = torch.zeros(2, d, dtype=dtype)
                self.data[node_type].x = torch.cat(
                    (self.data[node_type].x, zeros), dim=0
                )

                # Update edge_index
                n = self.data[node_type]["x"].size(0)
                e_1 = frozenset({-1, -2})
                e_2 = frozenset({-2, -3})
                edges = [[n - 1, n - 2], [n - 2, n - 1]]
                self.data[edge_type].edge_index = toTensor(edges)
                if node_type == "branch":
                    self.idxs_branches["idx_to_id"][n - 1] = e_1
                    self.idxs_branches["idx_to_id"][n - 2] = e_2
                else:
                    self.idxs_proposals["idx_to_id"][n - 1] = e_1
                    self.idxs_proposals["idx_to_id"][n - 2] = e_2

    # -- Getters --
    def n_branch_features(self):
        """
        Gets the dimension of feature vector for branches.

        Returns
        -------
        int
            Dimension of feature vector for branches.
        """
        return self.data["branch"]["x"].size(1)

    def n_proposal_features(self):
        """
        Gets the dimension of feature vector for proposals.

        Returns
        -------
        int
            Dimension of feature vector for proposals.

        """
        return self.data["proposal"]["x"].size(1)

    def n_edge_features(self):
        """
        Gets the dimension of feature vector for edges.

        Returns
        -------
        int
            Dimension of feature vector for edges.
        """
        edge_type = ("proposal", "edge", "proposal")
        return self.data[edge_type]["x"].size(1)

    def n_proposals(self):
        """
        Counts the number of proposals in the dataset.

        Returns
        -------
        int
            Number of proposals across all graphs in self.
        """
        return len(self.proposals)

    # --- Set Edges ---
    def proposal_to_proposal(self):
        """
        Generates edge indices between nodes corresponding to proposals.

        Returns
        -------
        list
            Edges generated from line_graph generated from proposals which are
            part of an edge_index for a graph dataset.
        """
        edge_index = []
        line_graph = ml_util.line_graph(self.proposals)
        for e1, e2 in line_graph.edges:
            v1 = self.idxs_proposals["id_to_idx"][frozenset(e1)]
            v2 = self.idxs_proposals["id_to_idx"][frozenset(e2)]
            edge_index.extend([[v1, v2], [v2, v1]])
        return toTensor(edge_index)

    def branch_to_branch(self):
        """
        Generates edge indices between nodes corresponding to branches
        (i.e. edges in some neurograph).

        Returns
        -------
        torch.Tensor
            Edges generated from "branches_line_graph" which are a subset of
            an edge_index for a graph dataset.
        """
        edge_index = []
        for e1, e2 in nx.line_graph(self.graph).edges:
            e1_edge_bool = frozenset(e1) not in self.proposals
            e2_edge_bool = frozenset(e2) not in self.proposals
            if e1_edge_bool and e2_edge_bool:
                v1 = self.idxs_branches["id_to_idx"][frozenset(e1)]
                v2 = self.idxs_branches["id_to_idx"][frozenset(e2)]
                edge_index.extend([[v1, v2], [v2, v1]])
        return toTensor(edge_index)

    def branch_to_proposal(self):
        """
        Generates edge indices between nodes that correspond to proposals and
        edges.

        Returns
        -------
        torch.Tensor
            Edges generated from proposals which are a subset of an edge_index
            for a graph dataset.
        """
        edge_index = []
        for p in self.proposals:
            i, j = tuple(p)
            v1 = self.idxs_proposals["id_to_idx"][frozenset(p)]
            for k in self.graph.neighbors(i):
                if frozenset((i, k)) not in self.proposals:
                    v2 = self.idxs_branches["id_to_idx"][frozenset((i, k))]
                    edge_index.extend([[v2, v1]])
            for k in self.graph.neighbors(j):
                if frozenset((j, k)) not in self.proposals:
                    v2 = self.idxs_branches["id_to_idx"][frozenset((j, k))]
                    edge_index.extend([[v2, v1]])
        return toTensor(edge_index)

    # Set Edge Attributes
    def set_edge_attrs(self, x_nodes, edge_type, idx_map):
        """
        Generate proposal edge attributes in the case where the edge connects
        nodes with the same type.

        Parameters
        ----------
        ...
        """
        attrs = []
        if self.data[edge_type].edge_index.size(0) > 0:
            for i in range(self.data[edge_type].edge_index.size(1)):
                e1, e2 = self.data[edge_type].edge_index[:, i]
                v = node_intersection(idx_map, e1, e2)
                if v < 0:
                    attrs.append(np.zeros(self.n_branch_features() + 1))
                else:
                    attrs.append(x_nodes[v])
        arrs = torch.tensor(np.array(attrs), dtype=DTYPE)
        self.data[edge_type].edge_attr = arrs

    def set_hetero_edge_attrs(self, x_nodes, edge_type, idx_map_1, idx_map_2):
        """
        Generate proposal edge attributes in the case where the edge connects
        nodes with different types.

        Parameters
        ----------
        ...
        """
        attrs = []
        for i in range(self.data[edge_type].edge_index.size(1)):
            e1, e2 = self.data[edge_type].edge_index[:, i]
            v = hetero_node_intersection(idx_map_1, idx_map_2, e1, e2)
            attrs.append(x_nodes[v])
        arrs = torch.tensor(np.array(attrs), dtype=DTYPE)
        self.data[edge_type].edge_attr = arrs


class HeteroGraphMultiModalDataset(HeteroGraphDataset):

    def __init__(
        self,
        graph,
        proposals,
        x_dict,
        y_proposals,
        idxs,
    ):
        # Call parent class
        super().__init__(
            graph,
            proposals,
            x_dict,
            y_proposals,
            idxs,
        )

    def init_nodes(self, x_dict, y_proposals):
        self.data = HeteroGraphData()
        self.data["branch"].x = torch.tensor(x_dict["branches"], dtype=DTYPE)
        self.data["proposal"].x = torch.tensor(x_dict["proposals"], dtype=DTYPE)
        self.data["proposal"].y = torch.tensor(y_proposals, dtype=DTYPE)
        self.data["patch"].x = torch.tensor(x_dict["patch"], dtype=DTYPE)

    def check_missing_edge_types(self):
        for node_type in ["branch", "proposal"]:
            edge_type = (node_type, "edge", node_type)
            if len(self.data[edge_type].edge_index) == 0:
                # Add dummy features - nodes
                dtype = self.data[node_type].x.dtype
                if node_type == "branch":
                    d = self.n_branch_features()
                else:
                    d = self.n_proposal_features()

                zeros = torch.zeros(2, d, dtype=dtype)
                self.data[node_type].x = torch.cat(
                    (self.data[node_type].x, zeros), dim=0
                )

                # Add dummy features - patches
                if node_type == "proposal":
                    patch_shape = self.data["patch"].x.size()[1:]
                    zeros = torch.zeros((2,) + patch_shape, dtype=dtype)
                    self.data["patch"].x = torch.cat(
                        (self.data["patch"].x, zeros), dim=0
                    )

                # Update edge_index
                n = self.data[node_type]["x"].size(0)
                e_1 = frozenset({-1, -2})
                e_2 = frozenset({-2, -3})
                edges = [[n - 1, n - 2], [n - 2, n - 1]]
                self.data[edge_type].edge_index = toTensor(edges)
                if node_type == "branch":
                    self.idxs_branches["idx_to_id"][n - 1] = e_1
                    self.idxs_branches["idx_to_id"][n - 2] = e_2
                else:
                    self.idxs_proposals["idx_to_id"][n - 1] = e_1
                    self.idxs_proposals["idx_to_id"][n - 2] = e_2


# -- util --
def node_intersection(idx_map, e1, e2):
    """
    Computes the common node between "e1" and "e2" in the case where these
    edges connect nodes of the same type.

    Parameters
    ----------
    e1 : torch.Tensor
        Edge to be checked.
    e2 : torch.Tensor
        Edge to be checked.

    Returns
    -------
    int
        Common node between "e1" and "e2".
    """
    hat_e1 = idx_map["idx_to_id"][int(e1)]
    hat_e2 = idx_map["idx_to_id"][int(e2)]
    node = list(hat_e1.intersection(hat_e2))
    assert len(node) == 1, "Node intersection is not unique!"
    return node[0]


def hetero_node_intersection(idx_map_1, idx_map_2, e1, e2):
    """
    Computes the common node between "e1" and "e2" in the case where these
    edges connect nodes of different types.

    Parameters
    ----------
    e1 : torch.Tensor
        Edge to be checked.
    e2 : torch.Tensor
        Edge to be checked.

    Returns
    -------
    int
        Common node between "e1" and "e2".
    """
    hat_e1 = idx_map_1["idx_to_id"][int(e1)]
    hat_e2 = idx_map_2["idx_to_id"][int(e2)]
    node = list(hat_e1.intersection(hat_e2))
    assert len(node) == 1, "Node intersection is empty or not unique!"
    return node[0]


def n_edge_features(x):
    """
    Gets the number of edge features.

    Parameters
    ----------
    x : dict
        Dictionary that maps node (from a neurograph) to feature vectors.

    Returns
    -------
    int
        Number of edge features.
    """
    key = sample(list(x.keys()), 1)[0]
    return x[key].shape[0]


def toTensor(arr):
    """
    Converts an array to a tensor with contiguous memory.

    Parameters
    ----------
    arr : ArrayLike
        Array to be converted.

    Returns
    -------
    torch.Tensor
        Tensor.
    """
    return torch.Tensor(arr).t().contiguous().long()
