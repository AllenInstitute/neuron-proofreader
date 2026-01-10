"""
Created on Sat April 12 11:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Custom datasets for training graph neural networks.

"""

from torch_geometric.data import HeteroData
from torch_geometric.utils import to_undirected

import numpy as np
import torch

from neuron_proofreader.utils import graph_util as gutil


class HeteroGraphData(HeteroData):
    """
    A custom data class for heterogenous graphs. The graph is internally
    represented as a line graph to facilitate edge-based message passing in
    a GNN.
    """

    def __init__(self, graph, features):
        # Call parent class
        super().__init__()

        # Index mappings
        self.idxs_branches = features.edge_index_mapping
        self.idxs_proposals = features.proposal_index_mapping

        # Node features
        self["branch"].x = torch.tensor(features.edge_features)
        self["proposal"].x = torch.tensor(features.proposal_features)
        self["proposal"].y = torch.tensor(features.targets)
        self["patch"].x = torch.tensor(features.proposal_features)

        # Edge indices
        self.build_proposal_adjacency(graph)
        self.build_branch_adjacency(graph)
        self.build_proposal_branch_adjacency(graph)

        # Edge features
        self.init_edge_attrs(features.node_features)

    # --- Core Routines ---
    def build_proposal_adjacency(self, graph):
        """
        Builds proposal–proposal adjacency based on shared node incidence.

        Parameters
        ----------
        graph : ProposalGraph
            Graph containing proposals.
        """
        edges = graph.proposals
        edge_index = self._build_adjacency(edges, self.idxs_proposals)
        self.set_edge_index(edge_index, ("proposal", "to", "proposal"))

    def build_branch_adjacency(self, graph):
        """
        Builds branch–branch adjacency based on shared node incidence.

        Parameters
        ----------
        graph : ProposalGraph
            Graph containing branches.
        """
        edge_index = self._build_adjacency(graph.edges, self.idxs_branches)
        self.set_edge_index(edge_index, ("branch", "to", "branch"))

    def build_branch_proposal_adjacency(self, graph):
        """
        Builds branch–proposal adjacency based on shared node incidence.

        Parameters
        ----------
        graph : ProposalGraph
            Graph containing branches and proposals.
        """
        edge_index = []
        for proposal in graph.proposals:
            idx_proposal = self.idxs_proposals["id_to_idx"][proposal]
            for i in proposal:
                for j in graph.neighbors(i):
                    branch = frozenset((i, j))
                    idx_branch = self.idxs_branches["id_to_idx"][branch]
                    edge_index.append([idx_proposal, idx_branch])
        self.set_edge_index(edge_index, ("branch", "to", "proposal"))

    # Set Edge Attributes
    def init_edge_attrs(self, x_nodes):
        """
        Initializes edge attributes.
        """
        # Proposal edges
        edge_type = ("proposal", "to", "proposal")
        self.set_edge_attrs(x_nodes, edge_type, self.idxs_proposals)

        # Branch edges
        edge_type = ("branch", "to", "branch")
        self.set_edge_attrs(x_nodes, edge_type, self.idxs_branches)

        # Branch-Proposal edges
        edge_type = ("branch", "to", "proposal")
        self.set_hetero_edge_attrs(
            x_nodes, edge_type, self.idxs_branches, self.idxs_proposals
        )

    def set_edge_attrs(self, x_nodes, edge_type, idx_map):
        """
        Generate proposal edge attributes in the case where the edge connects
        nodes with the same type.

        Parameters
        ----------
        ...
        """
        attrs = []
        if self[edge_type].edge_index.size(0) > 0:
            for i in range(self[edge_type].edge_index.size(1)):
                e1, e2 = self[edge_type].edge_index[:, i]
                v = node_intersection(idx_map, e1, e2)
                if v < 0:
                    attrs.append(np.zeros(self["branch"]["x"].size(1) + 1))
                else:
                    attrs.append(x_nodes[v])
        arrs = torch.tensor(np.array(attrs))
        self[edge_type].edge_attr = arrs

    def set_hetero_edge_attrs(self, x_nodes, edge_type, idx_map_1, idx_map_2):
        """
        Generate proposal edge attributes in the case where the edge connects
        nodes with different types.

        Parameters
        ----------
        ...
        """
        attrs = []
        for i in range(self[edge_type].edge_index.size(1)):
            e1, e2 = self[edge_type].edge_index[:, i]
            v = hetero_node_intersection(idx_map_1, idx_map_2, e1, e2)
            attrs.append(x_nodes[v])
        arrs = torch.tensor(np.array(attrs))
        self[edge_type].edge_attr = arrs

    # --- Helpers ---
    @staticmethod
    def _build_adjacency(self, edges, index_mapping):
        # Build edge index
        edge_index = []
        line_graph = gutil.edges_to_line_graph(edges)
        for e1, e2 in line_graph.edges:
            v1 = index_mapping["id_to_idx"][frozenset(e1)]
            v2 = index_mapping["id_to_idx"][frozenset(e2)]
            edge_index.append([v1, v2])
        return edge_index

    def set_edge_index(self, edge_index, edge_type):
        # Check if edge index is empty
        if len(edge_index) == 0:
            self[edge_type].edge_index = torch.empty((2, 0), dtype=torch.long)

        # Reformat edge index
        edge_index = to_undirected(edge_index)
        edge_index = torch.Tensor(edge_index).t().contiguous().long()
        self[edge_type].edge_index = edge_index
        self[edge_type[::-1]].edge_index = edge_index


# -- Helpers --
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
