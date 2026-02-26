"""
Created on Sat July 15 9:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Implementation of a custom subclass of Networkx.Graph called "ProposalGraph".
After initializing an instance of this subclass, the graph is built by reading
and processing SWC files (i.e. neuron fragments). It then stores the relevant
information into the graph structure.

"""

from collections import defaultdict

import networkx as nx
import numpy as np

from neuron_proofreader.skeleton_graph import SkeletonGraph
from neuron_proofreader.split_proofreading import (
    groundtruth_generation
)
from neuron_proofreader.split_proofreading.proposal_generation import (
    ProposalGenerator,
    trim_proposal_endpoints
)
from neuron_proofreader.utils import geometry_util, graph_util


class ProposalGraph(SkeletonGraph):
    """
    Custom subclass of NetworkX.Graph constructed from neuron fragments. The
    graph's nodes are irreducible, meaning each node has either degree 1
    (leaf) or degree 3+ (branching points). Each edge has an attribute that
    stores a dense path of points connecting irreducible nodes. Additionally,
    the graph has an attribute called "proposals", which is a set of potential
    connections between pairs of neuron fragments.
    """

    def __init__(
        self,
        anisotropy=(1.0, 1.0, 1.0),
        gt_path=None,
        max_proposals_per_leaf=3,
        min_size=0,
        min_size_with_proposals=0,
        node_spacing=1,
        prune_depth=20.0,
        remove_high_risk_merges=False,
        segmentation_path=None,
        soma_centroids=None,
        verbose=True,
    ):
        """
        Instantiates a ProposalGraph object.

        Parameters
        ----------
        anisotropy : Tuple[int], optional
            Image to physical coordinates scaling factors to account for the
            anisotropy of the microscope. Default is (1.0, 1.0, 1.0).
        min_size : float, optional
            Minimum path length of fragments loaded into graph. Default is 0.
        min_size_with_proposals : float, optional
            Minimum fragment path length required for proposals. Default is 0.
        node_spacing : float, optional
            Distance between points in edges.
        prune_depth : float, optional
            Branches with length less than "prune_depth" microns are removed.
            Default is 20um.
        remove_high_risk_merges : bool, optional
            Indication of whether to remove high risk merge sites (i.e. close
            branching points). Default is False.
        segmentation_path : str, optional
            Path to segmentation stored in GCS bucket. Default is None.
        soma_centroids : List[Tuple[float]] or None, optional
            Phyiscal coordinates of soma centroids. Default is None.
        verbose : bool, optional
            Indication of whether to display a progress bar while building
            graph. Default is True.
        """
        # Call parent class
        super().__init__()

        # Instance attributes - Graph
        self.anisotropy = anisotropy
        self.component_id_to_swc_id = dict()
        self.gt_path = gt_path
        self.soma_ids = set()
        self.verbose = verbose

        # Instance attributes - Proposals
        self.accepts = set()
        self.gt_accepts = set()
        self.merged_ids = set()
        self.n_merges_blocked = 0
        self.n_proposals_blocked = 0

        self.reset_proposals()
        self.proposal_generator = ProposalGenerator(
            self,
            max_proposals_per_leaf=max_proposals_per_leaf,
            min_size_with_proposals=min_size_with_proposals
        )

        # Graph Loader
        self.graph_loader = graph_util.GraphLoader(
            anisotropy=anisotropy,
            min_size=min_size,
            node_spacing=node_spacing,
            prune_depth=prune_depth,
            remove_high_risk_merges=remove_high_risk_merges,
            segmentation_path=segmentation_path,
            soma_centroids=soma_centroids,
            verbose=verbose,
        )

    # --- Update Structure ---
    def connect_soma_fragments(self, soma_centroids):
        # MUST BE UPDATED!
        merge_cnt = 0
        soma_cnt = 0
        self.set_kdtree()
        for soma_xyz in soma_centroids:
            node_ids = self.find_fragments_near_xyz(soma_xyz, 25)
            if len(node_ids) > 1:
                # Find closest node to soma location
                soma_cnt += 1
                best_dist = np.inf
                best_node = None
                for i in node_ids:
                    dist = geometry_util.dist(soma_xyz, self.node_xyz[i])
                    if dist < best_dist:
                        best_dist = dist
                        best_node = i
                soma_component_id = self.node_component_id[best_node]
                self.soma_ids.add(soma_component_id)
                node_ids.remove(best_node)

                # Merge fragments to soma
                soma_xyz = self.node_xyz[best_node]
                for i in node_ids:
                    attrs = {
                        "radius": np.array([2, 2]),
                        "xyz": np.array([soma_xyz, self.node_xyz[i]]),
                    }
                    self._add_edge((best_node, i), attrs)
                    self.update_component_ids(soma_component_id, i)
                    merge_cnt += 1

        # Summarize results
        results = [f"# Somas Connected: {soma_cnt}"]
        results.append(f"# Soma Fragments Merged: {merge_cnt}")
        return "\n".join(results)

    def relabel_nodes(self):
        """
        Reassigns contiguous node IDs and update all dependent structures.
        """
        # Call parent class
        old_proposals = self.list_proposals()
        old_to_new = super().relabel_nodes()

        # Update proposals
        self.reset_proposals()
        for proposal in old_proposals:
            i, j = proposal
            self.add_proposal(int(old_to_new[i]), int(old_to_new[j]))

    # --- Proposal Operations ---
    def add_proposal(self, i, j):
        """
        Adds proposal between nodes "i" and "j".

        Parameters
        ----------
        i : int
            Node ID.
        j : int
            Node ID
        """
        proposal = frozenset({i, j})
        self.node_proposals[i].add(j)
        self.node_proposals[j].add(i)
        self.proposals.add(proposal)

    def generate_proposals(self, search_radius, allow_nonleaf_targets=False):
        """
        Generates proposals from leaf nodes.

        Parameters
        ----------
        search_radius : float
            Search radius used to generate proposals.
        allow_nonleaf_targets : bool, optional
            Indication of whether to generate proposals between leaf and nodes
            with degree 2. Default is False.
        """
        # Proposal generation
        proposals = self.proposal_generator(
            search_radius, allow_nonleaf_targets=allow_nonleaf_targets
        )

        self.search_radius = search_radius
        self.store_proposals(proposals)
        self.trim_proposals()
        self.relabel_nodes()

        # Set groundtruth (if applicable)
        if self.gt_path:
            gt_graph = SkeletonGraph(anisotropy=self.anisotropy)
            gt_graph.load(self.gt_path)
            self.gt_accepts = groundtruth_generation.run(gt_graph, self)

    def is_mergeable(self, i, j):
        one_leaf = self.degree[i] == 1 or self.degree[j] == 1
        branching = self.degree[i] > 2 or self.degree[j] > 2
        somas_check = not (self.is_soma(i) and self.is_soma(j))
        return somas_check and (one_leaf and not branching)

    def is_single_proposal(self, proposal):
        """
        Checks if "proposal" is the only proposal generated for the
        corresponding nodes.

        Parameters
        ----------
        proposal : Frozenset[int]
            Pair of node IDs corresponding to a proposal.

        Returns
        -------
        bool
            True if "proposal" is the only proposal generated for the
            corresponding nodes; otherwise, False
        """
        i, j = tuple(proposal)
        single_i = len(self.node_proposals[i]) == 1
        single_j = len(self.node_proposals[j]) == 1
        return single_i and single_j

    def leaf_to_leaf(self, proposal):
        """
        Checks if both nodes in a proposal are leafs.

        Parameters
        ----------
        proposal : Frozenset[int]
            Pair of nodes that form a proposal.

        Returns
        -------
        bool
            True if both nodes in a proposal are leafs; otherwise, False.
        """
        i, j = tuple(proposal)
        return self.degree[i] == 1 and self.degree[j] == 1

    def list_proposals(self):
        """
        Lists proposals in self.

        Returns
        -------
        List[Frozenset[int]]
            Proposals.
        """
        return list(self.proposals)

    def merge_proposal(self, proposal):
        i, j = tuple(proposal)
        if self.is_mergeable(i, j):
            # Update attributes
            attrs = {
                "radius": self.node_radius[np.array([i, j], dtype=int)],
                "xyz": self.node_xyz[np.array([i, j], dtype=int)]
            }

            # Update component_ids
            self.merged_ids.add((self.get_swc_id(i), self.get_swc_id(j)))
            if self.is_soma(i):
                component_id = self.node_component_id[i]
                self.update_component_ids(component_id, j)
            else:
                component_id = self.node_component_id[j]
                self.update_component_ids(component_id, i)

            # Update graph
            self._add_edge((i, j), attrs)
            self.accepts.add(proposal)
            self.proposals.remove(proposal)
        else:
            self.n_merges_blocked += 1

    def n_proposals(self):
        """
        Counts the number of proposals in the graph.

        Returns
        -------
        int
            Number of proposals in the graph.
        """
        return len(self.proposals)

    def remove_proposal(self, proposal):
        """
        Removes an existing proposal between two nodes.

        Parameters
        ----------
        proposal : Frozenset[int]
            Pair of node IDs corresponding to a proposal.
        """
        i, j = tuple(proposal)
        self.node_proposals[i].remove(j)
        self.node_proposals[j].remove(i)
        self.proposals.remove(proposal)

    def reset_proposals(self):
        self.node_proposals = defaultdict(set)
        self.proposals = set()

    def sorted_proposals(self):
        """
        Returns proposals sorted by physical length.

        Returns
        -------
        List[Frozenset[int]]
            Proposals sorted by physical length.
        """
        proposals = self.list_proposals()
        lengths = [self.proposal_length(p) for p in proposals]
        return [proposals[i] for i in np.argsort(lengths)]

    def store_proposals(self, proposals):
        self.node_proposals = defaultdict(set)
        for proposal in proposals:
            i, j = tuple(proposal)
            self.add_proposal(i, j)

    def trim_proposals(self):
        for proposal in self.list_proposals():
            leaf_to_leaf = self.leaf_to_leaf(proposal)
            is_single = self.is_single_proposal(proposal)
            if leaf_to_leaf and is_single:
                trim_proposal_endpoints(self, proposal)

    # --- Proposal Feature Generation ---
    def proposal_directionals(self, proposal, depth):
        # Extract points along branches
        i, j = tuple(proposal)
        path_i = self.path_thru_node(i, depth)
        path_j = self.path_thru_node(j, depth)
        path_xyz_i = self.node_xyz[np.array(path_i)]
        path_xyz_j = self.node_xyz[np.array(path_j)]

        # Compute tangent vectors - branches
        dir_i = geometry_util.tangent(path_xyz_i)
        dir_j = geometry_util.tangent(path_xyz_j)
        dir_proposal = geometry_util.tangent(self.proposal_xyz(proposal))

        # Compute features
        dot_i = abs(np.dot(dir_proposal, dir_i))
        dot_j = abs(np.dot(dir_proposal, dir_j))
        if self.leaf_to_leaf(proposal):
            dot_ij = np.dot(dir_i, dir_j)
        else:
            dot_ij = np.dot(dir_i, dir_j)
            if not self.leaf_to_leaf(proposal):
                dot_ij = max(dot_ij, -dot_ij)
        return np.array([dot_i, dot_j, dot_ij])

    def proposal_length(self, proposal):
        return self.dist(*tuple(proposal))

    def proposal_midpoint(self, proposal):
        return geometry_util.midpoint(*self.proposal_xyz(proposal))

    def proposal_radius(self, proposal):
        i, j = tuple(proposal)
        return self.node_radius[i], self.node_radius[j]

    def proposal_xyz(self, proposal):
        i, j = tuple(proposal)
        return self.node_xyz[i], self.node_xyz[j]

    # --- Helpers ---
    def computation_graph(self):
        # Add nodes
        graph = ProposalComputationGraph()
        graph.add_nodes_from(self.irreducible_nodes())
        graph.add_nodes_from(set().union(*self.proposals))
        graph.proposals = self.proposals
        graph.gt_accepts = self.gt_accepts

        # Extract edges
        visited = set()
        for i in map(int, self.irreducible_nodes()):
            for j in map(int, self.neighbors(i)):
                # Check if already visited
                if frozenset({i, j}) in visited:
                    continue

                # Walk through degree-2 chain
                path = [i, j]
                prev, curr = i, j
                while curr not in graph.nodes:
                    nbs = list(self.neighbors(curr))
                    nxt = nbs[0] if nbs[1] == prev else nbs[1]
                    path.append(nxt)
                    prev, curr = curr, int(nxt)
                edge_id = frozenset({i, curr})
                graph.edge_to_path[edge_id] = np.array(path, dtype=int)
                assert i in graph.nodes and curr in graph.nodes
                graph.add_edge(i, curr)

                # Mark edges as visited
                for a, b in zip(path[:-1], path[1:]):
                    visited.add(frozenset({a, b}))
        return graph

    def find_fragments_near_xyz(self, query_xyz, max_dist):
        hits = dict()
        xyz_list = geometry_util.query_ball(self.kdtree, query_xyz, max_dist)
        for xyz in xyz_list:
            i, j = self.xyz_to_edge[tuple(xyz)]
            dist_i = geometry_util.dist(self.node_xyz[i], query_xyz)
            dist_j = geometry_util.dist(self.node_xyz[j], query_xyz)
            hits[self.node_component_id[i]] = i if dist_i < dist_j else j
        return list(hits.values())

    def is_soma(self, i):
        """
        Check whether a node belongs to a component containing a soma.

        Parameters
        ----------
        i : str
            Node ID.

        Returns
        -------
        bool
            True if the node belongs to a connected component with a soma;
            False otherwise.
        """
        return self.node_component_id[i] in self.soma_ids

    def update_component_ids(self, component_id, root):
        """
        Updates the component_id of all nodes connected to "root".

        Parameters
        ----------
        component_id : str
            Connected component id.
        root : int
            Node ID
        """
        queue = [root]
        visited = set(queue)
        while len(queue) > 0:
            i = queue.pop()
            self.node_component_id[i] = component_id
            visited.add(i)
            for j in [j for j in self.neighbors(i) if j not in visited]:
                queue.append(j)


# --- Computation Graph ---
class ProposalComputationGraph(nx.Graph):

    def __init__(self):
        # Call parent class
        super().__init__()

        # Instance attributes
        self.edge_to_path = dict()
        self.gt_accepts = set()
        self.proposals = set()
