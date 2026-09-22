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

import numpy as np
import rustworkx as rx

from neuron_proofreader.fragments_graph import FragmentsGraph
from neuron_proofreader.split_proofreading import groundtruth_generation
from neuron_proofreader.split_proofreading.proposal_generation import (
    ProposalGenerator,
    trim_proposal_endpoints,
)
from neuron_proofreader.utils import geometry_util


class ProposalGraph(FragmentsGraph):
    """
    Custom subclass of FragmentsGraph constructed from neuron fragments. This
    graph has an attribute called "proposals", which is a set of potential
    connections between pairs of neuron fragments. This class has subroutines
    for generating, processing, and operating on proposals.
    """

    def __init__(
        self,
        anisotropy=(1.0, 1.0, 1.0),
        gt_path=None,
        min_cable_length=0,
        node_spacing=1,
        prune_depth=20.0,
        verbose=True,
    ):
        """
        Instantiates a ProposalGraph object.

        Parameters
        ----------
        anisotropy : Tuple[int], optional
            Image to physical coordinates scaling factors to account for the
            anisotropy of the microscope. Default is (1.0, 1.0, 1.0).
        min_cable_length : float, optional
            Minimum cable length of fragments loaded into graph. Default is 0.
        node_spacing : float, optional
            Distance (in microns) between nodes. Default is 1.
        prune_depth : float, optional
            Branches with length less than "prune_depth" microns are removed.
            Default is 20um.
        verbose : bool, optional
            Indication of whether to display a progress bar while building
            graph. Default is True.
        """
        # Call parent class
        super().__init__(
            anisotropy=anisotropy,
            min_cable_length=min_cable_length,
            node_spacing=node_spacing,
            prune_depth=prune_depth,
            verbose=verbose,
        )

        # Instance attributes - Proposals
        self.accepts = set()
        self.gt_accepts = set()
        self.gt_path = gt_path
        self.merged_ids = set()
        self.n_merges_blocked = 0
        self.n_proposals_blocked = 0
        self.reset_proposals()

    # --- Update Structure ---
    def relabel_nodes(self):
        """
        Reassigns contiguous node IDs and update all dependent structures.
        """
        # Call parent class
        old_proposals = self.list_proposals()
        old_to_new = super().relabel_nodes()

        # Update proposals, dropping any whose endpoint was removed
        self.reset_proposals()
        for i, j in old_proposals:
            if i in old_to_new and j in old_to_new:
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
        # has_node is O(1); "i in self.node_indices()" is a linear scan
        assert self.has_node(i) and self.has_node(j)
        self.node_proposals[i].add(j)
        self.node_proposals[j].add(i)
        self.proposals.add(frozenset({i, j}))

    def generate_proposals(
        self,
        search_radius,
        allow_nonleaf_proposals=False,
        max_proposals_per_leaf=3,
        min_size_with_proposals=0,
    ):
        """
        Generates proposals from leaf nodes.

        Parameters
        ----------
        search_radius : float
            Search radius used to generate proposals.
        allow_nonleaf_proposals : bool, optional
            Indication of whether to generate proposals between leaf and nodes
            with degree 2. Default is False.
        min_size_with_proposals : float
            Minimum cable length (in microns) of connected components that
            proposals are generated from. Default is 0.
        """
        # Proposal generation
        if self.num_nodes() == 0:
            return
        assert len(self.kdtree.data) == self.num_nodes()
        proposal_generator = ProposalGenerator(
            self,
            allow_nonleaf_proposals=allow_nonleaf_proposals,
            max_proposals_per_leaf=max_proposals_per_leaf,
            min_size_with_proposals=min_size_with_proposals,
        )
        proposals = proposal_generator(search_radius)

        self.search_radius = search_radius
        self.store_proposals(proposals)
        self.trim_proposals()

        # Set groundtruth (if applicable)
        if self.gt_path:
            gt_graph = FragmentsGraph(anisotropy=self.anisotropy)
            gt_graph.load(self.gt_path)
            self.gt_accepts = groundtruth_generation.run(gt_graph, self)

    def keep_fragments(self, swc_ids):
        """
        Removes every fragment whose SWC ID is not in the given set. Must be
        called before proposals are generated.

        Parameters
        ----------
        swc_ids : Set[str]
            SWC IDs of fragments to keep.
        """
        rm_component_ids = [
            component_id
            for component_id, swc_id in self.component_id_to_swc_id.items()
            if (swc_id if "." in swc_id else f"{swc_id}.0") not in swc_ids
        ]
        if rm_component_ids:
            rm_nodes = np.where(
                np.isin(self.node_component_id, rm_component_ids)
            )[0]
            self.remove_nodes(rm_nodes.tolist())

    def set_gt_accepts(self, swc_id_pairs):
        """
        Sets accepted proposals by matching each proposal's endpoint SWC IDs
        against the given pairs. Matching on SWC IDs rather than node IDs
        lets ground truth computed at one node spacing be applied to a graph
        built at another.

        Parameters
        ----------
        swc_id_pairs : Set[Frozenset[str]]
            Unordered pairs of SWC IDs whose proposal is accepted.
        """
        self.gt_accepts = set()
        for proposal in self.proposals:
            i, j = tuple(proposal)
            pair = frozenset({self.node_swc_id(i), self.node_swc_id(j)})
            if pair in swc_id_pairs:
                self.gt_accepts.add(proposal)

    def is_mergeable(self, i, j):
        one_leaf = self.degree(i) == 1 or self.degree(j) == 1
        not_branching = self.degree(i) < 3 and self.degree(j) < 3
        both_somas = self.is_soma(i) and self.is_soma(j)
        return not both_somas and (one_leaf and not_branching)

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
        i, j = proposal
        single_i = len(self.node_proposals[i]) == 1
        single_j = len(self.node_proposals[j]) == 1
        return single_i and single_j

    def is_leaf2leaf(self, proposal):
        """
        Checks if both proposal nodes are leafs.

        Parameters
        ----------
        proposal : Frozenset[int]
            Pair of nodes that form a proposal.

        Returns
        -------
        bool
            True if both nodes in a proposal are leafs; otherwise, False.
        """
        i, j = proposal
        return self.degree(i) == 1 and self.degree(j) == 1

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
        i, j = proposal
        if self.is_mergeable(i, j):
            # Update component_ids
            self.merged_ids.add((self.node_swc_id(i), self.node_swc_id(j)))
            if self.is_soma(i):
                component_id = self.node_component_id[i]
                self.update_component_ids(component_id, j)
            else:
                component_id = self.node_component_id[j]
                self.update_component_ids(component_id, i)

            # Update graph
            self.add_edge(i, j, None)
            self.accepts.add(proposal)
            self.remove_proposal(proposal)
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
        i, j = proposal
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
            i, j = proposal
            self.add_proposal(i, j)

    def trim_proposals(self):
        for proposal in self.list_proposals():
            is_leaf2leaf = self.is_leaf2leaf(proposal)
            is_single = self.is_single_proposal(proposal)
            if is_leaf2leaf and is_single:
                trim_proposal_endpoints(self, proposal)
        self.relabel_nodes()

    # --- Proposal Feature Generation ---
    def proposal_directionals(self, proposal, depth):
        # Extract points along branches
        i, j = proposal
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
        dot_ij = np.dot(dir_i, dir_j)
        if not self.is_leaf2leaf(proposal):
            dot_ij = abs(dot_ij)
        return np.array([dot_i, dot_j, dot_ij])

    def proposal_length(self, proposal):
        """
        Length of proposed edge.
        """
        return self.dist(*tuple(proposal))

    def proposal_midpoint(self, proposal):
        return self.midpoint(*proposal)

    def proposal_radius(self, proposal):
        i, j = proposal
        return self.node_feats["radius"][i], self.node_feats["radius"][j]

    def proposal_xyz(self, proposal):
        i, j = proposal
        return self.node_xyz[i], self.node_xyz[j]

    # --- Helpers ---
    def computation_graph(self):
        def is_computation_node(i):
            return self.degree(i) != 2 or len(self.node_proposals[i]) > 0

        # Add nodes
        graph = ProposalComputationGraph()
        nodes = self.irreducible_nodes()
        nodes = nodes.union(set().union(*self.proposals))
        graph.proposals = self.proposals
        graph.gt_accepts = self.gt_accepts

        # Extract edges
        visited = set()
        for i in map(int, nodes):
            for j in map(int, self.neighbors(i)):
                # Check if already visited
                if frozenset({i, j}) in visited:
                    continue

                # Walk through degree-2 chain
                path = [i, j]
                prev, curr = i, j
                while not is_computation_node(curr):
                    nbs = list(self.neighbors(curr))
                    nxt = nbs[0] if nbs[1] == prev else nbs[1]
                    path.append(nxt)
                    prev, curr = curr, int(nxt)

                # Add computation edge
                edge_id = frozenset({i, curr})
                graph.edge_to_path[edge_id] = np.array(path, dtype=int)
                graph.pg_add_edge(i, curr)

                # Mark edges as visited
                for a, b in zip(path[:-1], path[1:]):
                    visited.add(frozenset({a, b}))
        return graph

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
        return self.node_component_id[i] in self.soma_component_ids

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
        nodes = list(rx.node_connected_component(self, root))
        self.node_component_id[nodes] = component_id


# --- Computation Graph ---
class ProposalComputationGraph(rx.PyGraph):

    def __new__(cls, *args, **kwargs):
        # multigraph=False keeps repeated add_edge(u, v) idempotent, matching
        # networkx.Graph semantics (parallel chains between the same pair of
        # computation nodes would otherwise create duplicate edges).
        return super().__new__(cls, multigraph=False)

    def __init__(self, max_proposals=64):
        # Call parent class
        super().__init__()

        # Instance attributes
        self._pg_to_comp = {}  # ProposalGraph node ID → comp graph index
        self.edge_to_path = dict()
        self.gt_accepts = set()
        self.max_proposals = max_proposals
        self.proposals = set()

    # --- ProposalGraph-ID-aware helpers ---
    def pg_add_node(self, pg_id):
        """Add a node keyed by its ProposalGraph ID (no-op if present)."""
        if pg_id not in self._pg_to_comp:
            comp_idx = self.add_node(pg_id)
            self._pg_to_comp[pg_id] = comp_idx

    def pg_add_edge(self, pg_i, pg_j, weight=None):
        """Add an edge between ProposalGraph nodes, creating them if needed."""
        self.pg_add_node(pg_i)
        self.pg_add_node(pg_j)
        self.add_edge(self._pg_to_comp[pg_i], self._pg_to_comp[pg_j], weight)

    def pg_neighbors(self, pg_id):
        """Return ProposalGraph IDs of the neighbors of node pg_id."""
        comp_idx = self._pg_to_comp[pg_id]
        return [self[n] for n in self.neighbors(comp_idx)]

    @property
    def pg_nodes(self):
        """Return all ProposalGraph node IDs present in this graph."""
        return list(self._pg_to_comp.keys())

    @property
    def pg_edges(self):
        """Return edges as frozensets of ProposalGraph node IDs."""
        return [frozenset({self[i], self[j]}) for i, j in self.edge_list()]

    def is_full(self):
        return self.n_proposals() >= self.max_proposals

    def n_proposals(self):
        return len(self.proposals)

    def proposal_margin(self):
        return self.max_proposals - self.n_proposals()
