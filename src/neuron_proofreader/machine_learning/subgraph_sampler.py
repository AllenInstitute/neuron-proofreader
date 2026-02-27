"""
Created on Sat November 04 15:30:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Implements a class that extracts subgraphs from a graph in order to create
batches suitable for GNN input.

"""

from collections import deque

import numpy as np

from neuron_proofreader.proposal_graph import ProposalComputationGraph
from neuron_proofreader.utils import util


class SubgraphSampler:
    """
    A class that extracts ProposalComputationGraphs from a ProposalGraphs in
    order to create batches suitable for GNN input.
    """

    def __init__(self, graph, gnn_depth=2, max_proposals=64):
        """
        Instantiates a SubgraphSampler object.

        Parameters
        ----------
        graph : ProposalGraph
            Graph to be sampled from.
        gnn_depth : int, optional
            Depth of graph neural network. Default is 2.
        max_proposals : int, optional
            Maximum number of proposals in subgraph. Default is 64.
        """
        # Instance attributes
        self.max_proposals = max_proposals
        self.gnn_depth = gnn_depth
        self.graph = graph
        self.proposals = set(graph.list_proposals())

        # Identify clustered proposals
        self.flagged = set()  # self.find_proposal_clusters(5)

    def find_proposal_clusters(self, k):
        flagged = set()
        visited = set()
        for proposal in self.proposals:
            if proposal not in visited:
                cluster = self.extract_cluster(proposal)
                if len(cluster) >= k:
                    flagged = flagged.union(cluster)
                visited = visited.union(cluster)
        return flagged

    def extract_cluster(self, proposal):
        """
        -- NOT FUNCTIONAL ---

        Extracts the connected component that "proposal" belongs to in the
        proposal induced subgraph.

        Parameters
        ----------
        proposal : Frozenset[int]
            Proposal used as the root to extract its connected component in
            the proposal induced subgraph.

        Returns
        -------
        visited : Set[Frozenset[int]]
            Connected component that "proposal" belongs to in the proposal
            induced subgraph.
        """
        queue = deque([proposal])
        visited = set(queue)
        while len(queue) > 0:
            # Visit proposal
            proposal = queue.pop()

            # Update queue
            for i in proposal:
                for j in self.graph.node_proposals[i]:
                    proposal_ij = frozenset({i, j})
                    if proposal_ij not in visited:
                        queue.append(proposal_ij)
                        visited.add(proposal_ij)
        return visited

    # --- Core Routines---
    def __iter__(self):
        """
        Samples a subgraph by running a BFS using proposals as roots until
        every proposal has been visited.

        Returns
        -------
        subgraph : ProposalComputationGraph
            Sampled subgraph with a bounded number of proposals.
        """
        while self.proposals:
            # Run BFS
            subgraph = self.init_subgraph()
            while not self.is_subgraph_full(subgraph) and self.proposals:
                root = util.sample_once(self.proposals)
                self.populate_via_bfs(subgraph, root)

            # Yield batch
            yield subgraph

    def populate_via_bfs(self, subgraph, root):
        i, j = tuple(root)
        queue = deque([(i, 0), (j, 0)])
        visited = {i, j}
        while queue:
            # Visit node
            i, d_i = queue.popleft()
            self.add_nbhd(i, subgraph, visited)
            self.add_proposals(subgraph, queue, visited, i)

            # Update queue
            for j in subgraph.neighbors(i):
                if j not in visited:
                    n_j = len(self.graph.node_proposals[j])
                    d_j = min(d_i + 1, -n_j)
                    if d_j <= self.gnn_depth:
                        queue.append((j, d_j))
                        visited.add(j)

    def add_nbhd(self, i, subgraph, visited):
        """
        Adds the neighborhood of node "i" to the given sugraph.

        Parameters
        ----------
        i : int
            Node id.
        subgraph : ProposalComputationGraph
            Graph to be updated.
        visited : Set[int]
            Nodes that have already been visited.
        """
        for j in self.graph.neighbors(i):
            if j not in visited:
                # Walk through degree-2 chain
                path = [i, j]
                prev, curr = i, j
                while not self.is_computation_node(curr):
                    nbs = list(self.graph.neighbors(curr))
                    nxt = nbs[0] if nbs[1] == prev else nbs[1]
                    path.append(nxt)
                    prev, curr = curr, int(nxt)
                    visited.add(curr)

                # Store computation edge
                edge_id = frozenset({i, curr})
                subgraph.edge_to_path[edge_id] = np.array(path, dtype=int)
                subgraph.add_edge(i, curr)
                visited.remove(curr)

    def add_proposals(self, subgraph, queue, visited, i):
        if subgraph.n_proposals() < self.max_proposals:
            for j in self.graph.node_proposals[i]:
                # Visit proposal
                pair = frozenset({i, j})
                if pair in self.proposals:
                    # Add proposal to subgraph
                    subgraph.proposals.add(pair)
                    if pair in self.graph.gt_accepts:
                        subgraph.gt_accepts.add(pair)

                    # Update instance state
                    self.proposals.remove(pair)
                    if j not in visited:
                        queue.append((j, 0))

                # Check if proposal is flagged
                # proposal in self.flagged and proposal in self.proposals:
                if False:
                    self.visit_flagged_proposal(subgraph)

    def visit_flagged_proposal(self, subgraph, queue, visited, proposal):
        nodes_added = set()
        for proposal in self.extract_cluster(proposal):
            # Add proposal
            i, j = proposal
            subgraph.add_edge(i, j)
            subgraph.add_proposal(proposal)

            # Update queue
            if not (i in visited and i in nodes_added):
                queue.append((i, 0))
            if not (j in visited and j in nodes_added):
                queue.append((j, 0))

    # --- Helpers ---
    def init_subgraph(self):
        """
        Instantiates an empty instance of ProposalGraph.

        Returns
        -------
        subgraph : ProposalComputationGraph
            Empty graph.
        """
        subgraph = ProposalComputationGraph()
        return subgraph

    def is_computation_node(self, i):
        """
        Checks if the given node is either irreducible or contains at least
        one proposal, hence needs to be a node in the computation graph.

        Parameters
        ----------
        i : int
            Node ID.

        Returns
        -------
        bool
            True if node needs to be contained in the computation graph;
            otherwise, False.
        """
        is_irreducible = self.graph.degree[i] != 2
        has_propsoals = len(self.graph.node_proposals[i]) > 0
        return is_irreducible or has_propsoals

    def is_subgraph_full(self, subgraph):
        return subgraph.n_proposals() >= self.max_proposals
