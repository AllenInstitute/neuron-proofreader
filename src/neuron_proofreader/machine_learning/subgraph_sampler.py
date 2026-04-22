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
        self.set_proposal_clusters()

    def set_proposal_clusters(self, k=2):
        self.clusters = dict()
        visited = set()
        for proposal in self.proposals:
            if proposal not in visited:
                # Get cluster containing proposal
                cluster = self.extract_cluster(proposal)
                visited = visited.union(cluster)

                # Check whether to cache cluster
                if len(cluster) >= k and len(cluster) < self.max_proposals:
                    self.clusters.update({p: cluster for p in list(cluster)})

    def extract_cluster(self, proposal):
        """
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
        while queue:
            # Visit proposal
            node1, node2 = queue.pop()

            # Update queue
            for i in [node1, node2]:
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
            while not subgraph.is_full() and self.proposals:
                root = self.sample_proposal(subgraph)
                if root:
                    self.populate_via_bfs(subgraph, root)
                else:
                    break

            # Yield batch
            yield subgraph

    def populate_via_bfs(self, subgraph, root):
        i, j = tuple(root)
        queue = deque([(i, 0), (j, 0)])
        visited = {i, j}
        while queue:
            # Visit node
            i, d_i = queue.popleft()
            subgraph.add_node(i)
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

    def add_proposals(self, subgraph, queue, visited, i):
        nodes = list(self.graph.node_proposals[i])
        while subgraph.is_full() and nodes:
            # Visit proposal
            j = nodes.pop(0)
            pair = frozenset({i, j})

            # Check if pair is proposal part of a cluster
            if pair in self.clusters:
                if self.cluster_size(pair) <= subgraph.proposal_margin():
                    for i in list(set().union(*self.clusters[pair])):
                        queue.insert(0, (i, 0))
                else:
                    continue

            # Add proposal to graph
            if pair in self.proposals:
                # Add proposal to subgraph
                subgraph.proposals.add(pair)
                if pair in self.graph.gt_accepts:
                    subgraph.gt_accepts.add(pair)

                # Update instance state
                self.clusters.pop(pair, None)
                self.proposals.remove(pair)
                if j not in visited:
                    queue.append((j, 0))

    # --- Helpers ---
    def cluster_size(self, proposal):
        return len(self.clusters[proposal])

    def init_subgraph(self):
        """
        Instantiates an empty instance of a ProposalComputationGraph.

        Returns
        -------
        subgraph : ProposalComputationGraph
            Empty graph.
        """
        return ProposalComputationGraph(max_proposals=self.max_proposals)

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

    def sample_proposal(self, subgraph):
        if self.clusters:
            cnt = 0
            while cnt < 10:
                cnt += 1
                proposal = util.sample_once(self.clusters.keys())
                if self.cluster_size(proposal) < subgraph.proposal_margin():
                    return proposal
            return None 
        else:
            return util.sample_once(self.proposals)
