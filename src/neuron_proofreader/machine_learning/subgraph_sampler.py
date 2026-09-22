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

    Batches are seeded by walking the proposals in spatial (Morton / Z-order)
    order of their midpoints rather than in random order. Consecutive batches
    therefore read image patches from the same neighborhood of the volume,
    which lets the image cache serve most chunk reads; with random order
    nearly every patch is a cold read from cloud storage.
    """

    # Number of proposals inspected past the cursor when looking for a batch
    # seed whose cluster still fits in the batch.
    max_lookahead = 20

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

        # Spatial visiting order
        self.ordered_proposals = self.sort_spatially(self.proposals)
        self.cursor = 0

        # Identify clustered proposals
        self.set_proposal_clusters()

    def sort_spatially(self, proposals):
        """
        Sorts proposals by the Morton code of their midpoint voxel so that
        proposals that are close in the volume are close in the ordering.

        Parameters
        ----------
        proposals : Iterable[Frozenset[int]]
            Proposals to be sorted.

        Returns
        -------
        List[Frozenset[int]]
            Proposals in spatial order.
        """
        proposals = list(proposals)
        if not proposals:
            return proposals
        pairs = np.array([tuple(p) for p in proposals], dtype=int)
        xyz = (self.graph.node_xyz[pairs[:, 0]] + self.graph.node_xyz[pairs[:, 1]]) / 2
        voxels = xyz[:, ::-1] / self.graph.anisotropy[::-1]
        voxels = np.clip(voxels, 0, 2**21 - 1).astype(np.uint64)
        keys = np.zeros(len(proposals), dtype=np.uint64)
        for bit in range(21):
            for axis in range(3):
                keys |= ((voxels[:, axis] >> np.uint64(bit)) & np.uint64(1)) << np.uint64(3 * bit + axis)
        return [proposals[i] for i in np.argsort(keys, kind="stable")]

    def set_proposal_clusters(self, k=2):
        self.clusters = dict()
        visited = set()
        for proposal in self.proposals:
            if proposal not in visited:
                # Get cluster containing proposal
                cluster = self.extract_cluster(proposal)
                visited.update(cluster)

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

    def populate_via_bfs(self, subgraph, root_proposal):
        i, j = root_proposal
        queue = deque([(i, 0), (j, 0)])
        visited = {i, j}
        while queue:
            # Visit node
            i, d_i = queue.popleft()
            subgraph.pg_add_node(i)
            self.add_nbhd(i, subgraph, visited)

            # Visit proposals at node
            cluster = self.node2cluster(i)
            if len(cluster) <= subgraph.proposal_margin():
                for j in list(set().union(*cluster)):
                    if j not in visited:
                        queue.insert(0, (j, 0))
                        visited.add(j)
                self.add_proposals(subgraph, queue, visited, i)

            # Update queue
            for j in subgraph.pg_neighbors(i):
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
                visited.add(j)
                while not self.is_computation_node(curr):
                    nbs = list(self.graph.neighbors(curr))
                    nxt = nbs[0] if nbs[1] == prev else nbs[1]
                    path.append(nxt)
                    prev, curr = curr, int(nxt)
                    visited.add(curr)

                # Store computation edge
                edge_id = frozenset({i, curr})
                subgraph.edge_to_path[edge_id] = np.array(path, dtype=int)
                subgraph.pg_add_edge(i, curr)

    def add_proposals(self, subgraph, queue, visited, i):
        nodes = list(self.graph.node_proposals[i])
        while not subgraph.is_full() and nodes:
            # Visit proposal
            j = nodes.pop(0)
            pair = frozenset({i, j})

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
                    visited.add(j)

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
        is_irreducible = self.graph.degree(i) != 2
        has_proposals = len(self.graph.node_proposals[i]) > 0
        return is_irreducible or has_proposals

    def node2cluster(self, i):
        if self.graph.node_proposals[i]:
            j = util.sample_once(self.graph.node_proposals[i])
            if frozenset({i, j}) in self.clusters:
                return self.clusters[frozenset({i, j})]
        return set()

    def sample_proposal(self, subgraph):
        """
        Picks the next batch seed: the first unvisited proposal at or after
        the cursor in spatial order whose cluster (if any) fits in the
        remaining batch capacity. If none of the next "max_lookahead"
        proposals fit, returns None so the current batch is emitted.

        Parameters
        ----------
        subgraph : ProposalComputationGraph
            Batch currently being built.

        Returns
        -------
        Frozenset[int] or None
            Proposal to seed the BFS from, or None.
        """
        # Skip proposals already consumed by earlier BFS expansions
        ordered = self.ordered_proposals
        while self.cursor < len(ordered) and ordered[self.cursor] not in self.proposals:
            self.cursor += 1

        margin = subgraph.proposal_margin()
        for idx in range(self.cursor, min(self.cursor + self.max_lookahead, len(ordered))):
            proposal = ordered[idx]
            if proposal not in self.proposals:
                continue
            if proposal not in self.clusters or self.cluster_size(proposal) <= margin:
                return proposal
        return None
