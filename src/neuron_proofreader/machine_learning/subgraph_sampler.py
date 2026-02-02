"""
Created on Sat November 04 15:30:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Implements a class that extracts subgraphs from a graph in order to create
batches suitable for GNN input.

"""

from collections import deque

from neuron_proofreader.proposal_graph import ProposalGraph
from neuron_proofreader.utils import util


class SubgraphSampler:
    """
    A class that extracts subgraphs from a graph in order to create batches
    suitable for GNN input.
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
        subgraph : ProposalGraph
            Sampled subgraph with a bounded number of proposals.
        """
        while self.proposals:
            # Run BFS
            subgraph = self.init_subgraph()
            while not self.is_subgraph_full(subgraph) and self.proposals:
                root = util.sample_once(self.proposals)
                self.populate_via_bfs(subgraph, root)

            # Yield batch
            self.populate_attributes(subgraph)
            print("Batch Size:", subgraph.n_proposals())
            yield subgraph

    def populate_via_bfs(self, subgraph, root):
        i, j = tuple(root)
        queue = deque([(i, 0), (j, 0)])
        visited = set([i, j])
        while queue:
            # Visit node
            i, d_i = queue.popleft()
            self.add_nbhd(subgraph, i)
            self.add_proposals(subgraph, queue, visited, i)

            # Update queue
            for j in self.graph.neighbors(i):
                if j not in visited:
                    n_j = len(self.graph.node_proposals[j])
                    d_j = min(d_i + 1, -n_j)
                    if d_j <= self.gnn_depth:
                        queue.append((j, d_j))
                        visited.add(j)

    def add_nbhd(self, subgraph, i):
        """
        Adds the neighborhood of node "i" to the given sugraph.

        Parameters
        ----------
        subgraph : ProposalGraph
            Graph to be updated.
        i : int
            Node id.
        """
        for j in self.graph.neighbors(i):
            subgraph.add_edge(i, j)

    def add_proposals(self, subgraph, queue, visited, i):
        if subgraph.n_proposals() < self.max_proposals:
            for j in self.graph.node_proposals[i]:
                # Visit proposal
                pair = frozenset({i, j})
                if pair in self.proposals:
                    # Add proposal to subgraph
                    subgraph.add_proposal(i, j)
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
        for p in self.extract_cluster(proposal):
            # Add proposal
            u, v = tuple(p)
            subgraph.add_edge(u, v)
            subgraph.add_proposal(p)

            # Update queue
            if not (u in visited and u in nodes_added):
                queue.append((u, 0))
            if not (v in visited and v in nodes_added):
                queue.append((v, 0))

    def populate_attributes(self, subgraph):
        # TO DO
        pass

    # --- Helpers ---
    def init_subgraph(self):
        """
        Instantiates an empty instance of ProposalGraph.

        Returns
        -------
        subgraph : ProposalGraph
            Empty graph.
        """
        subgraph = ProposalGraph(
            anisotropy=self.graph.anisotropy,
            node_spacing=self.graph.node_spacing,
        )
        return subgraph

    def is_subgraph_full(self, subgraph):
        return subgraph.n_proposals() < self.max_proposals


class SeededSubgraphSampler(SubgraphSampler):

    def __init__(self, graph, max_proposals=200, gnn_depth=2):
        # Call parent class
        super(SeededSubgraphSampler, self).__init__(
            graph, max_proposals, gnn_depth
        )

    # --- Batch Generation ---
    def sample(self):
        soma_connected_proposals_exist = True
        while soma_connected_proposals_exist:
            # Run BFS
            subgraph = self.init_subgraph()
            while not self.is_subgraph_full(subgraph) and self.proposals:
                root = self.find_bfs_root()
                if root:
                    self.populate_via_seeded_bfs(subgraph, root)
                else:
                    soma_connected_proposals_exist = False
                    break

            # Yield batch
            if subgraph.n_proposals():
                yield subgraph

        # Call parent class dataloader
        for batch in super().__iter__():
            yield batch

    def find_bfs_root(self):
        for proposal in self.proposals:
            i, j = tuple(proposal)
            if self.graph.is_soma(i):
                return i
            elif self.graph.is_soma(j):
                return j
        return False

    def populate_via_seeded_bfs(self, subgraph, root):
        queue = self.init_seeded_queue(root)
        visited = set({root})
        while queue:
            # Visit node
            i, d_i = queue.popleft()
            self.visit_nbhd(subgraph, i)
            self.visit_proposals_seeded(subgraph, queue, visited, i)

            # Update queue
            for j in self.graph.neighbors(i):
                if j not in visited:
                    n_j = len(self.graph.node_proposals[j])
                    d_j = min(d_i + 1, -n_j)
                    if d_j <= self.gnn_depth:
                        queue.append((j, d_j))
                        visited.add(j)

    def init_seeded_queue(self, root):
        seeded_queue = deque([(root, 0)])
        queue = deque([root])
        visited = set({root})
        while queue:
            # Visit node
            i = queue.pop()
            if self.graph.node_proposals[i]:
                seeded_queue.append((i, 0))

            # Update queue
            for j in self.graph.neighbors(i):
                if j not in visited:
                    queue.append(j)
                    visited.add(j)
        return seeded_queue

    def visit_proposals_seeded(self, batch, queue, visited, i):
        if len(batch["proposals"]) < self.max_proposals:
            for j in self.graph.node_proposals[i]:
                # Visit proposal
                proposal = frozenset({i, j})
                if proposal in self.proposals:
                    batch["graph"].add_edge(i, j)
                    batch["proposals"].add(proposal)
                    self.proposals.remove(proposal)
                    if j not in visited:
                        queue.append((j, 0))

                # Check if proposal is connected to soma
                if self.graph.is_soma(i) or self.graph.is_soma(j):
                    batch["soma_proposals"].add(proposal)
