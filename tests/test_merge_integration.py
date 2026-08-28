"""Integration test for proposal generation and merging.

SkeletonGraph is CSR-backed rather than nx-backed, and this is the one workload
that mutates the graph an edge at a time: merge_proposal calls add_edge and then
update_component_ids immediately BFSes the merged component through neighbors().
That interleaving is what a static CSR format is bad at, so it gets an
end-to-end test rather than only unit coverage of the primitives.

Self-contained by construction: the fragmented graph is built here rather than
loaded, so the test has no data dependency. The full-scale check against the
recorded nx baseline (20 patchseq cells, 3,662 proposals) lives in the capsule
at scratch/refactor_verification/v1/, because it needs the real cohort.
"""

import unittest

import numpy as np

from neuron_proofreader.fragments_graph import FragmentsGraph
from neuron_proofreader.proposal_graph import ProposalGraph
from neuron_proofreader.split_proofreading.proposal_generation import (
    ProposalGenerator,
)

SEED = 0
SEARCH_RADIUS = 25
N_FRAGMENTS = 24
NODES_PER_FRAGMENT = 60


def build_fragmented_graph(cls=ProposalGraph):
    """
    Builds a graph of straight fragments laid out so that their endpoints fall
    inside the search radius of one another, which is what gives the generator
    something to propose.
    """
    rng = np.random.default_rng(SEED)
    edges, xyz, offset = list(), list(), 0
    for k in range(N_FRAGMENTS):
        idxs = np.arange(NODES_PER_FRAGMENT) + offset
        edges.append(np.stack([idxs[:-1], idxs[1:]], axis=1))

        # Fragments run along x in a grid of rows, with a gap between
        # consecutive fragments small enough to be bridgeable.
        origin = np.array([(k % 6) * 70.0, (k // 6) * 40.0, 0.0])
        step = np.array([1.0, 0.0, 0.0])
        jitter = rng.normal(scale=0.3, size=(NODES_PER_FRAGMENT, 3))
        xyz.append(origin + np.arange(NODES_PER_FRAGMENT)[:, None] * step
                   + jitter)
        offset += NODES_PER_FRAGMENT

    graph = cls(verbose=False)
    graph.add_nodes_from(range(offset))
    graph.add_edges_from(map(tuple, np.concatenate(edges)))

    graph.init_node_attrs(offset)
    graph.node_xyz[:] = np.concatenate(xyz)
    graph.node_radius[:] = 1.0
    graph.node_type[:] = 3
    component_id = np.repeat(
        np.arange(1, N_FRAGMENTS + 1), NODES_PER_FRAGMENT
    )
    graph.node_component_id[:] = component_id
    for cid in range(1, N_FRAGMENTS + 1):
        graph.component_id_to_swc_id[cid] = f"frag-{cid}"
    graph.set_kdtree()
    graph.soma_centroids = list()
    graph.soma_component_ids = list()
    return graph


class ProposalMergeTest(unittest.TestCase):
    """Runs the real generator and accept loop over a fragmented graph."""

    def setUp(self):
        self.graph = build_fragmented_graph()
        self.assertEqual(
            N_FRAGMENTS, self.graph.number_connected_components()
        )

    def test_accept_loop_is_consistent(self):
        proposals = ProposalGenerator(self.graph)(SEARCH_RADIUS)
        self.assertGreater(len(proposals), 0, "no proposals to merge")
        self.graph.store_proposals(proposals)

        edges_before = self.graph.number_of_edges()
        components_before = self.graph.number_connected_components()
        rebuilds_before = self.graph.rebuild_count

        for proposal in sorted(tuple(sorted(p)) for p in proposals):
            self.graph.merge_proposal(frozenset(proposal))

        n_accepts = len(self.graph.accepts)
        self.assertGreater(n_accepts, 0, "every proposal was blocked")
        self.assertEqual(
            len(proposals), n_accepts + self.graph.n_merges_blocked
        )

        # Each accept adds exactly one edge. Components merely do not increase:
        # is_mergeable screens on degree and soma, not on connectivity, so an
        # accept can close a cycle instead of joining two components.
        self.assertEqual(
            edges_before + n_accepts, self.graph.number_of_edges()
        )
        self.assertLessEqual(
            self.graph.number_connected_components(), components_before
        )

        # The overlay must absorb every accept without a rebuild, otherwise
        # each one costs an O(E) materialization.
        self.assertEqual(rebuilds_before, self.graph.rebuild_count)

    def test_has_path_guarded_loop_stays_a_forest(self):
        """
        Mirrors split_inference.py:276, which guards merge_proposal with
        has_path. That reads connectivity from a graph mutated on the previous
        iteration, so a stale component label shows up here as a cycle.
        """
        proposals = ProposalGenerator(self.graph)(SEARCH_RADIUS)
        self.graph.store_proposals(proposals)

        n_accepts = 0
        for proposal in sorted(tuple(sorted(p)) for p in proposals):
            i, j = proposal
            if not self.graph.has_path(i, j):
                before = len(self.graph.accepts)
                self.graph.merge_proposal(frozenset(proposal))
                n_accepts += len(self.graph.accepts) - before

        self.assertGreater(n_accepts, 0)
        num_components = self.graph.number_connected_components()
        self.assertEqual(N_FRAGMENTS - n_accepts, num_components)
        self.assertEqual(
            self.graph.number_of_nodes() - num_components,
            self.graph.number_of_edges(),
        )

    def test_component_ids_track_the_merges(self):
        proposals = ProposalGenerator(self.graph)(SEARCH_RADIUS)
        self.graph.store_proposals(proposals)
        for proposal in sorted(tuple(sorted(p)) for p in proposals):
            self.graph.merge_proposal(frozenset(proposal))

        for nodes in self.graph.connected_components():
            ids = self.graph.node_component_id[list(nodes)]
            self.assertEqual(1, len(set(ids.tolist())))


class ConversionTest(unittest.TestCase):
    """
    from_fragments_graph and to_fragments_graph used nx's dict "update" to copy
    the structure, which has no CSR equivalent.
    """

    def setUp(self):
        self.source = build_fragmented_graph(cls=FragmentsGraph)
        self.source.soma_centroids = list()
        self.source.soma_component_ids = list()
        self.source.graph_loader = None

    def assert_same_structure(self, expected, actual):
        self.assertEqual(
            expected.number_of_nodes(), actual.number_of_nodes()
        )
        self.assertEqual(
            expected.number_of_edges(), actual.number_of_edges()
        )
        self.assertEqual(
            {frozenset(e) for e in expected.edges},
            {frozenset(e) for e in actual.edges},
        )
        for i in range(expected.number_of_nodes()):
            self.assertEqual(
                sorted(expected.neighbors(i)),
                sorted(actual.neighbors(i)),
                f"node {i}",
            )
        np.testing.assert_array_equal(expected.node_xyz, actual.node_xyz)
        self.assertEqual(
            expected.component_id_to_swc_id, actual.component_id_to_swc_id
        )

    def test_round_trip_preserves_structure(self):
        graph = ProposalGraph.from_fragments_graph(self.source)
        self.assert_same_structure(self.source, graph)
        self.assert_same_structure(self.source, graph.to_fragments_graph())

    def test_conversion_is_independent_of_the_source(self):
        graph = ProposalGraph.from_fragments_graph(self.source)
        graph.add_edge(0, NODES_PER_FRAGMENT)
        self.assertEqual(
            self.source.number_of_edges() + 1, graph.number_of_edges()
        )
        self.assertNotIn(
            NODES_PER_FRAGMENT, list(self.source.neighbors(0))
        )
        graph.node_xyz[0] = 12345.0
        self.assertNotEqual(12345.0, self.source.node_xyz[0][0])


if __name__ == "__main__":
    unittest.main()
