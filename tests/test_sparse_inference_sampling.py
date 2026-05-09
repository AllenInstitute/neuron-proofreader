"""
Tests for biased sparse inference-time sampling
("compute_interesting_nodes"). Uses a minimal duck-typed graph so the
suite runs without importing the merge_inference vision/ML stack.
"""

import unittest

import networkx as nx
import numpy as np
from scipy.spatial import KDTree

from neuron_proofreader.merge_proofreading.sparse_sampling import (
    compute_interesting_nodes,
)


class _FakeSkeletonGraph(nx.Graph):
    """
    Minimal stand-in for SkeletonGraph that exposes only what
    "compute_interesting_nodes" depends on.
    """

    def __init__(self, node_xyz, node_component_id):
        super().__init__()
        self.node_xyz = np.asarray(node_xyz, dtype=np.float32)
        self.node_component_id = np.asarray(node_component_id, dtype=int)
        self.kdtree = None
        self.add_nodes_from(range(len(self.node_xyz)))

    def branching_nodes(self):
        return [n for n in self.nodes if self.degree[n] > 2]

    def dist(self, i, j):
        return float(np.linalg.norm(self.node_xyz[i] - self.node_xyz[j]))

    def set_kdtree(self):
        self.kdtree = KDTree(self.node_xyz)


def _build_graph(node_xyz, node_component_id, edges):
    graph = _FakeSkeletonGraph(node_xyz, node_component_id)
    graph.add_edges_from(edges)
    graph.set_kdtree()
    return graph


class TestComputeInterestingNodes(unittest.TestCase):

    def test_isolated_linear_axon_yields_empty_set(self):
        # 100 nodes in a straight line, 1um spacing, single component, no
        # branches. Nothing should be interesting.
        n = 100
        xyz = np.zeros((n, 3))
        xyz[:, 0] = np.arange(n)
        cid = np.zeros(n, dtype=int)
        edges = [(i, i + 1) for i in range(n - 1)]

        graph = _build_graph(xyz, cid, edges)
        result = compute_interesting_nodes(
            graph, branch_radius=25.0, proximity_radius=15.0
        )
        self.assertEqual(result, set())

    def test_linear_axon_with_midbranch_picks_only_branch_region(self):
        # Main axon: nodes 0..99 along x at 1um spacing.
        # Spur: 5 extra nodes branching off node 50 along y.
        # Node 50 is the only branching node (degree 3).
        n_main = 100
        n_branch = 5
        xyz = np.zeros((n_main + n_branch, 3))
        xyz[:n_main, 0] = np.arange(n_main)
        xyz[n_main:, 0] = 50.0
        xyz[n_main:, 1] = np.arange(1, n_branch + 1)
        cid = np.zeros(len(xyz), dtype=int)

        edges = [(i, i + 1) for i in range(n_main - 1)]
        edges.append((50, n_main))
        edges.extend(
            [(n_main + k, n_main + k + 1) for k in range(n_branch - 1)]
        )

        graph = _build_graph(xyz, cid, edges)
        result = compute_interesting_nodes(
            graph, branch_radius=10.0, proximity_radius=15.0
        )

        # Within 10um of the branch (node 50): nodes 41..49 and 51..59,
        # plus all 5 spur nodes (max dist 5um).
        self.assertIn(50, result)
        self.assertIn(45, result)
        self.assertIn(55, result)
        self.assertIn(n_main, result)
        self.assertIn(n_main + n_branch - 1, result)
        # Outside the branch_radius window
        self.assertNotIn(0, result)
        self.assertNotIn(99, result)
        self.assertNotIn(35, result)
        self.assertNotIn(70, result)

    def test_two_parallel_axons_picks_only_close_middle(self):
        # Axon A (cid=0): y=0 for nodes 0..59.
        # Axon B (cid=1): y=30 at the ends, y=5 for middle nodes 25..34.
        # Only the middles are within proximity_radius=10 of each other.
        # No branches anywhere.
        n = 60
        xyz_a = np.zeros((n, 3))
        xyz_a[:, 0] = np.arange(n)

        xyz_b = np.zeros((n, 3))
        xyz_b[:, 0] = np.arange(n)
        xyz_b[:, 1] = 30.0
        for i in range(25, 35):
            xyz_b[i, 1] = 5.0

        xyz = np.vstack([xyz_a, xyz_b])
        cid = np.zeros(2 * n, dtype=int)
        cid[n:] = 1

        edges = [(i, i + 1) for i in range(n - 1)]
        edges.extend([(n + i, n + i + 1) for i in range(n - 1)])

        graph = _build_graph(xyz, cid, edges)
        result = compute_interesting_nodes(
            graph, branch_radius=25.0, proximity_radius=10.0
        )

        # Axon A node 30 sits at (30, 0); Axon B node n+30 at (30, 5):
        # distance 5 < 10, both interesting. End nodes are ~30um apart.
        self.assertIn(30, result)
        self.assertIn(n + 30, result)
        self.assertNotIn(0, result)
        self.assertNotIn(n - 1, result)
        self.assertNotIn(n, result)
        self.assertNotIn(2 * n - 1, result)


if __name__ == "__main__":
    unittest.main()
