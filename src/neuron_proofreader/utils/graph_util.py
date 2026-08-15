"""
Created on Fri Aug 15 12:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Graph utility routines for converting Arborist skeleton subgraphs into
data structures consumed by graph-based models.

"""

from collections import defaultdict

import numpy as np

from arborist.data.datasets import TreeSample
from arborist.utils.graph_utils import topological_decomposition


def subgraph_to_tree_sample(subgraph, root):
    """
    Convert an Arborist rooted subgraph into a TreeSample for the Arborist
    encoder.

    Topologically decomposes the subgraph into curves (paths between the root,
    branching nodes, and leaves), represents each curve as first-order finite
    differences, and builds the corresponding line-graph edge index.

    Parameters
    ----------
    subgraph : SkeletonGraph
        Local subgraph returned by SkeletonGraph.rooted_subgraph.
    root : int
        Node ID of the subgraph root (e.g. a candidate merge site).

    Returns
    -------
    TreeSample
        Curve finite-differences and line-graph edge index ready for
        arborist.models.arborist.Arborist.encode.
    """
    _, paths, topo_edge_index = topological_decomposition(subgraph, root=root)

    curves = []
    for path in paths:
        xyz = subgraph.node_xyz[path].copy()
        xyz -= xyz[0]
        if len(xyz) > 1:
            xyz[1:] -= xyz[:-1].copy()
        curves.append(xyz)

    edge_index = build_line_graph_edge_index(topo_edge_index)
    return TreeSample(curves, edge_index)


def build_line_graph_edge_index(topo_edge_index):
    """
    Convert a topological edge list into a line-graph edge index.

    Two curves share a line-graph edge when they meet at the same topological
    node (root, branching node, or leaf).

    Parameters
    ----------
    topo_edge_index : list of (int, int)
        Pairs of indices into the topological node list, one pair per curve.

    Returns
    -------
    numpy.ndarray
        Shape (2, E), dtype int64 — source and destination curve indices.
    """
    topo_to_curves = defaultdict(list)
    for curve_idx, (u, v) in enumerate(topo_edge_index):
        topo_to_curves[u].append(curve_idx)
        topo_to_curves[v].append(curve_idx)

    src, dst = [], []
    for neighbors in topo_to_curves.values():
        for i in neighbors:
            for j in neighbors:
                if i != j:
                    src.append(i)
                    dst.append(j)

    if not src:
        return np.zeros((2, 0), dtype=np.int64)
    return np.array([src, dst], dtype=np.int64)
