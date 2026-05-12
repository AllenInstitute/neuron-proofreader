"""
Helpers for biased sparse inference-time sampling: pre-selects skeleton
nodes that sit near a branch point or near another axon, so that merge
detection can skip long, isolated axon segments.

Kept in its own module (only numpy + a networkx graph object) so it can be
exercised by unit tests without pulling in the full merge_inference image
and ML stack.
"""

import numpy as np


def compute_interesting_nodes(graph, branch_radius=25.0, proximity_radius=15.0):
    """
    Selects nodes worth running merge detection on: the union of (a) nodes
    within "branch_radius" graph-distance of a branching node, and (b) nodes
    within "proximity_radius" Euclidean distance of a node belonging to a
    different connected component.

    Parameters
    ----------
    graph : SkeletonGraph
        Skeleton graph with "node_xyz", "node_component_id", and "kdtree"
        populated. Must expose "get_branchings()", "neighbors(i)",
        "dist(i, j)", and "set_kdtree()".
    branch_radius : float, optional
        Graph-distance window (microns) around branching nodes. Default 25.
    proximity_radius : float, optional
        Euclidean threshold (microns) for nodes treated as "near another
        axon". Default 15 (matches "geometry_util.is_double_merge").

    Returns
    -------
    Set[int]
        Node IDs to sample at inference time.
    """
    # Branch-region nodes: bounded DFS from every branching node
    branch_set = set(graph.get_branchings())
    queue = [(i, 0.0) for i in branch_set]
    while queue:
        i, dist_i = queue.pop()
        for j in graph.neighbors(i):
            dist_j = dist_i + graph.dist(i, j)
            if j not in branch_set and dist_j < branch_radius:
                branch_set.add(j)
                queue.append((j, dist_j))

    # Inter-component proximity nodes
    if graph.kdtree is None:
        graph.set_kdtree()
    proximity_set = set()
    for i in graph.nodes:
        idxs = np.array(
            graph.kdtree.query_ball_point(graph.node_xyz[i], proximity_radius)
        )
        if idxs.size and np.any(
            graph.node_component_id[idxs] != graph.node_component_id[i]
        ):
            proximity_set.add(i)

    return branch_set | proximity_set
