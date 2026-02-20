"""
Created on Fri March 1 16:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org


This code identifies proposals between fragments that align with the same
ground truth skeleton and are structurally consistent.

    Algorithm
    ---------
    1. Find fragments aligned to a single ground truth skeleton and build a
       dictionary that maps these fragment IDs to the corresponding ground
       truth ID.

    2. Iterate over all proposals generated between fragments. A proposal is
       accepted if:
            - Both fragments align to the same ground truth skeleton.
            - Proposal is structurally consistent, meaning the connection
              preserves geometric continuity and branching topology consistent
              with the ground truth structure.

Note: We use the convention that a fragment refers to a connected component in
      "pred_graph".
"""

from collections import defaultdict
from copy import deepcopy

import networkx as nx
import numpy as np

from neuron_proofreader.utils import geometry_util, util


def run(gt_graph, pred_graph):
    """
    Determines ground truth for edge proposals.

    Parameters
    ----------
    gt_graph : ProposalGraph
        Graph built from ground truth SWC files.
    pred_graph : ProposalGraph
        Graph build from predicted SWC files.

    Returns
    -------
    gt_accepts : List[Frozenset[int]]
        Proposals aligned to and structurally consistent with ground truth.
        Note: a model will learn to accept these proposals.
    """
    # Initializations
    gt_graph.set_kdtree()
    pred_graph.set_kdtree()
    pred_to_gt = get_pred_to_gt_mapping(gt_graph, pred_graph)

    # Main
    accepts_graph = deepcopy(pred_graph)
    gt_accepts = list()
    for proposal in pred_graph.get_sorted_proposals():
        # Extract proposal info
        i, j = tuple(proposal)
        id1 = pred_graph.node_component_id[i]
        id2 = pred_graph.node_component_id[j]

        # Check if fragments are aligned to the same ground truth skeletons
        if pred_to_gt[id1] != pred_to_gt[id2] or pred_to_gt[id1] is None:
            continue

        # Check proposal projection distance
        dist = compute_proposal_proj_dist(gt_graph, pred_graph, proposal)
        if dist > 8:
            continue

        # Check if proposal is structurally consistent
        is_consistent = is_structure_consistent(
            gt_graph, pred_graph, accepts_graph, pred_to_gt[id1], proposal
        )
        if is_consistent:
            accepts_graph.add_edge(i, j)
            gt_accepts.append(proposal)
    return gt_accepts


# --- Core Routines ---
def compute_proposal_proj_dist(gt_graph, pred_graph, proposal):
    """
    Computes the average projection distance of a proposed edge to the ground
    truth graph.

    Parameters
    ----------
    gt_graph : ProposalGraph
        Graph built from ground truth SWC files.
    pred_graph : ProposalGraph
        Graph build from predicted SWC files.
    proposal : FrozenSet[int]
        Propoal to compute projection distance of.

    Returns
    -------
    float
        Average projection distance.
    """
    # Extract proposal info
    i, j = proposal
    xyz_i = pred_graph.node_xyz[i]
    xyz_j = pred_graph.node_xyz[j]
    n_pts = max(int(pred_graph.proposal_length(proposal)) + 1, 1)

    # Compute projection distances
    proj_dists = list()
    for xyz in geometry_util.make_line(xyz_i, xyz_j, n_pts):
        dist, _ = gt_graph.kdtree.query(xyz)
        proj_dists.append(dist)
    return np.percentile(proj_dists, 90)


def find_aligned_component(gt_graph, pred_graph, nodes):
    """
    Determines if the given nodes are spatially aligned to a single connected
    component in the ground truth graph. The node coordinates are projected
    onto "gt_graph", and the average projection distance is computed. If this
    avg distance is less than 4 µm and most projections fall within the same
    connected component of gt_graph, the fragment is considered aligned.

    Parameters
    ----------
    gt_graph : ProposalGraph
        Graph built from ground truth SWC files.
    pred_graph : ProposalGraph
        Graph build from predicted SWC files.
    nodes : Set[int]
        Nodes from a connected component in "pred_graph".

    Returns
    -------
    str or None
        Indication of whether connected component "nodes" is aligned to a
        connected component in "gt_graph".
    """
    # Compute projection distances
    xyz_arr = pred_graph.node_xyz[np.array(nodes, dtype=int)]
    dists_arr, nodes_arr = gt_graph.kdtree.query(xyz_arr)

    # Group projection distances by connected component
    dists_dict = dict()
    gt_ids = gt_graph.node_component_id[nodes_arr]
    for gt_id in np.unique(gt_ids):
        dists_dict[gt_id] = dists_arr[gt_ids == gt_id]

    # Deterine whether aligned
    gt_id = util.find_best(dists_dict)
    dists = np.array(dists_dict[gt_id])
    percent_aligned = len(dists) / len(nodes)
    aligned_score = np.percentile(dists, 60)

    if (aligned_score < 7 and gt_id is not None) and percent_aligned > 0.6:
        return gt_id
    else:
        return None


def get_pred_to_gt_mapping(gt_graph, pred_graph):
    """
    Gets fragments aligned to a single ground truth skeleton and builds a
    dictionary that maps these fragment IDs to the corresponding ground truth
    ID.

    Parameters
    ----------
    gt_graph : ProposalGraph
        Graph built from ground truth SWC files.
    pred_graph : ProposalGraph
        Graph build from predicted SWC files.

    Returns
    -------
    pred_to_gt : Dict[int, int]
        Dictionary that maps fragment IDs to the corresponding ground truth
        ID.
    """
    pred_to_gt = defaultdict(lambda: None)
    for nodes in map(list, nx.connected_components(pred_graph)):
        gt_id = find_aligned_component(gt_graph, pred_graph, nodes)
        if gt_id is not None:
            pred_id = pred_graph.node_component_id[nodes[0]]
            pred_to_gt[pred_id] = gt_id
            pred_id = pred_graph.component_id_to_swc_id[pred_id]
            gt_id = gt_graph.component_id_to_swc_id[gt_id]
    return pred_to_gt


def is_structure_consistent(
    gt_graph, pred_graph, accepts_graph, gt_id, proposal
):
    """
    Determines if the proposal connects two branches corresponding to either
    the same or adjacent branches on the ground truth. If either condition
    holds, then a subroutine is called to do additional check.

    Parameters
    ----------
    gt_graph : ProposalGraph
        Graph built from ground truth SWC files.
    pred_graph : ProposalGraph
        Graph build from predicted SWC files.
    proposal : Frozenset[int]
        Proposal to be checked.

    Returns
    -------
    bool
        Indication of whether proposal is structurally consistent with ground
        truth.
    """
    # Find irreducible edges in gt_graph closest to edges connected to proposal
    i, j = tuple(proposal)
    gt_edge_i = find_closest_gt_edge(gt_graph, pred_graph, gt_id, i)
    gt_edge_j = find_closest_gt_edge(gt_graph, pred_graph, gt_id, j)
    if gt_edge_i is None or gt_edge_j is None:
        return False

    # Case 1
    if gt_edge_i == gt_edge_j:
        return True

    # Case 2
    if set(gt_edge_i).intersection(set(gt_edge_j)):
        # Orient ground truth edges
        i, j = tuple(proposal)
        hat_edge_xyz_i, hat_edge_xyz_j = orient_edges(
            gt_graph.edges[gt_edge_i]["xyz"],
            gt_graph.edges[gt_edge_j]["xyz"]
        )

        # Find index of closest points on ground truth edges
        xyz_i = pred_graph.node_xyz[i]
        xyz_j = pred_graph.node_xyz[j]
        idx_i = find_closest_point(hat_edge_xyz_i, xyz_i)
        idx_j = find_closest_point(hat_edge_xyz_j, xyz_j)

        len_1 = length_up_to(hat_edge_xyz_i, idx_i)
        len_2 = length_up_to(hat_edge_xyz_j, idx_j)
        gt_dist = len_1 + len_2
        proposal_dist = pred_graph.proposal_length(proposal)
        return abs(proposal_dist - gt_dist) < 40

    # Fail: Proposal is between non-adjacent branches
    return False


# --- Helpers ---
def find_closest_gt_edge(gt_graph, pred_graph, gt_id, root):
    """
    Finds the closest ground truth irreducible edge corresponding to a rooted
    subgraph at the given node from "pred_graph".

    Parameters
    ----------
    gt_graph : ProposalGraph
        Ground truth graph to be searched.
    pred_graph : ProposalGraph
        Graph to extract rooted subgraph from.
    gt_id : int
        Connected component ID of component in ground truth graph.
    i : int
        Node ID of the root of the subgraph to be extracted and projected.

    Returns
    -------
    gt_edge : Tuple[int] or None
        Closest ground-truth edge to the rooted subgraph at the given node, or
        None if no edge could be found.
    """
    edge_cnt_dict = defaultdict(int)
    for node in pred_graph.nodes_within_distance(root, 40):
        gt_node = gt_graph.find_closest_node(pred_graph.node_xyz[node])
        gt_edge = get_irreducible_edge(gt_graph, gt_node)
        component_id = gt_graph.node_component_id[gt_edge[0]]
        if component_id == gt_id:
            edge_cnt_dict[gt_edge] += 1
    return util.find_best(edge_cnt_dict) if edge_cnt_dict else None


def find_closest_point(xyz_list, query_xyz):
    """
    Finds the index of the point closest to the given query coordinate.

    Parameters
    ----------
    xyz_list : List[Tuple[float]]
        List of coordinates to be searched.
    query_xyz : Tuple[float]
        Query 3D coordinate.

    Returns
    -------
    best_idx : int
        Index of the closest point in "xyz_list".
    """
    best_dist = np.inf
    best_idx = np.inf
    for idx, xyz in enumerate(xyz_list):
        dist = geometry_util.dist(query_xyz, xyz)
        if dist < best_dist:
            best_dist = dist
            best_idx = idx
    return best_idx


def get_irreducible_edge(graph, node):
    # Search
    edge = list()
    queue = [node]
    visited = set(queue)
    while queue:
        # Visit node
        i = queue.pop()
        if graph.degree[i] != 2:
            edge.append(i)
            if len(edge) == 2:
                break

        # Update queue
        for j in graph.neighbors(i):
            if j not in visited:
                queue.append(j)
                visited.add(j)
    return tuple(edge)


def length_up_to(path_pts, idx):
    """
    Computes the cumulative path length from the start of a 3D point path up
    to a given index.

    Parameters
    ----------
    path_pts : numpy.ndarray
        3D points defining a continuous path.
    idx : int
        Index up to which the cumulative length is computed.

    Returns
    -------
    length : float
         Cumulative path length from the start up to point "idx".
    """
    length = 0
    for i in range(0, idx):
        length += geometry_util.dist(path_pts[i], path_pts[i + 1])
    return length


def orient_edges(xyz_edge_i, xyz_edge_j):
    """
    Orients two edges so that their closest endpoints are aligned at index 0.

    Parameters
    ----------
    xyz_edge_i : numpy.ndarray
        Ordered 3D coordinates defining the first branch.
    xyz_edge_j : numpy.ndarray
        Ordered 3D coordinates defining the second branch.
    """
    # Compute distances
    dist_1 = geometry_util.dist(xyz_edge_i[0], xyz_edge_j[0])
    dist_2 = geometry_util.dist(xyz_edge_i[0], xyz_edge_j[-1])
    dist_3 = geometry_util.dist(xyz_edge_i[-1], xyz_edge_j[0])
    dist_4 = geometry_util.dist(xyz_edge_i[-1], xyz_edge_j[-1])

    # Orient coordinates to match at 0-th index
    min_dist = np.min([dist_1, dist_2, dist_3, dist_4])
    if dist_2 == min_dist or dist_4 == min_dist:
        xyz_edge_j = np.flip(xyz_edge_j, axis=0)
    if dist_3 == min_dist or dist_4 == min_dist:
        xyz_edge_i = np.flip(xyz_edge_i, axis=0)
    return xyz_edge_i, xyz_edge_j
