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

from neuron_proofreader.utils import geometry_util, graph_util as gutil, util


def run(gt_graph, pred_graph):
    """
    Initializes ground truth for edge proposals.

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
    gt_graph.set_kdtree(node_type="branching")
    pred_to_gt = get_pred_to_gt_mapping(gt_graph, pred_graph)

    # Main
    accepts_graph = deepcopy(pred_graph)
    gt_accepts = list()
    for proposal in get_sorted_proposals(pred_graph):
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

        # Check if nodes are connected
        if check_connectedness(accepts_graph, proposal):
            continue

        # Check if proposal is structurally consistent
        gt_id = pred_to_gt[id1]
        is_consistent = is_structure_consistent(
            gt_graph, pred_graph, accepts_graph, gt_id, proposal
        )
        if is_consistent:
            accepts_graph.add_edge(i, j)
            gt_accepts.append(proposal)
    return gt_accepts


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
    for nodes in nx.connected_components(pred_graph):
        gt_id = find_aligned_component(gt_graph, pred_graph, nodes)
        if gt_id:
            node = util.sample_once(nodes)
            pred_to_gt[pred_graph.node_component_id[node]] = gt_id
    return pred_to_gt


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
    dists = defaultdict(list)
    n_pts = 0
    for edge in pred_graph.subgraph(nodes).edges:
        for xyz in pred_graph.edges[edge]["xyz"]:
            hat_xyz = geometry_util.kdtree_query(gt_graph.kdtree, xyz)
            gt_id = gt_graph.xyz_to_component_id(hat_xyz)
            d = geometry_util.dist(hat_xyz, xyz)
            dists[gt_id].append(d)
            n_pts += 1

    # Compute alignment score
    gt_id = util.find_best(dists)
    dists = np.array(dists[gt_id])
    percent_aligned = len(dists) / n_pts
    aligned_score = np.mean(dists[dists < np.percentile(dists, 80)])

    # Deterine whether aligned
    if (aligned_score < 4 and gt_id) and percent_aligned > 0.6:
        return gt_id
    else:
        return None


def check_connectedness(pred_graph, proposal):
    # Get path between nodes
    i, j = proposal
    try:
        path = nx.shortest_path(pred_graph, source=i, target=j)
    except nx.NetworkXNoPath:
        return False

    # Search path
    for node1, node2 in zip(path[0:-1], path[1:]):
        if frozenset((node1, node2)) not in pred_graph.proposals:
            return True
    return False

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
    for pt in geometry_util.make_line(xyz_i, xyz_j, n_pts):
        dist, _ = gt_graph.kdtree.query(pt)
        proj_dists.append(dist)
    return np.max(proj_dists)


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
    hat_edge_i = find_closest_gt_edge(gt_graph, pred_graph, gt_id, i)
    hat_edge_j = find_closest_gt_edge(gt_graph, pred_graph, gt_id, j)
    if hat_edge_i is None or hat_edge_j is None:
        return False

    # Case 1
    if hat_edge_i == hat_edge_j:
        return True

    # Case 2
    if set(hat_edge_i).intersection(set(hat_edge_j)):
        # Orient ground truth edges
        i, j = tuple(proposal)
        hat_edge_xyz_i, hat_edge_xyz_j = orient_edges(
            gt_graph.edges[hat_edge_i]["xyz"],
            gt_graph.edges[hat_edge_j]["xyz"]
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
def find_closest_gt_edge(gt_graph, pred_graph, gt_id, i):
    """
    Finds the closest ground truth edge corresponding to a rooted subgraph
    at the given node from "pred_graph".

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
    hat_edge_i : Tuple[int] or None
        Closest ground-truth edge to the rooted subgraph at the given node, or
        None if no edge could be found.
    """
    depth = 16
    while depth <= 64:
        hat_edge_i = project_region(gt_graph, pred_graph, gt_id, i, depth)
        if hat_edge_i is None:
            depth += 16
        else:
            break
    return hat_edge_i


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


def get_sorted_proposals(pred_graph):
    """
    Return proposals sorted by physical length.

    Parameters
    ----------
    pred_graph : ProposalGraph
        Graph containing proposals to be sorted.

    Returns
    -------
    List[Frozenset[int]]
        List of proposals sorted by increasing proposal length.
    """
    proposals = pred_graph.list_proposals()
    lengths = [pred_graph.proposal_length(p) for p in proposals]
    return [proposals[i] for i in np.argsort(lengths)]


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


def project_region(gt_graph, pred_graph, gt_id, i, depth=16):
    """
    Projects a rooted subgraph onto the given ground truth graph.

    Parameters
    ----------
    gt_graph : ProposalGraph
        Graph to be projected onto.
    pred_graph : ProposalGraph
        Graph to extract rooted subgraph from.
    gt_id : int
        Connected component ID of component in ground truth graph.
    i : int
        Node ID of the root of the subgraph to be extracted and projected.
    depth : int, optional
        Depth (in microns) of rooted subgraph to extract.
    """
    hits = defaultdict(list)
    for edge_xyz_list in pred_graph.truncated_edge_attr_xyz(i, 24):
        for xyz in edge_xyz_list:
            hat_xyz = geometry_util.kdtree_query(gt_graph.kdtree, xyz)
            hat_edge = gt_graph.xyz_to_edge[hat_xyz]
            if gt_graph.node_component_id[hat_edge[0]] == gt_id:
                hits[hat_edge].append(hat_xyz)
    return util.find_best(hits)
