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
"""

from collections import defaultdict

import networkx as nx
import numpy as np

from neuron_proofreader.utils import geometry_util, util


def run(gt_graph, pred_graph):
    """
    Determines the set of accepted proposals.

    Parameters
    ----------
    gt_graph : SkeletonGraph
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
    gt_accepts = list()
    for proposal in pred_graph.get_sorted_proposals():
        # Proposal info
        i, j = tuple(proposal)
        id1 = pred_graph.node_component_id[i]
        id2 = pred_graph.node_component_id[j]

        # Check if fragments are aligned to the same GT skeletons
        if pred_to_gt[id1] != pred_to_gt[id2] or pred_to_gt[id1] is None:
            continue

        # Check proposal projection distance
        dist = compute_proposal_proj_dist(gt_graph, pred_graph, proposal)
        if dist > 8:
            continue

        # Check if proposal is structurally consistent
        gt_id = pred_to_gt[id1]
        if is_structure_consistent(gt_graph, pred_graph, gt_id, proposal):
            gt_accepts.append(proposal)
    return gt_accepts


# --- Core Routines ---
def compute_proposal_proj_dist(gt_graph, pred_graph, proposal):
    """
    Computes the average projection distance of a proposed edge to the ground
    truth graph.

    Parameters
    ----------
    gt_graph : SkeletonGraph
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
    gt_graph : SkeletonGraph
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

    # Find GT component best aligned to
    gt_id = util.find_best(dists_dict)
    if gt_id is None:
        return None

    # Compute alignment scores
    dists = np.array(dists_dict[gt_id])
    percent_aligned = len(dists) / len(nodes)
    aligned_score = np.percentile(dists, 60)
    return gt_id if aligned_score < 7 and percent_aligned > 0.6 else None


def get_pred_to_gt_mapping(gt_graph, pred_graph):
    """
    Gets fragments aligned to a single ground truth skeleton and builds a
    dictionary that maps these fragment IDs to the corresponding ground truth
    ID.

    Parameters
    ----------
    gt_graph : SkeletonGraph
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
    return pred_to_gt


def is_structure_consistent(gt_graph, pred_graph, gt_id, proposal):
    """
    Determines if the proposal connects two branches corresponding to either
    the same or adjacent branches on the ground truth. If either condition
    holds, then a subroutine is called to do additional check.

    Parameters
    ----------
    gt_graph : SkeletonGraph
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

    # Case 1: GT edges are identical
    if gt_edge_i == gt_edge_j:
        return True

    # Case 2: GT edges are adjacent
    if set(gt_edge_i).intersection(set(gt_edge_j)):
        # Proposal info
        i, j = tuple(proposal)
        xyz_i = pred_graph.node_xyz[i]
        xyz_j = pred_graph.node_xyz[j]

        # Get GT paths
        k = get_common_node(gt_edge_i, gt_edge_j)
        path_ik = get_path(gt_graph, k, xyz_i)
        path_jk = get_path(gt_graph, k, xyz_j)

        # Compare distances
        gt_dist = gt_graph.path_length(path_ik + path_jk[::-1])
        proposal_dist = pred_graph.proposal_length(proposal)
        return abs(proposal_dist - gt_dist) < 40

    return False


# --- Helpers ---
def find_closest_gt_edge(gt_graph, pred_graph, gt_id, root):
    """
    Finds the closest ground truth irreducible edge corresponding to a rooted
    subgraph at the given node from "pred_graph".

    Parameters
    ----------
    gt_graph : SkeletonGraph
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
    # Compute projections
    edge_cnt_dict = defaultdict(int)
    for node in pred_graph.nodes_within_distance(root, 40):
        gt_node = gt_graph.closest_node(pred_graph.node_xyz[node])
        gt_edge = get_irreducible_edge(gt_graph, gt_node)
        component_id = gt_graph.node_component_id[gt_edge[0]]
        if component_id == gt_id:
            edge_cnt_dict[gt_edge] += 1

    # Determine best match
    if edge_cnt_dict:
        edge = util.find_best(edge_cnt_dict)
        return frozenset(edge)
    else:
        return None


def get_common_node(edge1, edge2):
    """
    Finds the common node between the given edges.

    Parameters
    ----------
    edge1 : Tuple[int]
        Edge ID.
    edge2 : Tuple[int]
        Edge ID.
    """
    common_nodes = set(edge1).intersection(edge2)
    assert len(common_nodes) == 1
    return common_nodes.pop()


def get_irreducible_edge(graph, node):
    """
    Finds the irreducible edge containing the given node. Note that if the
    node is a branching point, then the first irreducible edge that is found
    is returned.

    Parameters
    ----------
    graph : SkeletonGraph
        Graph to be searched.
    node : int
        Node ID contained in the given graph.

    Returns
    -------
    edge : Tuple[int]
        Irreducible edge containing the given node.
    """
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


def get_path(gt_graph, source, xyz):
    """
    Returns the shortest path in the graph from a source node to the node
    closest to a given 3D coordinate.

    Parameters
    ----------
    gt_graph : SkeletonGraph
        Ground truth graph to be searched.
    source : int
        Node ID from which the path will start.
    xyz : numpy.ndarray
        3D coordinate used to identify the target node. The node in the graph
        closest to this coordinate will be used as the path endpoint.

    Returns
    -------
    path : List[int]
        Ordered list of node IDs representing the shortest path.
    """
    target = gt_graph.closest_node(xyz)
    return nx.shortest_path(gt_graph, source=source, target=target)
