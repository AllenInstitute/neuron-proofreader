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
from scipy.spatial import KDTree

import numpy as np
import rustworkx as rx

from neuron_proofreader.utils import util


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
    pred_to_gt = get_pred_to_gt_mapping(gt_graph, pred_graph)

    # Main
    gt_accepts = list()
    for proposal in pred_graph.list_proposals():
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


def merge_duplicate_tracings(
    gt_graph, max_dist=2.0, min_overlap=20, rm_dist=5.0, stitch_dist=10.0
):
    """
    Merges pairs of ground truth components that trace the same axon. Two
    components are duplicates when at least "min_overlap" nodes of one lie
    within "max_dist" microns of the other. The smaller component's nodes
    within "rm_dist" of the larger one are then removed (the looser radius
    absorbs tracing jitter along the shared stretch) and its remaining pieces
    are connected to the larger one, so the neuron becomes a single tree.

    Parameters
    ----------
    gt_graph : SkeletonGraph
        Graph built from ground truth SWC files, modified in place.
    max_dist : float, optional
        Distance (in microns) within which nodes count as duplicates when
        detecting pairs. Default is 2.0.
    min_overlap : int, optional
        Minimum number of duplicated nodes (i.e. microns at node_spacing=1)
        for two components to be merged. Default is 20.
    rm_dist : float, optional
        Distance (in microns) within which the smaller component's nodes are
        removed once a pair is confirmed. Default is 5.0.
    stitch_dist : float, optional
        Maximum distance (in microns) for connecting a remaining piece of the
        smaller component to the larger one. Default is 10.0.

    Returns
    -------
    int
        Number of merges performed.
    """
    n_merges = 0
    while True:
        pair = _find_duplicate_pair(gt_graph, max_dist, min_overlap)
        if pair is None:
            return n_merges
        _merge_duplicate_pair(gt_graph, *pair, rm_dist, stitch_dist)
        n_merges += 1


def _find_duplicate_pair(gt_graph, max_dist, min_overlap):
    overlap = defaultdict(int)
    for nodes in map(list, rx.connected_components(gt_graph)):
        component_id = gt_graph.node_component_id[nodes[0]]
        nbhds = gt_graph.kdtree.query_ball_point(
            gt_graph.node_xyz[np.array(nodes, dtype=int)], r=max_dist
        )
        for nbhd in nbhds:
            other_ids = set(gt_graph.node_component_id[nbhd].tolist())
            other_ids.discard(component_id)
            for other_id in other_ids:
                overlap[frozenset({component_id, other_id})] += 1
    if not overlap:
        return None
    best = max(overlap, key=overlap.get)
    return tuple(best) if overlap[best] >= min_overlap else None


def _merge_duplicate_pair(gt_graph, id1, id2, rm_dist, stitch_dist):
    nodes1 = np.where(gt_graph.node_component_id == id1)[0]
    nodes2 = np.where(gt_graph.node_component_id == id2)[0]
    if len(nodes1) < len(nodes2):
        nodes1, nodes2 = nodes2, nodes1
    big_tree = KDTree(gt_graph.node_xyz[nodes1])

    # Remove duplicated nodes from the smaller component
    dists, _ = big_tree.query(gt_graph.node_xyz[nodes2])
    dup_nodes = nodes2[dists < rm_dist]
    gt_graph.remove_nodes(dup_nodes.tolist(), relabel_nodes=False)

    # Connect each remaining piece of the smaller component to the larger one
    remaining = set(nodes2[dists >= rm_dist].tolist())
    for piece in rx.connected_components(gt_graph):
        piece = [int(i) for i in piece if i in remaining]
        if not piece:
            continue
        piece_dists, big_idxs = big_tree.query(gt_graph.node_xyz[piece])
        k = int(np.argmin(piece_dists))
        if piece_dists[k] < stitch_dist:
            gt_graph.add_edge(piece[k], int(nodes1[big_idxs[k]]), None)
    gt_graph.relabel_nodes()


def find_fragments_on_gt(gt_graph, pred_graph, min_frac=0.3, max_dist=7):
    """
    Finds fragments that lie on the ground truth, meaning at least "min_frac"
    of their nodes are within "max_dist" microns of a ground truth skeleton.
    Fragments failing this test belong to neurons that were not traced, so
    proposals involving them cannot be labeled.

    Parameters
    ----------
    gt_graph : SkeletonGraph
        Graph built from ground truth SWC files.
    pred_graph : ProposalGraph
        Graph built from predicted SWC files.
    min_frac : float, optional
        Minimum fraction of nodes near ground truth. Default is 0.3.
    max_dist : float, optional
        Distance (in microns) within which a node counts as near ground
        truth. Default is 7.

    Returns
    -------
    Set[str]
        SWC IDs of fragments that lie on the ground truth.
    """
    swc_ids = set()
    for nodes in map(list, rx.connected_components(pred_graph)):
        xyz = pred_graph.node_xyz[np.array(nodes, dtype=int)]
        dists, _ = gt_graph.kdtree.query(xyz)
        if (dists < max_dist).mean() >= min_frac:
            swc_ids.add(pred_graph.node_swc_id(nodes[0]))
    return swc_ids


def read_swc_ids(path):
    """
    Reads SWC IDs saved one per line.

    Parameters
    ----------
    path : str
        Path to text file.

    Returns
    -------
    Set[str]
        SWC IDs.
    """
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


def read_gt_accepts(path):
    """
    Reads accepted proposals saved as lines of the form "swc_id1, swc_id2".

    Parameters
    ----------
    path : str
        Path to text file.

    Returns
    -------
    Set[Frozenset[str]]
        Unordered pairs of SWC IDs whose proposal is accepted.
    """
    pairs = set()
    with open(path) as f:
        for line in f:
            if line.strip():
                id1, id2 = (s.strip() for s in line.split(","))
                pairs.add(frozenset({id1, id2}))
    return pairs


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
    n_pts = max(int(pred_graph.proposal_length(proposal)) + 1, 2)

    # Compute projection distances
    t = np.linspace(0, 1, n_pts)[:, None]
    line = (1 - t) * xyz_i + t * xyz_j
    proj_dists, _ = gt_graph.kdtree.query(line)
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
    # Project nodes onto ground truth
    xyz_arr = pred_graph.node_xyz[np.array(nodes, dtype=int)]
    dists_arr, nodes_arr = gt_graph.kdtree.query(xyz_arr)
    gt_ids = gt_graph.node_component_id[nodes_arr]

    # Find GT component with the most nearby nodes
    is_close = dists_arr < 7
    if not is_close.any():
        return None
    ids, counts = np.unique(gt_ids[is_close], return_counts=True)
    gt_id = ids[np.argmax(counts)]

    # Check alignment score
    percent_aligned = counts.max() / len(nodes)
    return gt_id if percent_aligned > 0.6 else None


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
    for nodes in map(list, rx.connected_components(pred_graph)):
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
        # Compare proposal length to GT path length between the projections
        # of its endpoints. Note: the path is not routed via the common node
        # since both endpoints may project onto the same GT edge.
        source = gt_graph.closest_node(pred_graph.node_xyz[i])
        path = get_path(gt_graph, source, pred_graph.node_xyz[j])
        if len(path) > 0:
            gt_dist = gt_graph.path_length(np.array(path))
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
            edge_cnt_dict[frozenset(gt_edge)] += 1

    # Determine best match
    if edge_cnt_dict:
        return util.find_best(edge_cnt_dict)
    else:
        return None


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
        if graph.degree(i) != 2:
            edge.append(i)
            if len(edge) == 2:
                break
            else:
                continue

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
    if target == source:
        return [source]
    paths = rx.graph_dijkstra_shortest_paths(
        gt_graph, source, target=target, default_weight=1.0
    )
    return list(paths[target]) if target in paths else list()
