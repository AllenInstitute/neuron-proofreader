"""
Created on Sat July 15 9:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code that generates edge proposals for a given graph.

"""

from collections import deque

import numpy as np
from tqdm import tqdm

from neuron_proofreader.utils import geometry_util

DOT_THRESHOLD = -0.35
SEARCH_SCALING_FACTOR = 1.5


def run(graph, search_radius, complex_bool=False, long_range_bool=True):
    """
    Generates edge proposals from leaf nodes to other leaf nodes within the
    given search radius. Note: proposals can be generated between leaf and
    non-leaf nodes if "complex_bool" is set to True.

    Parameters
    ----------
    graph : ProposalGraph
        Graph that proposals will be generated for.
    search_radius : float
        Maximum Euclidean distance between endpoints of proposal.
    complex_bool : bool, optional
        Indication of whether to generate proposals between leaf and non-
        leaf nodes. Default is False.
    long_range_bool : bool, optional
        Indication of wether to attempt generating proposals for a leaf
        without existing proposals by expanding the search radius to
        "SEARCH_SCALING_FACTOR" × radius microns. Default is True.
    """
    # Initializations
    connections = dict()
    kdtree = init_kdtree(graph, complex_bool)
    if graph.verbose:
        iterable = tqdm(graph.get_leafs(), desc="Proposals")
    else:
        iterable = graph.get_leafs()

    # Main
    for leaf in iterable:
        # Check fragment satisfies size requirement
        length = graph.path_length(
            max_depth=graph.min_size_with_proposals, root=leaf
        )
        if length < graph.min_size_with_proposals:
            continue

        # Generate potential proposals
        candidates = get_candidates(
            graph,
            leaf,
            kdtree,
            search_radius,
            graph.proposals_per_leaf,
            complex_bool,
        )

        # Generate long range proposals (if applicable)
        if len(candidates) == 0 and long_range_bool:
            candidates = get_candidates(
                graph,
                leaf,
                kdtree,
                search_radius * SEARCH_SCALING_FACTOR,
                -1,
                True,
            )

        # Determine which potential proposals to keep
        for i in candidates:
            leaf_component_id = graph.node_component_id[leaf]
            node_component_id = graph.node_component_id[i]
            pair_id = frozenset((leaf_component_id, node_component_id))
            if pair_id in connections.keys():
                cur_proposal = connections[pair_id]
                cur_dist = graph.proposal_length(cur_proposal)
                if graph.dist(leaf, i) < cur_dist:
                    graph.remove_proposal(cur_proposal)
                    del connections[pair_id]
                else:
                    continue

            # Add proposal
            graph.add_proposal(leaf, i)
            connections[pair_id] = frozenset({leaf, i})


def init_kdtree(graph, complex_bool):
    """
    Initializes a KD-Tree used to generate proposals.

    Parameters
    ----------
    graph : FragmentsGraph
        Graph that proposals will be generated for.
    complex_bool : bool
        Indication of whether to generate complex proposals.

    Returns
    -------
    scipy.spatial.cKDTree
        kdtree built from all xyz coordinates across edges in graph if
        complex_bool is True; otherwise, only built from leaf nodes.
    """
    if complex_bool:
        return graph.get_kdtree()
    else:
        return graph.get_kdtree(node_type="leaf")


def get_candidates(
    graph, leaf, kdtree, radius, max_proposals, complex_bool
):
    # Generate candidates
    candidates = list()
    for xyz in search_kdtree(graph, leaf, kdtree, radius, max_proposals):
        i = get_connecting_node(graph, leaf, xyz, radius, complex_bool)
        if i is not None:
            if graph.is_valid_proposal(leaf, i, complex_bool):
                candidates.append(i)

    # Process results
    if max_proposals < 0 and len(candidates) == 1:
        is_leaf = graph.degree[candidates[0]] == 1
        return candidates if is_leaf else []
    else:
        return list() if max_proposals < 0 else candidates


def search_kdtree(graph, leaf, kdtree, radius, max_proposals):
    """
    Generates proposals extending from node "leaf" by finding candidate xyz
    points on distinct connected components nearby.

    Parameters
    ----------
    graph : FragmentsGraph
        Graph that proposals will be generated for.
    kdtree : scipy.spatial.cKDTree
        ...
    leaf : int
        Leaf node that proposals are to be generated from.
    radius : float
        Maximum Euclidean distance between endpoints of proposal.
    max_proposals : int
        Maximum number of proposals that can be generated from a leaf node.

    Returns
    -------
    List[Tuple[float]]
        xyz coordinates of candidates.
    """
    # Generate candidates
    candidates = dict()
    leaf_xyz = graph.node_xyz[leaf]
    for xyz in geometry_util.query_ball(kdtree, leaf_xyz, radius):
        component_id = graph.xyz_to_component_id(xyz)
        if component_id != graph.node_component_id[leaf]:
            dist = geometry_util.dist(leaf_xyz, xyz)
            if component_id not in candidates.keys():
                candidates[component_id] = {"dist": dist, "xyz": tuple(xyz)}
            elif dist < candidates[component_id]["dist"]:
                candidates[component_id] = {"dist": dist, "xyz": tuple(xyz)}

    # Check whether to filter
    if max_proposals < 0:
        return [] if len(candidates) > 1 else list_candidates_xyz(candidates)
    else:
        return get_best(candidates, max_proposals)


def get_best(candidates, max_proposals):
    """
    Gets the best candidates to become edge proposals given a dictionary
    containing candidates.

    Parameters
    ----------
    candidates : dict
        Dictionary that maps swc ids to a dictionary containing a distance and
        xyz coordinate.
    max_proposals : int
        Maximum number of proposals that can be generated from a leaf node.

    Returns
    -------
    List[Tuple[float]]
        xyz coordinates of candidates.
    """
    while len(candidates) > max_proposals:
        worst_key = None
        for key in candidates.keys():
            if worst_key is None:
                worst_key = key
            elif candidates[key]["dist"] > candidates[worst_key]["dist"]:
                worst_key = key
        del candidates[worst_key]
    return list_candidates_xyz(candidates)


def get_connecting_node(graph, leaf, xyz, radius, complex_bool):
    """
    Gets node that proposal emanating from "leaf" will connect to.

    Parameters
    ----------
    graph : FragmentsGraph
        Graph containing "leaf".
    leaf : int
        Leaf node.
    xyz : numpy.ndarray
        xyz coordinate.

    Returns
    -------
    int
        Node id that proposal will connect to.
    """
    # Check if edge exists
    try:
        edge = graph.xyz_to_edge[xyz]
    except:
        return None

    # Find connecting node
    node = get_closer_endpoint(graph, edge, xyz)
    if graph.dist(leaf, node) < radius:
        return node
    elif complex_bool:
        attrs = graph.get_edge_data(*edge)
        idx = np.where(np.all(attrs["xyz"] == xyz, axis=1))[0][0]
        if type(idx) is int:
            return graph.split_edge(edge, attrs, idx)
    return None


def get_closer_endpoint(graph, edge, xyz):
    """
    Gets node from "edge" that is closer to "xyz".

    Parameters
    ----------
    graph : FragmentsGraph
        Graph containing "edge".
    edge : tuple
        Edge to be checked.
    xyz : numpy.ndarray
        xyz coordinate.

    Returns
    -------
    int
        Node closer to "xyz".
    """
    i, j = tuple(edge)
    d_i = geometry_util.dist(graph.node_xyz[i], xyz)
    d_j = geometry_util.dist(graph.node_xyz[j], xyz)
    return i if d_i < d_j else j


# --- Trim Endpoints ---
def run_endpoint_trimming(graph, search_radius):
    # Initializations
    augmented_search_radius = search_radius * SEARCH_SCALING_FACTOR
    long_range, in_range = deque(), deque()
    for p in graph.proposals:
        if graph.proposal_length(p) < augmented_search_radius:
            in_range.append(p)
        else:
            long_range.append(p)

    # Trim endpoints by proposal type
    trim_proposal_endpoints(graph, in_range, search_radius)
    trim_proposal_endpoints(
        graph, long_range, augmented_search_radius
    )


def trim_proposal_endpoints(graph, proposals, max_length):
    while proposals:
        p = proposals.pop()
        is_simple = graph.is_simple(p)
        is_single = graph.is_single_proposal(p)
        if is_simple and is_single:
            trim_endpoints_at_proposal(graph, p, max_length)
        elif graph.proposal_length(p) > max_length:
            graph.remove_proposal(p)


def trim_endpoints_at_proposal(graph, proposal, max_length):
    # Find closest points between proposal branches
    i, j = tuple(proposal)
    pts_i = graph.edge_attr(i, key="xyz", ignore=True)[0]
    pts_j = graph.edge_attr(j, key="xyz", ignore=True)[0]
    dist_ij, (idx_i, idx_j) = find_closest_pair(pts_i, pts_j)

    # Update branches (if applicable)
    if dist_ij > max_length:
        graph.remove_proposal(frozenset((i, j)))
    elif dist_ij + 2 < geometry_util.dist(pts_i[0], pts_j[0]):
        if compute_dot(pts_i, pts_j, idx_i, idx_j) < DOT_THRESHOLD:
            trim_to_idx(graph, i, idx_i)
            trim_to_idx(graph, j, idx_j)


def find_closest_pair(pts1, pts2):
    best_dist, best_idxs = np.inf, (0, 0)
    i, length1 = -1, 0
    while length1 < 20 and i < len(pts1) - 1:
        i += 1
        length1 += geometry_util.dist(pts1[i], pts1[i - 1]) if i > 0 else 0

        # Search other branch
        j, length2 = -1, 0
        while length2 < 20 and j < len(pts2) - 1:
            j += 1
            length2 += geometry_util.dist(pts2[j], pts2[j - 1]) if j > 0 else 0

            # Check distance between points
            dist = geometry_util.dist(pts1[i], pts2[j])
            if dist < best_dist:
                best_dist = dist
                best_idxs = (i, j)
    return best_dist, best_idxs


def trim_to_idx(graph, i, idx):
    """
    Trims the end of a branch specified by the leaf node "i".

    Parameters
    ----------
    graph : FragmentsGraph
        Graph containing node "i"
    i : int
        Leaf node ID.
    idx : int
        Branch is trimmed to the index "idx".
    """
    # Update node
    edge_xyz = graph.edge_attr(i, key="xyz", ignore=True)[0]
    edge_radii = graph.edge_attr(i, key="radius", ignore=True)[0]
    graph.node_xyz[i] = edge_xyz[idx]
    graph.node_radius[i] = edge_radii[idx]

    # Update edge
    nb = list(graph.neighbors(i))[0]
    graph.edges[i, nb]["xyz"] = edge_xyz[idx:]
    graph.edges[i, nb]["radius"] = edge_radii[idx:]
    for k in range(idx):
        try:
            del graph.xyz_to_edge[tuple(edge_xyz[k])]
        except KeyError:
            pass


# --- Helpers ---
def list_candidates_xyz(candidates):
    """
    Lists the xyz coordinates of candidates.

    Parameters
    ----------
    candidates : dict
        Dictionary that maps swc ids to a dictionary containing a distance and
        xyz coordinate.

    Returns
    -------
    List[Tuple[int]]
        xyz coordinates of candidates.
    """
    return [candidates[key]["xyz"] for key in candidates.keys()]


def compute_dot(branch1, branch2, idx1, idx2):
    """
    Computes dot product between principal components of "branch1" and
    "branch_2".

    Parameters
    ----------
    branch1 : numpy.ndarray
        xyz coordinates of some branch from a graph.
    branch_2 : numpy.ndarray
        xyz coordinates of some branch from a graph.
    idx1 : int
        Index that "branch1" would be trimmed to (i.e. xyz coordinates from 0
        to "idx1" would be deleted from "branch1").
    idx2 : int
        Index that "branch_2" would be trimmed to (i.e. xyz coordinates from 0
        to "idx2" would be deleted from "branch_2").

    Returns
    -------
    float
        Dot product between principal components of "branch1" and "branch_2".
    """
    # Initializations
    midpoint = geometry_util.midpoint(branch1[idx1], branch2[idx2])
    b1 = branch1 - midpoint
    b2 = branch2 - midpoint

    # Main
    dot5 = np.dot(tangent(b1, idx1, 5), tangent(b2, idx2, 5))
    dot10 = np.dot(tangent(b1, idx1, 10), tangent(b2, idx2, 10))
    dot20 = np.dot(tangent(b1, idx1, 20), tangent(b2, idx2, 20))
    return min(dot5, min(dot10, dot20))


def tangent(branch, idx, depth):
    """
    Computes tangent vector of "branch" after indexing from "idx".

    Parameters
    ----------
    branch : numpy.ndarray
        xyz coordinates that form a path.
    idx : int
        Index of a row in "branch".

    Returns
    -------
    numpy.ndarray
        Tangent vector of "branch".
    """
    end = min(idx + depth, len(branch))
    return geometry_util.tangent(branch[idx:end])
