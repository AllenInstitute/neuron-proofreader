"""
Created on Sat Nov 15 9:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code for processing geometric data.

"""

from itertools import chain
from scipy.interpolate import UnivariateSpline
from scipy.linalg import svd
from scipy.spatial.distance import euclidean
from tqdm import tqdm

import numpy as np
import rustworkx as rx


def fit_spline_1d(pts, k=3, s=None):
    """
    Fits a spline to 1D curve.

    Parameters
    ----------
    pts : numpy.ndarray
        Points to be smoothed.
    k : int, optional
        Degree of the spline. Default is 3.
    s : float, optional
        Parameter that controls the smoothness of the spline. Default is None.

    Returns
    -------
    UnivariateSpline
        Spline fit to the given points.
    """
    t = np.linspace(0, 1, len(pts))
    s = len(pts) / s if s else len(pts) / 15
    return UnivariateSpline(t, pts, k=k, s=s)


def fit_spline_3d(pts, k=3, s=None):
    """
    Fits a cubic spline to an array containing xyz coordinates.

    Parameters
    ----------
    pts : numpy.ndarray
        Array of xyz coordinates to be smoothed.
    k : int, optional
        Degree of the spline. Default is 3.
    s : float, optional
        Parameter that controls the smoothness of the spline. Default is None.

    Returns
    -------
    spline_x : UnivariateSpline
        Spline fit to x-coordinates of the given points.
    spline_y : UnivariateSpline
        Spline fit to the y-coordinates of the given points.
    spline_z : UnivariateSpline
        Spline fit to the z-coordinates of the given points.
    """
    spline_x = fit_spline_1d(pts[:, 0], k=k, s=s)
    spline_y = fit_spline_1d(pts[:, 1], k=k, s=s)
    spline_z = fit_spline_1d(pts[:, 2], k=k, s=s)
    return spline_x, spline_y, spline_z


def path_length(curve):
    """
    Computes the Euclidean length of the given curve.

    Parameters
    ----------
    curve : numpy.ndarray
        Array of points that form an n-d curve.

    Returns
    -------
    float
        Euclidean length of the given curve.
    """
    return np.linalg.norm(np.diff(curve, axis=0), axis=1).sum()


def resample_curve_1d(pts, n_pts=None, s=None):
    """
    Smooths a 1D curve by fitting a spline and resampling it.

    Parameters
    ----------
    n_pts : int or None, optional
        Number of points to resample.
    s : float, optional
        Parameter that controls the smoothness of the spline. Default is None.

    Returns
    -------
    numpy.ndarray
        Resampled points.
    """
    # Fit spline
    dt = max(n_pts or len(pts), 5)
    k = min(3, len(pts) - 1)

    # Check for degenerate case
    if k == 0:
        return np.repeat(pts, n_pts, axis=0)

    # Resample points
    t = np.linspace(0, 1, dt)
    spline = fit_spline_1d(pts, k=k, s=s)
    return spline(t)


def resample_curve_3d(pts, n_pts=None, s=None):
    """
    Smooths an Nx3 array of points by fitting a spline. Points are assumed
    to form a continuous curve that does not have any branching points.

    Parameters
    ----------
    pts: numpy.ndarray
        Array of points to be smoothed.
    n_pts : int
        Number of points to be sampled from the spline. Default is None.
    s : float
        Parameter that controls the smoothness of the spline. Default is None.

    Returns
    -------
    pts : numpy.ndarray
        Resampled points.
    """
    # Compute spline parameters
    dt = max(n_pts or len(pts), 5)
    k = min(3, len(pts) - 1)

    # Check for degenerate case
    if k == 0:
        return np.repeat(pts, n_pts, axis=0)

    # Fit spline
    spline_x, spline_y, spline_z = fit_spline_3d(pts, k=k, s=s)

    # Resample points
    t = np.linspace(0, 1, dt)
    pts = np.column_stack(
        (
            spline_x(t).astype(np.float32),
            spline_y(t).astype(np.float32),
            spline_z(t).astype(np.float32),
        )
    )
    return pts


# --- Fragment Filtering ---
def remove_doubles(graph, max_cable_length, max_nodes=1000, search_radius=15):
    """
    Removes connected components from the graph that are likely doubles
    caused by image ghosting artifacts.

    A component is a double if it is small (at most "max_nodes" nodes, no
    branching nodes, cable length at most "max_cable_length") and most of its
    nodes project onto a single other component at a nearly constant offset.
    See "is_double" for the exact criteria. All steps are vectorized so the
    runtime is dominated by a single batched KD-tree query over the nodes of
    the candidate components.

    Parameters
    ----------
    graph : SkeletonGraph
        Graph to be searched for doubles.
    max_cable_length : float
        Maximum cable length of connected components to be searched.
    max_nodes : int, optional
        Maximum number of nodes of connected components to be searched.
        Default is 1000.
    search_radius : float, optional
        Radius (in microns) used to search for nodes of other components.
        Default is 15.
    """
    # Label nodes by connected component
    nodes, labels, sizes = component_labels(graph)
    if len(sizes) == 0:
        return

    # Find candidate components (small, unbranched, short)
    is_candidate = find_candidate_components(
        graph, labels, sizes, max_nodes, max_cable_length
    )
    candidate_nodes = nodes[is_candidate[labels[nodes]]]
    if len(candidate_nodes) == 0:
        return

    # Project each candidate node onto the nearest other component
    query_nodes, hit_nodes, hit_dists = nearest_other_component(
        graph, candidate_nodes, search_radius
    )

    # Check doubles criteria
    is_dbl = is_double(graph, labels, sizes, query_nodes, hit_nodes, hit_dists)
    nodes_to_remove = nodes[is_dbl[labels[nodes]]]
    if graph.verbose:
        print(
            f"Filter Doubles: removed {is_dbl.sum()} of "
            f"{is_candidate.sum()} candidate components"
        )

    # Update graph
    graph.remove_nodes(nodes_to_remove)


def component_labels(graph):
    """
    Labels every node with the index of its connected component.

    Parameters
    ----------
    graph : SkeletonGraph
        Graph to be labeled.

    Returns
    -------
    nodes : numpy.ndarray
        All node IDs in the graph.
    labels : numpy.ndarray
        Array indexed by node ID that gives the component index, or -1 for
        IDs that are not in the graph.
    sizes : numpy.ndarray
        Number of nodes in each component.
    """
    components = rx.connected_components(graph)
    n_components = len(components)
    sizes = np.fromiter(map(len, components), np.int64, n_components)
    nodes = np.fromiter(chain.from_iterable(components), np.int64, sizes.sum())

    n_ids = max(graph.node_indices(), default=-1) + 1
    labels = np.full(n_ids, -1, dtype=np.int64)
    labels[nodes] = np.repeat(np.arange(n_components), sizes)
    return nodes, labels, sizes


def find_candidate_components(
    graph, labels, sizes, max_nodes, max_cable_length
):
    """
    Finds components that are small enough to possibly be doubles.

    Parameters
    ----------
    graph : SkeletonGraph
        Graph to be searched.
    labels : numpy.ndarray
        Component index of each node, see "component_labels".
    sizes : numpy.ndarray
        Number of nodes in each component.
    max_nodes : int
        Maximum number of nodes of a candidate component.
    max_cable_length : float
        Maximum cable length of a candidate component.

    Returns
    -------
    numpy.ndarray
        Boolean array indexed by component index.
    """
    n_components = len(sizes)
    edges, edge_lengths = graph.edge_lengths()

    # Components containing a branching node (degree > 2)
    degree = np.bincount(edges.ravel(), minlength=len(labels))
    has_branch = np.zeros(n_components, dtype=bool)
    has_branch[labels[degree > 2]] = True

    # Cable length of each component (sum of its edge lengths)
    cable_lengths = np.bincount(
        labels[edges[:, 0]], weights=edge_lengths, minlength=n_components
    )

    is_small = sizes <= max_nodes
    is_short = cable_lengths <= max_cable_length
    return is_small & ~has_branch & is_short


def nearest_other_component(
    graph, query_nodes, radius, k=32, chunk_size=500000
):
    """
    Finds, for each query node, the nearest node that belongs to a different
    connected component and lies within "radius".

    Uses a batched k-nearest-neighbor query, then falls back to an exact ball
    query for the rare nodes whose k nearest neighbors all lie within the
    radius and in their own component. The result is therefore exact
    regardless of "k".

    Parameters
    ----------
    graph : SkeletonGraph
        Graph with "kdtree", "node_xyz", and "node_component_id" set.
    query_nodes : numpy.ndarray
        Node IDs to query.
    radius : float
        Search radius (inclusive).
    k : int, optional
        Number of nearest neighbors requested per node. Default is 32.
    chunk_size : int, optional
        Number of query nodes processed per KD-tree call, which bounds peak
        memory. Default is 500000.

    Returns
    -------
    query_nodes : numpy.ndarray
        Query nodes that have a neighbor in another component.
    hit_nodes : numpy.ndarray
        Nearest such neighbor for each returned query node.
    hit_dists : numpy.ndarray
        Distance to that neighbor.
    """
    cid = graph.node_component_id
    xyz = graph.node_xyz
    n_ids = len(xyz)
    k = min(k, n_ids)

    # Set progress bar
    chunks = range(0, len(query_nodes), chunk_size)
    if graph.verbose and len(chunks) > 1:
        chunks = tqdm(chunks, desc="Filter Doubles")

    # Search KD-Tree
    out_query, out_hit, out_dist = list(), list(), list()
    for start in chunks:
        # Query k nearest neighbors within radius (bound is exclusive, so
        # nudge it up to match query_ball_point's inclusive radius)
        nodes = query_nodes[start : start + chunk_size]
        dists, idxs = graph.kdtree.query(
            xyz[nodes],
            k=k,
            distance_upper_bound=np.nextafter(radius, np.inf),
            workers=-1,
        )
        dists = dists.reshape(len(nodes), k)
        idxs = idxs.reshape(len(nodes), k)

        # Missing neighbors are reported as index n_ids with distance inf
        is_found = idxs < n_ids
        idxs_safe = np.minimum(idxs, n_ids - 1)
        is_other = is_found & (cid[idxs_safe] != cid[nodes][:, None])

        # Nearest neighbor in another component (neighbors are sorted)
        has_other = is_other.any(axis=1)
        rows = np.flatnonzero(has_other)
        cols = is_other[rows].argmax(axis=1)
        out_query.append(nodes[rows])
        out_hit.append(idxs[rows, cols])
        out_dist.append(dists[rows, cols])

        # Exact fallback for nodes whose k neighbors were all in their own
        # component while more may exist within the radius
        for i in np.flatnonzero(~has_other & is_found[:, -1]):
            result = _nearest_other_component_exact(graph, nodes[i], radius)
            if result is not None:
                out_query.append(nodes[i : i + 1])
                out_hit.append(result[0])
                out_dist.append(result[1])

    return (
        np.concatenate(out_query),
        np.concatenate(out_hit),
        np.concatenate(out_dist),
    )


def _nearest_other_component_exact(graph, node, radius):
    """
    Finds the nearest node to "node" in another component within "radius"
    using a ball query, see "nearest_other_component".

    Returns
    -------
    Tuple[numpy.ndarray] or None
        Length-1 arrays (hit node, distance), or None if no such node.
    """
    xyz = graph.node_xyz
    idxs = np.array(graph.kdtree.query_ball_point(xyz[node], radius), int)
    idxs = idxs[graph.node_component_id[idxs] != graph.node_component_id[node]]
    if len(idxs) == 0:
        return None

    dists = np.linalg.norm(xyz[idxs].astype(np.float64) - xyz[node], axis=1)
    j = np.argmin(dists)
    return idxs[j : j + 1], dists[j : j + 1]


def is_double(graph, labels, sizes, query_nodes, hit_nodes, hit_dists):
    """
    Determines which components are doubles of another component.

    A component is a double if more than 10 of its nodes project onto the
    same other component and either
        (i) more than 60% of its nodes do so with a standard deviation of
            the projection distance below 2, or
        (ii) more than 80% of its nodes do so with a standard deviation
             below 2.5.

    Parameters
    ----------
    graph : SkeletonGraph
        Graph to be searched.
    labels : numpy.ndarray
        Component index of each node, see "component_labels".
    sizes : numpy.ndarray
        Number of nodes in each component.
    query_nodes : numpy.ndarray
        Nodes that were projected onto another component.
    hit_nodes : numpy.ndarray
        Node in another component that each query node projected onto.
    hit_dists : numpy.ndarray
        Projection distance for each query node.

    Returns
    -------
    numpy.ndarray
        Boolean array indexed by component index.
    """
    # Group projections by (component, hit component) pairs
    cid = graph.node_component_id
    keys = labels[query_nodes] * (cid.max() + 1) + cid[hit_nodes]
    order = np.argsort(keys, kind="stable")
    keys, dists = keys[order], hit_dists[order]
    starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    counts = np.diff(np.r_[starts, len(keys)])

    # Standard deviation of projection distances within each group
    means = np.add.reduceat(dists, starts) / counts
    sq_devs = (dists - np.repeat(means, counts)) ** 2
    stds = np.sqrt(np.add.reduceat(sq_devs, starts) / counts)

    # Check doubles criteria per group
    component = labels[query_nodes[order[starts]]]
    percent_hit = counts / sizes[component]
    is_tight = (percent_hit > 0.6) & (stds < 2)
    is_loose = (percent_hit > 0.8) & (stds < 2.5)
    is_dbl_group = (counts > 10) & (is_tight | is_loose)

    # A component is a double if any of its groups is
    is_dbl = np.zeros(len(sizes), dtype=bool)
    is_dbl[component[is_dbl_group]] = True
    return is_dbl


# --- Miscellaneous ---
def closest_pair(pts1, pts2):
    """
    Find the indices of the closest pair of points between two point sets.

    Parameters
    ----------
    pts1 : numpy.ndarray
        First set of points with shape (N, D).
    pts2 : numpy.ndarray
        Second set of points with shape (M, N).

    Returns
    -------
    (i, j) : Tuple[int]
        Indices such that "pts1[i]" and "pts2[j]" are the closest pair of
        points between the two sets.
    """
    diff = pts1[:, None, :] - pts2[None, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    return np.unravel_index(np.argmin(dists_sq), dists_sq.shape)


def compute_svd(pts):
    """
    Compute singular value decomposition (svd) of an NxD array where N is the
    number of points and D is the dimension of the space.

    Parameters
    ----------
    pts : numpy.ndarray
        Array containing data points.

    Returns
    -------
    numpy.ndarry
        Unitary matrix having left singular vectors as columns. Of shape
        (N, N) or (N, min(N, D)), depending on full_matrices.
    numpy.ndarray
        Singular values, sorted in non-increasing order. Of shape (K,), with
        K = min(N, D).
    numpy.ndarray
        Unitary matrix having right singular vectors as rows. Of shape (D, D)
        or (K, D) depending on full_matrices.
    """
    return svd(pts - np.mean(pts, axis=0))


def make_digital_line(p1, p2):
    """
    Generates integer voxel coordinates along a 3D line between p1 and p2.

    Parameters
    ----------
    p1 : Tuple[int]
        Start coordinate of line.
    p2 : Tuple[int]
        End coordinate of line.

    Returns
    -------
    line : numpy.ndarray
        Voxel coordinates representing the straight line between p1 and p2.
    """
    # Convert coordinates to arrays
    p1 = np.array(p1, dtype=int)
    p2 = np.array(p2, dtype=int)

    # Determine number of points
    diff = p2 - p1
    n = np.max(np.abs(diff))
    if n == 0:
        return p1[None, :]

    # Generate line
    t = np.linspace(0, 1, n + 1)
    line = np.round(p1 + np.outer(t, diff)).astype(int)
    return line


def make_digital_lines(p1s, p2s):
    """
    Vectorized version of "make_digital_line" for many segments at once.
    Produces exactly the voxels that calling make_digital_line(p1, p2) for
    each pair and concatenating would, including the numpy.linspace
    parameterization (t_k = k * (1 / n), t_n = 1).

    Parameters
    ----------
    p1s : ArrayLike
        Start coordinates with shape (n_segments, 3).
    p2s : ArrayLike
        End coordinates with shape (n_segments, 3).

    Returns
    -------
    numpy.ndarray
        Voxel coordinates of all segments, shape (n_voxels, 3).
    """
    p1s = np.asarray(p1s, dtype=int).reshape(-1, 3)
    p2s = np.asarray(p2s, dtype=int).reshape(-1, 3)
    if len(p1s) == 0:
        return np.zeros((0, 3), dtype=int)

    diffs = p2s - p1s
    n = np.max(np.abs(diffs), axis=1)
    counts = n + 1
    seg = np.repeat(np.arange(len(p1s)), counts)
    k = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)

    n_seg = n[seg]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = k * (1.0 / n_seg)
    t[k == n_seg] = 1.0  # linspace sets the endpoint exactly; covers n == 0
    return np.round(p1s[seg] + t[:, None] * diffs[seg]).astype(int)


def make_line(p1, p2, n_steps):
    """
    Generates a series of points representing a straight line between two 3D
    coordinates.

    Parameters
    ----------
    p1 : Tuple[float]
        Start coordinate of line.
    p2 : Tuple[float]
        End coordinate of line.
    n_steps : int
        Number of steps to interpolate between the two coordinates.

    Returns
    -------
    numpy.ndarray
        Coordinates representing the straight line between p1 and p2.
    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    t_steps = np.linspace(0, 1, n_steps)
    return np.array([(1 - t) * p1 + t * p2 for t in t_steps], dtype=int)


def make_voxels_connected(voxels):
    """
    Makes a list of voxels that form a discrete curve 27-connected.

    Parameters
    ----------
    voxels : List[Tuple[int]]
        List of voxel coordinates that form a discrete path.

    Returns
    -------
    voxels_out : numpy.ndarray
        List of voxels that is 27-connected.
    """
    voxels = np.asarray(voxels, dtype=int)
    voxels_out = []
    for a, b in zip(voxels[:-1], voxels[1:]):
        line = make_digital_line(a, b)
        if voxels_out:
            line = line[1:]
        voxels_out.extend(line)
    return np.array(voxels_out, dtype=int)


def midpoint(pt1, pt2):
    """
    Computes the midpoint between the two given points.

    Parameters
    ----------
    pt1 : numpy.ndarray
        N-dimensional coordinate.
    pt2 : numpy.ndarray
        N-dimensional coordinate.

    Returns
    -------
    numpy.ndarray
        Midpoint of "pt1" and "pt2".
    """
    return np.mean([pt1, pt2], axis=0)


def nearest_neighbor(pts, query_pt, return_index=False):
    """
    Finds the nearest neighbor in a list of 3D coordinates to a given target
    coordinate.

    Parameters
    ----------
    pts : numpy.ndarray
        3D coordinates to search for the nearest neighbor.
    query_pt : numpy.ndarray
        3D coordinate to query.
    return_index : bool, optional
        Indication of whether to return the index of the nearest neighbor.

    Returns
    -------
    best_pt : Tuple[float]
        Nearest neighbor in a list of 3D coordiantes to a given target.
    """
    pts = np.asarray(pts)
    dists = np.linalg.norm(pts - query_pt, axis=1)
    idx = np.argmin(dists)
    return idx if return_index else pts[idx]


def tangent(pts):
    """
    Computes the tangent vector at a given point or along a curve defined by
    an array of points.

    Parameters
    ----------
    pts : numpy.ndarray
        Array containing either two coordinates or an arbitrary number of
        defining a curve.

    Returns
    -------
    numpy.ndarray
        Tangent vector at the specified point or along the curve.
    """
    if len(pts) == 1:
        tangent_vec = np.zeros((3))
    elif len(pts) == 2:
        tangent_vec = (pts[1] - pts[0]) / (euclidean(pts[1], pts[0]) + 1e-5)
    else:
        _, _, VT = compute_svd(pts)
        tangent_vec = VT[0]
        if np.dot(tangent_vec, tangent([pts[0], pts[-1]])) < 0:
            tangent_vec *= -1
    return tangent_vec / (np.linalg.norm(tangent_vec) + 1e-5)
