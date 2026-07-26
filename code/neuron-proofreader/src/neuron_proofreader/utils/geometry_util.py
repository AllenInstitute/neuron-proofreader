"""
Created on Sat Nov 15 9:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code for processing geometric data.

"""

from collections import defaultdict
from scipy.interpolate import UnivariateSpline
from scipy.linalg import svd
from scipy.spatial.distance import euclidean
from sklearn.decomposition import PCA
from tqdm import tqdm

import networkx as nx
import numpy as np
import torch


# --- Curve Utils ---
def max_l2_error(curve1, curve2):
    """
    Computes maximum pointwise L2 error.

    Parameters
    ----------
    curve1 : numpy.ndarray
        Ground truth curve.
    curve2 : numpy.ndarray
        Reconstruction curve.

    Returns
    -------
    float
        Maximum Euclidean error.
    """
    assert curve1.shape == curve2.shape, "Curves have different number of pts"
    return np.linalg.norm(curve1 - curve2, axis=1).max()


def curves_pca_projection(curve1, curve2):
    """
    Projects two 3D curves into a shared PCA coordinate system.

    Parameters
    ----------
    curve1 : numpy.ndarray
        First curve with shape ``(N, 3)``.
    curve2 : numpy.ndarray
        Second curve with shape ``(M, 3)``.

    Returns
    -------
    tuple
        Two projected curves with shape ``(N, 2)`` and ``(M, 2)``.
    """
    # Compute PCA components
    pts = np.vstack([curve1, curve2])
    center = pts.mean(axis=0)

    pca = PCA(n_components=3)
    pca.fit(pts - center)

    # Compute projections
    curve1_proj = pca.transform(curve1 - center)
    curve2_proj = pca.transform(curve2 - center)
    return curve1_proj[:, :2], curve2_proj[:, :2]


def curve_principal_direction(curve):
    """
    Computes the principal direction of a 3D curve using PCA.

    Parameters
    ----------
    curve : numpy.ndarray
        Array with shape (N, 3) containing the 3D coordinates of the curve.

    Returns
    -------
    numpy.ndarray
        Unit vector of shape (3,) representing the principal direction of the
        curve.
    """
    curve_pca = PCA(n_components=1)
    curve_pca.fit(curve)
    direction = curve_pca.components_[0]
    if direction[2] < 0:
        direction = -direction
    return direction / np.linalg.norm(direction)


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


def reconstruct_diffs(diffs):
    """
    Reconstructs a curve from a sequence of offset vectors.

    Parameters
    ----------
    diffs : numpy.ndarray or torch.Tensor
        Array representing the differences between consecutive points.

    Returns
    -------
    numpy.ndarray
        Reconstructed curve.
    """
    if isinstance(diffs, torch.Tensor):
        start = torch.zeros(1, 3, device=diffs.device, dtype=diffs.dtype)
        return torch.cat([start, start + torch.cumsum(diffs, dim=0)], dim=0)
    else:
        start = np.zeros((1, 3))
        return np.concatenate(
            [start, start + np.cumsum(diffs, axis=0)], axis=0
        )


def rmse(curve1, curve2):
    """
    Computes Root Mean Squared Error (RMSE) between two curves.

    Parameters
    ----------
    curve1 : numpy.ndarray

    """
    return np.sqrt(np.mean(np.sum((curve1 - curve2) ** 2, axis=1)))


# --- Fragment Filtering ---
def remove_doubles(graph, max_cable_length):
    """
    Removes connected components from the graph that are likely doubles
    caused by image ghosting artifacts.

    Parameters
    ----------
    graph : SkeletonGraph
        Graph to be searched for doubles.
    max_cable_length : float
        Maximum cable length of connected components to be searched.
    """
    # Set progress bar
    iterator = nx.connected_components(graph)
    if graph.verbose:
        total = nx.number_connected_components(graph)
        iterator = tqdm(iterator, total=total, desc="Filter Doubles")

    # Search graph
    branching_nodes = set(graph.branching_nodes())
    nodes_to_remove = list()
    for nodes in iterator:
        # Check if component is obviously too big
        if len(nodes) > 1000:
            continue

        # Check for branching node
        if branching_nodes.intersection(nodes):
            continue

        # Check cable length
        nodes = list(nodes)
        length = graph.cable_length(max_depth=max_cable_length, root=nodes[0])
        if length > max_cable_length:
            continue

        # Check doubles criteria
        if is_double(graph, nodes):
            nodes_to_remove.extend(nodes)

    # Update graph
    graph.remove_nodes(nodes_to_remove)


def is_double(graph, nodes):
    """
    Determines if the connected component corresponding to "nodes" is a double
    of another connected component.

    Paramters
    ---------
    graph : SkeletonGraph
        Graph to be searched.
    nodes : List[int]
        Nodes corresponding to a single connected component.

    Returns
    -------
    bool
        True if the component is a double; otherwise, False.
    """
    # Compute projection distances
    cid = graph.node_component_id[nodes[0]]
    cid_to_dists = defaultdict(list)
    for i in nodes:
        # Find nearest neighbor
        idxs = np.array(graph.kdtree.query_ball_point(graph.node_xyz[i], 15))
        idxs = idxs[graph.node_component_id[idxs] != cid]
        if len(idxs) > 0:
            idx = nearest_neighbor(
                graph.node_xyz[idxs], graph.node_xyz[i], return_index=True
            )

            # Store distance
            j = idxs[idx]
            cid_to_dists[graph.node_component_id[j]].append(graph.dist(i, j))

    # Determine if double
    for dists in cid_to_dists.values():
        if len(dists) > 10:
            percent_hit = len(dists) / len(nodes)
            if percent_hit > 0.6 and np.std(dists) < 2:
                return True
            elif percent_hit > 0.8 and np.std(dists) < 2.5:
                return True
    return False


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
