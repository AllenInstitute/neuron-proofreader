"""
Created on Fri Sep 26 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Feature extraction for split correction using Arborist tree encodings.
Extends the existing split pipeline so that each batch carries one
TreeSample per proposal alongside the image patches and graph features.

Ported from arborist-experiments/split-correction-arborist.

"""

from queue import Queue
from threading import Thread

import numpy as np
from scipy.interpolate import splev, splprep

from arborist.data.datasets import build_tree_sample
from arborist.skeleton_graph import SkeletonGraph
from arborist.utils.graph_utils import topological_decomposition

from neuron_proofreader.split_proofreading.split_datasets import FragmentsDataset
from neuron_proofreader.split_proofreading.split_feature_extraction import HeteroGraphData


# --- Bridge geometry helpers ---

def bridge_points(start, end, spacing):
    """
    Samples points along the straight segment from start to end so that
    consecutive points are approximately spacing apart.
    """
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    length = np.linalg.norm(end - start)
    n = max(1, int(round(length / spacing))) if spacing > 0 else 1
    t = np.linspace(0.0, 1.0, n + 1)[1:]
    return start + t[:, None] * (end - start)


def tail_path(graph, node, max_length):
    """
    Walks from a leaf into its fragment along the unbranched chain up to
    max_length microns.
    """
    if graph.degree(node) != 1:
        return [node]
    path, length = [node], 0.0
    prev, curr = None, node
    while True:
        nbrs = [v for v in graph.neighbors(curr) if v != prev]
        if len(nbrs) != 1 and curr != node:
            break
        if not nbrs:
            break
        nxt = nbrs[0]
        step = graph.dist(curr, nxt)
        if length + step > max_length:
            break
        length += step
        path.append(nxt)
        prev, curr = curr, nxt
    return path[::-1]


def spline_bridge(graph, i, j, spacing, context=16.0):
    """
    Samples a bridge between two proposal endpoints from a cubic spline
    fitted through the tails of both fragments.
    """
    tail_i = tail_path(graph, i, context)
    tail_j = tail_path(graph, j, context)[::-1]
    xyz = np.asarray(graph.node_xyz[np.array(tail_i + tail_j)], dtype=float)

    keep = np.r_[True, np.linalg.norm(np.diff(xyz, axis=0), axis=1) > 1e-9]
    xyz = xyz[keep]
    idx_i = int(np.sum(keep[: len(tail_i)])) - 1
    idx_j = idx_i + 1
    if len(xyz) < 3 or idx_j >= len(xyz):
        return _linear_bridge(graph.node_xyz[i], graph.node_xyz[j], spacing)

    u = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
    k = min(3, len(xyz) - 1)
    try:
        tck, _ = splprep(xyz.T, u=u, k=k, s=0)
    except (ValueError, TypeError):
        return _linear_bridge(graph.node_xyz[i], graph.node_xyz[j], spacing)

    u_i, u_j = u[idx_i], u[idx_j]
    u_mid = 0.5 * (u_i + u_j)
    n_half = max(1, int(round(0.5 * (u_j - u_i) / spacing))) if spacing > 0 else 1

    def sample(u_from, u_to, endpoint_xyz):
        us = np.linspace(u_from, u_to, n_half + 1)[1:]
        pts = np.column_stack(splev(us, tck))
        pts[-1] = endpoint_xyz
        return pts

    root = np.array(splev(u_mid, tck), dtype=float)
    pts_i = sample(u_mid, u_i, graph.node_xyz[i])
    pts_j = sample(u_mid, u_j, graph.node_xyz[j])
    return root, pts_i, pts_j


def _linear_bridge(xyz_i, xyz_j, spacing):
    xyz_i = np.asarray(xyz_i, dtype=float)
    xyz_j = np.asarray(xyz_j, dtype=float)
    root = 0.5 * (xyz_i + xyz_j)
    return root, bridge_points(root, xyz_i, spacing), bridge_points(root, xyz_j, spacing)


# --- Rooted subgraph ---

def proposal_rooted_subgraph(
    graph,
    proposal,
    depth,
    node_spacing=None,
    bridge_mode="linear",
    spline_context=16.0,
):
    """
    Builds a SkeletonGraph rooted at the center of a proposal.

    Node 0 is a virtual root at the bridge midpoint. Two bridge paths lead
    from the root to each proposal endpoint, then the skeleton is expanded
    outward until the path length from the root reaches depth.

    Parameters
    ----------
    graph : ProposalGraph
        Fragments graph.
    proposal : Frozenset[int]
        Pair of node IDs forming the proposal.
    depth : float
        Maximum path length (microns) from the root to any subgraph node.
    node_spacing : float, optional
        Bridge node spacing. Defaults to graph.node_spacing.
    bridge_mode : {"linear", "spline"}, optional
        Bridge geometry. Default is "linear".
    spline_context : float, optional
        Fragment tail length (microns) used for spline fitting. Default is 16.

    Returns
    -------
    subgraph : SkeletonGraph
    endpoints : Tuple[int, int]
        Subgraph node IDs of the two proposal endpoints.
    """
    if bridge_mode not in ("linear", "spline"):
        raise ValueError(f"Unknown bridge_mode: {bridge_mode!r}")
    i, j = tuple(proposal)
    spacing = node_spacing or getattr(graph, "node_spacing", None) or 1.0
    radius_feats = getattr(graph, "node_feats", {}).get("radius")

    def radius_of(node):
        return float(radius_feats[node]) if radius_feats is not None else 1.0

    if bridge_mode == "spline":
        center, pts_i, pts_j = spline_bridge(graph, i, j, spacing, spline_context)
    else:
        center, pts_i, pts_j = _linear_bridge(
            graph.node_xyz[i], graph.node_xyz[j], spacing
        )

    subgraph = SkeletonGraph(anisotropy=graph.anisotropy, node_spacing=spacing)
    subgraph.add_node(None)
    xyz_list = [center]
    radius_list = [(radius_of(i) + radius_of(j)) / 2.0]

    node_mapping = {}
    start_dist = {}
    for endpoint, pts in ((i, pts_i), (j, pts_j)):
        r_center, r_end = radius_list[0], radius_of(endpoint)
        prev, prev_xyz, walked = 0, center, 0.0
        for k, xyz in enumerate(pts):
            idx = subgraph.add_node(None)
            subgraph.add_edge(prev, idx, None)
            frac = (k + 1) / len(pts)
            xyz_list.append(xyz)
            radius_list.append((1 - frac) * r_center + frac * r_end)
            walked += float(np.linalg.norm(xyz - prev_xyz))
            prev, prev_xyz = idx, xyz
        node_mapping[endpoint] = prev
        start_dist[endpoint] = round(walked, 9)

    visited = {i, j}
    queue = [(i, start_dist[i]), (j, start_dist[j])]
    while queue:
        u, dist_u = queue.pop()
        for v in graph.neighbors(u):
            dist_v = dist_u + graph.dist(u, v)
            if v not in visited and dist_v < depth:
                idx = subgraph.add_node(None)
                subgraph.add_edge(node_mapping[u], idx, None)
                node_mapping[v] = idx
                xyz_list.append(np.asarray(graph.node_xyz[v], dtype=float))
                radius_list.append(radius_of(v))
                queue.append((v, dist_v))
                visited.add(v)

    subgraph.node_xyz = np.array(xyz_list, dtype=np.asarray(graph.node_xyz).dtype)
    subgraph.node_feats = {"radius": np.array(radius_list, dtype=float)}
    return subgraph, (node_mapping[i], node_mapping[j])


# --- Feature extractor ---

class ProposalTreeFeatureExtractor:
    """
    Converts a proposal into an Arborist TreeSample. One instance can be
    reused across all subgraphs.

    Parameters
    ----------
    max_depth : float, optional
        Subgraph depth in microns from the proposal center. Default is 100.
    min_curve_len : int, optional
        Curves shorter than this are zero-padded. Should match
        curve_segment_len in the Arborist config. Default is 10.
    node_spacing : float, optional
        Bridge node spacing. Defaults to graph.node_spacing.
    bridge_mode : {"linear", "spline"}, optional
        Bridge geometry. Default is "linear".
    spline_context : float, optional
        Fragment tail length (microns) used for spline fitting. Default is 16.
    transform : callable, optional
        Applied to each curve's raw xyz array before differencing.
    graph_transform : callable, optional
        Applied to the full node_xyz array before decomposition.
    """

    def __init__(
        self,
        max_depth=100,
        min_curve_len=10,
        node_spacing=None,
        bridge_mode="linear",
        spline_context=16.0,
        transform=None,
        graph_transform=None,
    ):
        self.max_depth = max_depth
        self.min_curve_len = min_curve_len
        self.node_spacing = node_spacing
        self.bridge_mode = bridge_mode
        self.spline_context = spline_context
        self.transform = transform
        self.graph_transform = graph_transform

    @property
    def config(self):
        return {
            "max_depth": self.max_depth,
            "min_curve_len": self.min_curve_len,
            "node_spacing": self.node_spacing,
            "bridge_mode": self.bridge_mode,
            "spline_context": self.spline_context,
            "transform": type(self.transform).__name__ if self.transform else None,
            "graph_transform": (
                type(self.graph_transform).__name__ if self.graph_transform else None
            ),
        }

    def __call__(self, graph, proposal):
        """
        Parameters
        ----------
        graph : ProposalGraph
            Fragments graph containing the proposal endpoints.
        proposal : Frozenset[int]
            Pair of node IDs forming the proposal.

        Returns
        -------
        TreeSample
            root_curve_indices marks the two curves incident to the root
            (one per side of the proposal). proposal_length holds the
            Euclidean distance between the proposal endpoints.
        """
        subgraph, _ = proposal_rooted_subgraph(
            graph,
            proposal,
            self.max_depth,
            node_spacing=self.node_spacing,
            bridge_mode=self.bridge_mode,
            spline_context=self.spline_context,
        )
        if self.graph_transform:
            subgraph.node_xyz = self.graph_transform(subgraph.node_xyz)
        _, paths, topo_edge_index = topological_decomposition(subgraph)

        curves = []
        for path in paths:
            xyz = subgraph.node_xyz[path].copy()
            xyz -= xyz[0]
            xyz[-1:0:-1] -= xyz[-2::-1]
            if len(xyz) < self.min_curve_len:
                pad = np.zeros(
                    (self.min_curve_len - len(xyz), 3), dtype=xyz.dtype
                )
                xyz = np.concatenate([xyz, pad], axis=0)
            curves.append(xyz)

        sample = build_tree_sample(curves, topo_edge_index)
        sample.proposal_length = float(graph.dist(*tuple(proposal)))
        return sample


# --- Arborist-aware data classes ---

class ArboristHeteroGraphData(HeteroGraphData):
    """
    HeteroGraphData that carries TreeSamples through to model inputs under
    the "tree_samples" key.
    """

    def __init__(self, features):
        super().__init__(features)
        self.tree_samples = getattr(features, "tree_samples", None)

    def get_inputs(self):
        inputs = super().get_inputs()
        inputs["tree_samples"] = self.tree_samples
        return inputs


class ArboristSplitDataset(FragmentsDataset):
    """
    FragmentsDataset that attaches one TreeSample per proposal to each batch.

    Parameters
    ----------
    fragments_graph : ProposalGraph
        Graph with proposals generated.
    img_config : ImageConfig
        Image configuration.
    **tree_kwargs
        Forwarded to ProposalTreeFeatureExtractor (max_depth, min_curve_len,
        node_spacing, bridge_mode, spline_context, transform, graph_transform).
    """

    def __init__(
        self,
        fragments_graph,
        img_config,
        batch_size=32,
        gt_path=None,
        prefetch=2,
        **tree_kwargs,
    ):
        super().__init__(fragments_graph, img_config, batch_size=batch_size, gt_path=gt_path, prefetch=prefetch)
        self.tree_extractor = ProposalTreeFeatureExtractor(**tree_kwargs)

    def __iter__(self):
        queue = Queue(maxsize=self.prefetch)
        sentinel = object()

        def producer():
            try:
                sampler = self.get_sampler()
                subgraph = next(sampler, None)
                reads = self.feature_extractor.issue_reads(subgraph) if subgraph else None
                while subgraph is not None:
                    next_subgraph = next(sampler, None)
                    next_reads = (
                        self.feature_extractor.issue_reads(next_subgraph)
                        if next_subgraph is not None else None
                    )
                    features = self.feature_extractor(subgraph, reads)
                    idx_to_id = features.proposal_index_mapping.idx_to_id
                    features.tree_samples = [
                        self.tree_extractor(self.graph, idx_to_id[idx])
                        for idx in range(len(idx_to_id))
                    ]
                    queue.put(ArboristHeteroGraphData(features))
                    subgraph, reads = next_subgraph, next_reads
            except Exception as e:
                queue.put(e)
            finally:
                queue.put(sentinel)

        Thread(target=producer, daemon=True).start()
        while True:
            item = queue.get()
            if item is sentinel:
                break
            if isinstance(item, Exception):
                raise item
            yield item
