"""
Created on Fri April 11 11:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Feature extraction for split correction. Each proposal gets an image patch
and an Arborist TreeSample; a batch of proposals is packed with its
proposal/branch adjacency into a heterogeneous graph for the GNN.

"""

from concurrent.futures import as_completed, ThreadPoolExecutor
from torch_geometric.data import HeteroData

import numpy as np
import torch
from scipy.interpolate import splev, splprep

from arborist.data.datasets import build_tree_sample
from arborist.skeleton_graph import SkeletonGraph
from arborist.utils.graph_utils import (
    edges_to_line_graph,
    topological_decomposition,
)
from neuron_proofreader.machine_learning.image_dataloader import (
    ProposalPatchLoader,
)
from neuron_proofreader.utils import geometry_util, img_util
from neuron_proofreader.utils.ml_util import TensorDict


# --- Image patches ---
class ProposalPatchExtractor:
    """
    Reads an image patch centered on each proposal and renders the proposal
    and its two incident branches into a mask as a second channel.

    Attributes
    ----------
    max_workers : int
        Number of threads that turn image patches into features. Image reads
        are not bound by this: every read of a batch is issued up front and
        runs on TensorStore's own thread pool, so these threads only do CPU
        work. Keeping this small matters because the threads hold the GIL
        while annotating masks, and the GIL is shared with the thread that
        launches the GPU forward pass.
    """

    max_workers = 6

    def __init__(self, graph, img_config, padding=50):
        """
        Parameters
        ----------
        graph : ProposalGraph
            Graph with proposals generated.
        img_config : ImageConfig
            Image configuration.
        padding : int, optional
            Voxels added around the proposal endpoints when reading the
            patch. Default is 50.
        """
        self.graph = graph
        self.mask_kernel_size = img_config.mask_kernel_size
        self.patch_shape = img_config.patch_shape
        self.patch_loader = ProposalPatchLoader(graph, img_config, padding)
        self.transform = img_config.transform

        # Patches are resized with torch ops from "max_workers" Python
        # threads. Each thread would otherwise spin up its own intra-op
        # thread pool (max_workers x cores threads), which starves the rest
        # of the pipeline; one thread per op is fastest here.
        torch.set_num_threads(1)

    def issue_reads(self, subgraph):
        """
        Starts the image read of every proposal in the subgraph. The reads
        run on TensorStore's thread pool, so issuing them early (ideally
        while the previous batch is still being processed) hides most of the
        read latency.

        Returns
        -------
        Dict[Frozenset[int], tuple]
            Pending read per proposal, see "ProposalPatchLoader.read_async".
        """
        return {p: self.patch_loader.read_async(p) for p in subgraph.proposals}

    def __call__(self, subgraph, reads=None):
        """
        Extracts the input patch of every proposal in the subgraph.

        Parameters
        ----------
        subgraph : ProposalComputationGraph
            Batch of proposals.
        reads : Dict[Frozenset[int], tuple], optional
            Pending image reads from "issue_reads". Default is None, in which
            case they are issued here.

        Returns
        -------
        Dict[Frozenset[int], numpy.ndarray]
            Float16 array with shape (2, *patch_shape) per proposal, where
            channel 0 is the image and channel 1 the mask.
        """
        reads = reads if reads is not None else self.issue_reads(subgraph)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            pending = {
                executor.submit(self.extract, p, read): p
                for p, read in reads.items()
            }
            return {pending[t]: t.result() for t in as_completed(pending)}

    def extract(self, proposal, read):
        img, offset = self.patch_loader(proposal, read=read)
        mask = np.zeros(img.shape, dtype=np.float32)
        for node in proposal:
            voxels = self.branch_voxels(node, offset, img.shape)
            img_util.annotate_voxels(mask, voxels, self.mask_kernel_size)
        i, j = (self.graph.node_local_voxel(n, offset) for n in proposal)
        img_util.annotate_voxels(
            mask, geometry_util.make_digital_line(i, j), self.mask_kernel_size
        )

        img = img_util.resize(img, self.patch_shape)
        mask = img_util.resize_nearest(mask, self.patch_shape)
        patch = np.stack([img, mask], axis=0)
        if self.transform is not None:
            patch = self.transform(patch)
        # numpy's float16 cast is slow enough to dominate batch assembly
        return torch.from_numpy(patch).to(torch.float16).numpy()

    def branch_voxels(self, node, offset, shape):
        """
        Gets the voxels (in patch coordinates) of the branch containing the
        given node, clipped to the patch.
        """
        queue = [(node, self.graph.node_local_voxel(node, offset))]
        visited = {node}
        voxels = list()
        while queue:
            i, voxel_i = queue.pop()
            voxels.append(voxel_i)
            for j in self.graph.neighbors(i):
                voxel_j = self.graph.node_local_voxel(j, offset)
                if j not in visited and img_util.is_contained(voxel_j, shape):
                    queue.append((j, voxel_j))
                    visited.add(j)
        return geometry_util.make_voxels_connected(voxels)


# --- Tree samples ---
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
        # rustworkx returns neighbors in an order that changes between calls,
        # which would make the subgraph's node numbering, and so the curve
        # decomposition fed to the model, differ from one extraction to the next.
        for v in sorted(graph.neighbors(u)):
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


# --- Batch container ---
class HeteroGraphData(HeteroData):
    """
    One batch of proposals as a heterogeneous graph with proposal and branch
    nodes, where two objects are adjacent if they share a skeleton node. Each
    proposal carries an image patch and a TreeSample; branches carry no
    features and only relay messages.
    """

    def __init__(self, subgraph, patches, tree_samples):
        """
        Parameters
        ----------
        subgraph : ProposalComputationGraph
            Batch of proposals.
        patches : Dict[Frozenset[int], numpy.ndarray]
            Image patch per proposal.
        tree_samples : Dict[Frozenset[int], TreeSample]
            TreeSample per proposal.
        """
        super().__init__()
        self.idxs_branches = IndexMapping(subgraph.pg_edges)
        self.idxs_proposals = IndexMapping(subgraph.proposals)
        self.n_branches = len(subgraph.pg_edges)

        proposals = list(self.idxs_proposals.id_to_idx)
        self.x_img = torch.from_numpy(np.stack([patches[p] for p in proposals]))
        self.tree_samples = [tree_samples[p] for p in proposals]
        self["proposal"].y = torch.tensor(
            [[p in subgraph.gt_accepts] for p in proposals], dtype=torch.float32
        )

        b2p = [
            [self.idxs_branches.id_to_idx[frozenset((i, j))], idx]
            for p, idx in self.idxs_proposals.id_to_idx.items()
            for i in p
            for j in subgraph.pg_neighbors(i)
        ]
        self.set_edge_index(
            self.line_graph_edges(proposals, self.idxs_proposals),
            ("proposal", "to", "proposal"),
        )
        self.set_edge_index(
            self.line_graph_edges(subgraph.pg_edges, self.idxs_branches),
            ("branch", "to", "branch"),
        )
        self.set_edge_index(b2p, ("branch", "to", "proposal"))
        self.set_edge_index([e[::-1] for e in b2p], ("proposal", "to", "branch"))

    def get_inputs(self):
        return TensorDict(
            {
                "img": self.x_img,
                "tree_samples": self.tree_samples,
                "edge_index_dict": self.edge_index_dict,
                "n_branches": self.n_branches,
            }
        )

    def get_targets(self):
        return self["proposal"].y

    def n_proposals(self):
        return len(self["proposal"].y)

    @staticmethod
    def line_graph_edges(edges, index_mapping):
        """
        Adjacency list of the line graph induced by the given edges, i.e.
        pairs of edges that share a vertex, in both directions.
        """
        edge_index = []
        for e1, e2 in edges_to_line_graph(edges).edges:
            v1 = index_mapping.id_to_idx[frozenset(e1)]
            v2 = index_mapping.id_to_idx[frozenset(e2)]
            edge_index.extend([[v1, v2], [v2, v1]])
        return edge_index

    def set_edge_index(self, edge_index, edge_type):
        if edge_index:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        self[edge_type].edge_index = edge_index


class IndexMapping:
    """
    Maps object IDs to dense indices and back. Tuple IDs are stored as
    frozensets.
    """

    def __init__(self, object_ids):
        self.id_to_idx = dict()
        self.idx_to_id = dict()
        for idx, object_id in enumerate(object_ids):
            if isinstance(object_id, tuple):
                object_id = frozenset(object_id)
            self.id_to_idx[object_id] = idx
            self.idx_to_id[idx] = object_id
