"""
Created on Fri July 25 12:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Utilities for building batched topological graph inputs from skeleton
subgraphs, for use with CurveEncoder + graph classifier pipelines.

"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch


@dataclass
class TopoGraphInput:
    """
    Manages the inputs to a CurveEncoder + graph model pipeline for a single
    skeleton subgraph. Returned by a dataset's __getitem__ and consumed by
    collate_topo_graphs to form a batched TopoGraphBatch.

    Attributes
    ----------
    path_feats : List[numpy.ndarray]
        First-order finite differences per path with shape (N_i, 3),
        order-aligned with edge_index.
    edge_index : List[Tuple[int, int]]
        Edges of the reduced topological graph, as index pairs into topo_nodes.
    node_feats : numpy.ndarray
        Per-topological-node features (xyz_centered, radius) with shape (V, 4).
    """

    path_feats: List[np.ndarray]
    edge_index: List[Tuple[int, int]]
    node_feats: np.ndarray


@dataclass
class TopoGraphBatch:
    """
    Batched topological graph representation produced by collate_topo_graphs,
    holding tensors ready for CurveEncoder + graph model inference.

    Attributes
    ----------
    curve_input : List[numpy.ndarray]
        First-order finite differences per path with shape (N_i, 3), flattened
        across the batch and order-aligned with edge_index columns. Feed to
        CurveEncoder; the resulting CLS embeddings map directly to edge_attr.
    edge_index : torch.Tensor
        Block-diagonal edge index with shape (2, E_total), node IDs offset per
        sample.
    node_feats : torch.Tensor
        Stacked (xyz_centered, radius) for topological nodes across the batch,
        with shape (V_total, 4).
    node_batch : torch.Tensor
        Sample index per topological node for graph-level pooling, with shape
        (V_total,).
    """

    curve_input: List[np.ndarray]
    edge_index: torch.Tensor
    node_feats: torch.Tensor
    node_batch: torch.Tensor

    def to(self, device):
        return TopoGraphBatch(
            curve_input=self.curve_input,
            edge_index=self.edge_index.to(device),
            node_feats=self.node_feats.to(device),
            node_batch=self.node_batch.to(device),
        )


# ---Graph Featurization ---
def topological_decomposition(subgraph, root=0):
    """
    Decomposes a rooted subgraph into paths between topological points
    (i.e. root, branch points, leaves), for embedding with CurveEncoder.

    Parameters
    ----------
    subgraph : SkeletonGraph
        Rooted subgraph.
    root : int, optional
        Node ID of the root. Default is 0.

    Returns
    -------
    topo_nodes : List[int]
        Node IDs of topological nodes (i.e. leaf and branching).
    paths : List[List[int]]
        Node ID chains (inclusive of both endpoints), root-to-leaf order,
        parallel to edge_index.
    edge_index : List[Tuple[int, int]]
        Edges of the reduced topological graph, as index pairs into topo_nodes.
    """
    def is_topo(i):
        return i == root or subgraph.degree[i] == 1 or subgraph.degree[i] >= 3

    # Extract topological nodes
    topo_nodes = [i for i in range(subgraph.number_of_nodes()) if is_topo(i)]
    topo_idx = {n: k for k, n in enumerate(topo_nodes)}

    # Extract paths between topological nodes
    paths, edge_index = [], []
    stack = [(root, root, [root])]  # (parent, current, path_so_far)
    visited = {root}
    while stack:
        parent, curr, path = stack.pop()
        for nb in subgraph.neighbors(curr):
            if nb == parent or nb in visited:
                continue
            visited.add(nb)
            if is_topo(nb):
                paths.append(path + [nb])
                edge_index.append((topo_idx[path[0]], topo_idx[nb]))
                stack.append((curr, nb, [nb]))
            else:
                stack.append((curr, nb, path + [nb]))

    return topo_nodes, paths, edge_index


def curve_to_diffs(curve):
    """
    Converts a curve to first-order finite differences for CurveEncoder.

    Centers at the first point then computes in-place differences, producing
    diffs[0] == [0, 0, 0].

    Parameters
    ----------
    curve : numpy.ndarray
        Curve coordinates with shape (N, 3).

    Returns
    -------
    diffs : numpy.ndarray
        First-order finite differences with shape (N, 3).
    """
    diffs = curve.copy()
    diffs -= diffs[0]
    diffs[1:] -= diffs[:-1]
    return diffs


def path_features(subgraph, path):
    """
    Extracts first-order finite differences along a path for CurveEncoder.

    Parameters
    ----------
    subgraph : SkeletonGraph
        Graph containing the given path.
    path : list
        Ordered node IDs from one topological point to the next.

    Returns
    -------
    diffs : numpy.ndarray
        First-order finite differences with shape (len(path), 3).
    """
    return curve_to_diffs(subgraph.node_xyz[path])


# --- Collation ---
def collate_topo_graphs(batch):
    """
    Collates (patches, path_feats, edge_index, node_feats, label) samples into
    a batched representation ready for CurveEncoder + graph model inference.

    The key invariant: paths[i] corresponds to edge_index[i] by construction
    from topological_decomposition, and collation extends curve_input /
    edge_index in lockstep per sample. After running CurveEncoder once on
    curve_input, the resulting CLS embeddings map directly to edge_attr in
    row order -- do not reorder curve_input before encoding.

    Parameters
    ----------
    batch : list
        Each element is (patches, TopoGraphInput, label).

    Returns
    -------
    patches : torch.Tensor
        Batched image patches. Shape (B, 2, *patch_shape).
    topo_graphs : TopoGraphBatch
    labels : torch.Tensor
        Shape (B,).
    """
    patches_list, samples, labels = zip(*batch)

    patches = torch.tensor(np.stack(patches_list), dtype=torch.float32)

    curve_input, edge_index, node_batch, node_feats = [], [], [], []
    node_offset = 0
    for b, sample in enumerate(samples):
        n_nodes = len(sample.node_feats)
        curve_input.extend(sample.path_feats)
        edge_index.extend((u + node_offset, v + node_offset) for u, v in sample.edge_index)
        node_batch.extend([b] * n_nodes)
        node_feats.append(sample.node_feats)
        node_offset += n_nodes

    if edge_index:
        edge_index_t = torch.tensor(edge_index, dtype=torch.long).T
    else:
        edge_index_t = torch.zeros(2, 0, dtype=torch.long)

    topo_graphs = TopoGraphBatch(
        curve_input=curve_input,
        edge_index=edge_index_t,
        node_feats=torch.tensor(np.concatenate(node_feats, axis=0), dtype=torch.float32),
        node_batch=torch.tensor(node_batch, dtype=torch.long),
    )
    labels = torch.tensor(labels, dtype=torch.long)
    return patches, topo_graphs, labels
