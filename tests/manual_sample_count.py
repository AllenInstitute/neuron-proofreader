"""
Manual benchmark: count how many inference-time samples the dense and
biased-sparse strategies would emit on a real skeleton, without invoking
the model. Useful for verifying that the sparse strategy emits
substantially fewer predictions than the dense one.

Not part of unittest discover. Run directly:

    PYTHONPATH=src python tests/manual_sample_count.py <swc_pointer>

where <swc_pointer> is anything SkeletonGraph.load() accepts (a local
.zip of SWCs or a gs:// URI). For the project's standard test brain:

    SWC=gs://allen-nd-goog/automated_proofreading_dataset/raw_merge_sites/747807/<segmentation_id>/merged_fragments.zip
"""

import sys

import networkx as nx

from neuron_proofreader.merge_proofreading.sparse_sampling import (
    compute_interesting_nodes,
)
from neuron_proofreader.skeleton_graph import SkeletonGraph


def count_dense_samples(graph, step_size=10):
    """
    Mirror "DenseGraphDataset._generate_batch_nodes" node selection (every
    "step_size" microns OR a branching node), skipping image-bounds
    validity. Counts nodes that would be passed to the model.
    """
    count = 0
    for component in nx.connected_components(graph):
        root = next(iter(component))
        first = True
        last_node = root
        for i, j in nx.dfs_edges(graph, source=root):
            if first:
                count += 1
                last_node = i
                first = False
            is_next = graph.dist(last_node, j) >= step_size - 2
            is_branching = graph.degree[j] >= 3
            if is_next or is_branching:
                last_node = j
                count += 1
    return count


def count_sparse_samples(
    graph, step_size=10, branch_radius=25.0, proximity_radius=15.0
):
    """
    Mirror "SparseGraphDataset._generate_batch_nodes" node selection:
    the dense step_size cadence, gated on the interesting set.
    """
    interesting = compute_interesting_nodes(
        graph,
        branch_radius=branch_radius,
        proximity_radius=proximity_radius,
    )
    count = 0
    for component in nx.connected_components(graph):
        root = next(iter(component))
        first = True
        last_node = root
        for i, j in nx.dfs_edges(graph, source=root):
            if first:
                if i in interesting:
                    count += 1
                last_node = i
                first = False
            is_next = graph.dist(last_node, j) >= step_size - 2
            is_branching = graph.degree[j] >= 3
            if is_next or is_branching:
                last_node = j
                if j in interesting:
                    count += 1
    return count


def main(swc_pointer):
    print(f"Loading skeleton from {swc_pointer} ...")
    graph = SkeletonGraph(use_anisotropy=False)
    graph.load(swc_pointer)
    print(graph.summary(prefix="Loaded "))

    dense = count_dense_samples(graph, step_size=10)
    sparse = count_sparse_samples(
        graph, branch_radius=25.0, proximity_radius=15.0
    )
    ratio = dense / sparse if sparse else float("inf")

    print("\nSample-count comparison:")
    print(f"  Dense  (step_size=10um):                 {dense:>10}")
    print(f"  Sparse (branch=25um, proximity=15um):    {sparse:>10}")
    print(f"  Reduction factor: {ratio:.2f}x")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
