"""
Manual benchmark: compare inference-time sample counts between the dense
and biased-sparse strategies, and (optionally) measure the recall of
ground-truth merge sites against the sparse strategy's interesting set.

Run:

    PYTHONPATH=src python tests/manual_sample_count.py <swc_pointer>

    PYTHONPATH=src python tests/manual_sample_count.py <swc_pointer> \\
        --merge-sites-csv ../neurobase_mergedetection/data/merge_proofreading/merge_sites_df.csv \\
        --test-idxs-csv   ../neurobase_mergedetection/data/merge_proofreading/test_idxs.csv \\
        --brain-id 747807

If --merge-sites-csv (and optionally --test-idxs-csv + --brain-id) are
given, every site's nearest skeleton node is checked against the
interesting set; recall = fraction of test-set merge sites that fall
inside a region the sparse sampler would visit.
"""

import argparse
import ast

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import KDTree

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


def count_sparse_samples(graph, interesting, step_size=10):
    """
    Mirror "SparseGraphDataset._generate_batch_nodes" node selection:
    the dense step_size cadence, gated on the interesting set.
    """
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


def load_test_merge_sites(merge_sites_csv, test_idxs_csv, brain_id):
    df = pd.read_csv(merge_sites_csv)
    df["brain_id"] = df["brain_id"].astype(str)
    df["xyz"] = df["xyz"].apply(ast.literal_eval)

    if test_idxs_csv:
        idxs = pd.read_csv(test_idxs_csv)["Indices"].tolist()
        df = df.iloc[idxs]
    if brain_id:
        df = df[df["brain_id"] == str(brain_id)]

    return np.asarray(df["xyz"].tolist(), dtype=np.float32)


def report_recall(graph, interesting, merge_sites_xyz):
    if graph.kdtree is None:
        graph.set_kdtree()

    # Nearest skeleton node for each ground-truth merge site
    site_to_node_dist, nearest_nodes = graph.kdtree.query(merge_sites_xyz)
    interesting = set(interesting)
    caught_mask = np.array(
        [int(n) in interesting for n in nearest_nodes], dtype=bool
    )
    n_total = len(merge_sites_xyz)
    n_caught = int(caught_mask.sum())
    n_missed = n_total - n_caught

    print("\nRecall analysis vs. ground-truth merge sites:")
    print(f"  Test merge sites:                {n_total}")
    print(f"  Distance site -> nearest node:")
    print(f"    median {np.median(site_to_node_dist):.2f} um, "
          f"max {site_to_node_dist.max():.2f} um")
    if n_total == 0:
        return
    pct = 100.0 * n_caught / n_total
    print(f"  Covered by sparse (recall):      "
          f"{n_caught}/{n_total} = {pct:.1f}%")
    print(f"  Missed by sparse:                {n_missed}")

    if n_missed:
        # For each missed site, distance to the *closest* interesting node
        # — useful for tuning branch_radius / proximity_radius if recall is
        # below 100%.
        interesting_arr = np.array(sorted(interesting), dtype=int)
        int_xyz = graph.node_xyz[interesting_arr]
        int_kdtree = KDTree(int_xyz)
        missed_dists, _ = int_kdtree.query(merge_sites_xyz[~caught_mask])
        print(f"  Missed sites -> nearest interesting node (um):")
        print(f"    min {missed_dists.min():.1f}, "
              f"med {np.median(missed_dists):.1f}, "
              f"max {missed_dists.max():.1f}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("swc_pointer")
    parser.add_argument("--step-size", type=int, default=10)
    parser.add_argument("--branch-radius", type=float, default=25.0)
    parser.add_argument("--proximity-radius", type=float, default=15.0)
    parser.add_argument("--merge-sites-csv", default=None)
    parser.add_argument("--test-idxs-csv", default=None)
    parser.add_argument("--brain-id", default=None)
    args = parser.parse_args()

    print(f"Loading skeleton from {args.swc_pointer} ...")
    graph = SkeletonGraph(use_anisotropy=False)
    graph.load(args.swc_pointer)
    print(graph.summary(prefix="Loaded "))

    interesting = compute_interesting_nodes(
        graph,
        branch_radius=args.branch_radius,
        proximity_radius=args.proximity_radius,
    )

    dense = count_dense_samples(graph, step_size=args.step_size)
    sparse = count_sparse_samples(graph, interesting, step_size=args.step_size)
    ratio = dense / sparse if sparse else float("inf")

    print("\nSample-count comparison:")
    print(f"  Dense  (step_size={args.step_size}um):"
          f"                       {dense:>10}")
    print(f"  Sparse (branch={args.branch_radius:g}, "
          f"proximity={args.proximity_radius:g}):    {sparse:>10}")
    print(f"  Reduction factor: {ratio:.2f}x")

    if args.merge_sites_csv:
        sites = load_test_merge_sites(
            args.merge_sites_csv, args.test_idxs_csv, args.brain_id
        )
        report_recall(graph, interesting, sites)


if __name__ == "__main__":
    main()
