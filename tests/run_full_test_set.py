"""
Sweep every (brain_id, segmentation_id) fragment in the merge-detection
test set: load each skeleton, compute dense vs biased-sparse sample counts,
and measure ground-truth-merge-site recall against the sparse interesting
set. Write one CSV row per fragment.

Usage:
    PYTHONPATH=src python tests/run_full_test_set.py \\
        --merge-sites-csv ../neurobase_mergedetection/data/merge_proofreading/merge_sites_df.csv \\
        --test-idxs-csv   ../neurobase_mergedetection/data/merge_proofreading/test_idxs.csv \\
        --output-csv      tests/test_set_recall.csv

Each fragment is persisted to the CSV immediately after analysis, so the
script can be interrupted and re-run with --resume to pick up where it
left off.
"""

import argparse
import ast
import os
import sys
import time
import traceback

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from neuron_proofreader.merge_proofreading.sparse_sampling import (
    compute_interesting_nodes,
)
from neuron_proofreader.skeleton_graph import SkeletonGraph

# Local import — same dir
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from manual_sample_count import count_dense_samples, count_sparse_samples  # noqa: E402


GCS_ROOT = "gs://allen-nd-goog/automated_proofreading_dataset/raw_merge_sites"


def analyze_fragment(graph, sites_xyz, branch_radius, proximity_radius, step_size):
    interesting = compute_interesting_nodes(
        graph,
        branch_radius=branch_radius,
        proximity_radius=proximity_radius,
    )
    interesting_set = {int(n) for n in interesting}

    dense = count_dense_samples(graph, step_size=step_size)
    sparse = count_sparse_samples(graph, interesting_set, step_size=step_size)

    if graph.kdtree is None:
        graph.set_kdtree()
    site_to_node_dist, nearest_nodes = graph.kdtree.query(sites_xyz)
    caught_mask = np.array(
        [int(n) in interesting_set for n in nearest_nodes], dtype=bool
    )
    n_total = len(sites_xyz)
    n_caught = int(caught_mask.sum())
    n_missed = n_total - n_caught

    if n_missed and interesting_set:
        int_arr = np.array(sorted(interesting_set), dtype=int)
        int_xyz = graph.node_xyz[int_arr]
        int_kdtree = KDTree(int_xyz)
        missed_dists, _ = int_kdtree.query(sites_xyz[~caught_mask])
        missed_med = float(np.median(missed_dists))
        missed_max = float(missed_dists.max())
    else:
        missed_med = float("nan")
        missed_max = float("nan")

    return {
        "n_components": int(nx.number_connected_components(graph)),
        "n_nodes": int(graph.number_of_nodes()),
        "n_interesting": int(len(interesting_set)),
        "site_to_node_dist_med": float(np.median(site_to_node_dist)),
        "site_to_node_dist_max": float(site_to_node_dist.max()),
        "dense_samples": int(dense),
        "sparse_samples": int(sparse),
        "reduction_factor": (dense / sparse) if sparse else float("inf"),
        "n_caught": n_caught,
        "n_missed": n_missed,
        "recall": n_caught / n_total if n_total else float("nan"),
        "missed_to_interesting_med": missed_med,
        "missed_to_interesting_max": missed_max,
    }


def load_test_groups(merge_sites_csv, test_idxs_csv):
    df = pd.read_csv(merge_sites_csv)
    df["brain_id"] = df["brain_id"].astype(str)
    df["xyz"] = df["xyz"].apply(ast.literal_eval)
    if test_idxs_csv:
        idxs = pd.read_csv(test_idxs_csv)["Indices"].tolist()
        df = df.iloc[idxs].reset_index(drop=True)
    return df.groupby(["brain_id", "segmentation_id"], sort=False)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--merge-sites-csv", required=True)
    parser.add_argument("--test-idxs-csv", default=None)
    parser.add_argument(
        "--output-csv", default="tests/test_set_recall.csv"
    )
    parser.add_argument("--root", default=GCS_ROOT)
    parser.add_argument("--step-size", type=int, default=10)
    parser.add_argument("--branch-radius", type=float, default=25.0)
    parser.add_argument("--proximity-radius", type=float, default=15.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    groups = load_test_groups(args.merge_sites_csv, args.test_idxs_csv)
    n_groups = groups.ngroups
    print(f"Test fragments to process: {n_groups}")

    rows = []
    seen = set()
    if args.resume and os.path.exists(args.output_csv):
        prev = pd.read_csv(args.output_csv)
        rows = prev.to_dict("records")
        seen = {
            (str(r["brain_id"]), str(r["segmentation_id"]))
            for r in rows
            if pd.notna(r.get("recall", np.nan)) or r.get("error")
        }
        print(f"Resuming with {len(seen)} fragments already in {args.output_csv}")

    for k, ((brain_id, seg_id), group) in enumerate(groups, start=1):
        key = (str(brain_id), str(seg_id))
        if key in seen:
            print(f"[{k}/{n_groups}] skip {brain_id}/{seg_id} (already in CSV)")
            continue

        sites_xyz = np.asarray(group["xyz"].tolist(), dtype=np.float32)
        pointer = f"{args.root}/{brain_id}/{seg_id}/merged_fragments.zip"
        print(
            f"\n[{k}/{n_groups}] {brain_id} / {seg_id} "
            f"({len(sites_xyz)} sites)"
        )
        print(f"  pointer: {pointer}")

        t0 = time.time()
        row = {
            "brain_id": brain_id,
            "segmentation_id": seg_id,
            "n_test_sites": len(sites_xyz),
            "branch_radius": args.branch_radius,
            "proximity_radius": args.proximity_radius,
            "step_size": args.step_size,
            "error": "",
        }

        try:
            graph = SkeletonGraph(use_anisotropy=False)
            graph.load(pointer)
            metrics = analyze_fragment(
                graph,
                sites_xyz,
                args.branch_radius,
                args.proximity_radius,
                args.step_size,
            )
            row.update(metrics)
            print(
                f"  recall {metrics['n_caught']}/{len(sites_xyz)} = "
                f"{100 * metrics['recall']:.1f}%; "
                f"sparse {metrics['sparse_samples']:,} vs "
                f"dense {metrics['dense_samples']:,} "
                f"(x{metrics['reduction_factor']:.2f})"
            )
        except Exception as e:
            print(f"  [error] {e}")
            traceback.print_exc()
            row["error"] = repr(e)[:300]

        row["elapsed_sec"] = time.time() - t0
        rows.append(row)
        os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
        pd.DataFrame(rows).to_csv(args.output_csv, index=False)
        print(f"  saved -> {args.output_csv} ({len(rows)} rows)")

    # Final summary
    print("\n=== Summary ===")
    final = pd.DataFrame(rows)
    if "recall" in final.columns:
        ok = final[final["error"] == ""]
        if len(ok):
            n_sites_total = ok["n_test_sites"].sum()
            n_caught_total = ok["n_caught"].sum()
            agg_recall = n_caught_total / n_sites_total if n_sites_total else float("nan")
            mean_reduction = ok["reduction_factor"].mean()
            print(f"Fragments analyzed:    {len(ok)}/{len(final)}")
            print(f"Aggregate test sites:  {n_sites_total}")
            print(f"Aggregate recall:      {n_caught_total}/{n_sites_total} = {100*agg_recall:.1f}%")
            print(f"Mean reduction factor: x{mean_reduction:.2f}")
    print(f"Results CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
