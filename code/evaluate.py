from segmentation_skeleton_metrics.datamodules.graph_loading import (
    DataLoader,
)
from segmentation_skeleton_metrics.utils.img_util import TensorStoreImage
from scipy.spatial import KDTree
from tqdm import tqdm
from zipfile import ZipFile

import ast
import numpy as np
import os
import pandas as pd

from neuron_proofreader.utils import swc_util, util
from neuron_proofreading_evaluation.proofread_splits import (
    data_handling as data_util,
)


def main():
    # Load merge sites
    gt_sites = pd.read_csv(gt_sites_path)
    gt_sites["xyz"] = gt_sites["xyz"].apply(ast.literal_eval)

    pred_sites = pd.read_csv(pred_sites_path)
    pred_sites = pred_sites[pred_sites["Prediction"] > threshold]
    pred_sites["World"] = pred_sites["World"].apply(ast.literal_eval)
    pred_sites = np.array([xyz for xyz in pred_sites["World"]])

    # Run evaluation for each neuron
    results = list()
    for gt_id_i, gt_sites_i in tqdm(gt_sites.groupby("cell_id")["xyz"]):
        # Extract intersection pred sites
        gt_neuron = read_swc_points(get_gt_path(gt_id_i))
        gt_kdtree = KDTree(gt_neuron)
        dd, _ = gt_kdtree.query(pred_sites)
        pred_sites_i = pred_sites[dd < 5]

        # Extract true positive pred sites
        gt_sites_i = np.stack(gt_sites_i.values)
        gt_sites_kdtree = KDTree(gt_sites_i)
        dd, _ = gt_sites_kdtree.query(pred_sites_i)

        # Report performance metrics
        n_gt = len(gt_sites_i)
        n_pred = len(pred_sites_i)
        n_tp = len(pred_sites_i[dd <= 32])

        recall = n_tp / n_gt
        precision = n_tp / n_pred if n_pred > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # Store results
        results.append({
            "GroundTruth_ID": gt_id_i,
            "# GT Sites": n_gt,
            "# Pred Sites": n_pred,
            "# TP Sites": n_tp,
            "Recall": recall,
            "Precision": precision,
            "F1": f1
        })
        save_sites(gt_id_i, gt_sites_i, pred_sites_i)

    # Save results
    compute_overall_stats(results)
    path = os.path.join(output_dir, "evaluation_results.csv")
    df = pd.DataFrame(results)
    df.to_csv(path, index=False)
    print(df)
    #save_skeletons(gt_sites)


def compute_overall_stats(results):
    gt = sum(r["# GT Sites"] for r in results)
    pred = sum(r["# Pred Sites"] for r in results)
    tp = sum(r["# TP Sites"] for r in results)
    weights = [r["# GT Sites"] / gt for r in results]
    recall = sum(w * r["Recall"] for w, r in zip(weights, results))
    precision = sum(w * r["Precision"] for w, r in zip(weights, results))
    f1 = sum(w * r["F1"] for w, r in zip(weights, results))
    results.append({
        "GroundTruth_ID": "Overall",
        "# GT Sites": gt,
        "# Pred Sites": pred,
        "# TP Sites": tp,
        "Recall": recall,
        "Precision": precision,
        "F1": f1
    })


# --- Load Data ---
def get_gt_path(gt_id):
    bucket_name, subpath = util.parse_cloud_path(gt_swcs_path)
    for path in util.list_gcs_paths(bucket_name, subpath):
        if gt_id in path:
            return path
    raise Exception(f"GT SWC File not Found for {gt_id}")


def read_swc_points(swc_path):
    reader = swc_util.Reader(verbose=False)
    swc_dicts = reader(swc_path)
    return np.array([swc_dict["xyz"] for swc_dict in swc_dicts]).squeeze()


def create_swc_names(labels):
    return set([f"{label}.0" for label in list(labels)])


# --- Save Results ---
def save_sites(gt_id, gt_sites, pred_sites):
    # Get sites
    dd, _ = KDTree(gt_sites).query(pred_sites)
    tp_sites = pred_sites[dd <= 32]
    fp_sites = pred_sites[dd > 32]

    dd, _ = KDTree(tp_sites).query(gt_sites)
    fn_sites = gt_sites[dd > 32]

    # Save sites
    sites_path = os.path.join(output_dir, f"results-{gt_id}.zip")
    save_points(sites_path, tp_sites, "0.0 1.0 0.0", "true_positive")
    save_points(sites_path, fp_sites, "1.0 0.0 0.0", "false_positive")
    save_points(sites_path, fn_sites, "1.0 1.0 1.0", "false_negative")


def save_skeletons(gt_sites):
    # Initializations
    base_path = f"gs://allen-nd-goog/from_google/{brain_id}/whole_brain/{segmentation_id}/"
    fragments_path = os.path.join(base_path, "swcs")
    gt_dataloader = DataLoader(
        anisotropy=(0.748, 0.748, 1.0),
        use_anisotropy=True,
        verbose=False,
    )
    fragments_dataloader = DataLoader(
        anisotropy=(0.748, 0.748, 1.0),
        use_anisotropy=False,
        verbose=False,
    )
    segmentation = TensorStoreImage(base_path)

    # Save intersecting fragments
    for gt_id in tqdm(gt_sites["cell_id"].unique(), desc="Save Skeletons"):
        # Load GT skeleton
        gt_swc_path = get_gt_path(gt_id)
        gt_graphs = gt_dataloader.load_groundtruth(gt_swc_path, segmentation)
        gt_graph = list(gt_graphs.values())[0]

        # Load fragments
        swc_names = create_swc_names(gt_graph.node_labels())
        fragment_graphs = fragments_dataloader.load_fragments(
            fragments_path, swc_names=swc_names
        )
        fragment_graphs = data_util.flip_coordinates(fragment_graphs)

        # Create zip writer
        zip_path = os.path.join(output_dir, f"results-{gt_id}.zip")
        with ZipFile(zip_path, "a") as zip_writer:
            gt_graph.to_zipped_swcs(zip_writer)
            for key, graph in fragment_graphs.items():
                graph.to_zipped_swcs(zip_writer)


def save_points(zip_path, pts, color, prefix):
    swc_util.write_points(
        zip_path,
        pts,
        color=color,
        prefix=prefix,
        radius=10,
        write_mode="a"
    )


if __name__ == "__main__":
    # Parameters
    bucket_name = "allen-nd-goog"
    brain_id = "794495"
    segmentation_id = "denoised.unet_tri_denoised_2370000"
    suffix = "20260508_2158"
    threshold = 0.9

    # Paths
    gt_swcs_path = f"gs://allen-nd-goog/ground_truth_tracings/{brain_id}/world"
    gt_sites_path = "/home/jupyter/denoised_750k_tri-brain.csv"
    pred_sites_path = f"/home/jupyter/results/merge_detection/794495/dense-denoised.unet_tri_denoised_2370000/model_predictions.csv"
    output_dir = f"/home/jupyter/results/evaluate_merge_detection/{brain_id}/dense-{segmentation_id}"
    util.mkdir(output_dir, delete=True)

    # Run evaluation
    main()
