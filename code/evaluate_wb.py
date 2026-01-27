from scipy.spatial import KDTree

import ast
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from neuron_proofreader.skeleton_graph import SkeletonGraph
from neuron_proofreader.machine_learning.vision_models import CNN3D
from neuron_proofreader.merge_proofreading.merge_inference import (
    MergeDetector,
    DenseGraphDataset
)
from neuron_proofreader.utils import img_util, swc_util, util


def main():
    # Load merge sites
    path = f"gs://{bucket_name}/{dataset_prefix}/merge_sites.csv"
    merge_sites = pd.read_csv(path)

    # Run evaluation for each neuron
    print_experiment_details()
    swc_zip_paths = util.list_gcs_filenames(bucket_name, dataset_prefix, ".zip")
    for cnt, swc_zip_path in enumerate(sorted(swc_zip_paths)):
        # Extract data
        neuron_data_path = f"gs://{bucket_name}/{swc_zip_path}"
        filename = os.path.basename(neuron_data_path)
        name, ext = os.path.splitext(filename)
        neuron_merge_sites = merge_sites[merge_sites["GroundTruth_ID"] == name]

        # Evaluate
        print(f"\nFilename ({cnt + 1}/{len(swc_zip_paths)}):", name)
        evaluate_on_neuron(neuron_data_path, neuron_merge_sites, name)


def evaluate_on_neuron(data_path, merge_sites, name):
    # Parse name
    name = name.replace("_", "-")
    neuron_id = name.split("-")[0]

    # Run merge detection
    dataset = load_data(data_path)
    merge_detector = MergeDetector(
        dataset,
        model,
        model_path,
        device="cuda",
        threshold=accept_threshold,
    )
    merge_detector.search_graph()

    # Compute performance metrics
    detected_sites = merge_detector.get_detected_sites(accept_threshold)
    gt_sites = np.array(merge_sites["World"].apply(ast.literal_eval).tolist())
    compute_metrics(detected_sites, gt_sites, name)
    plot_precision_vs_recall(merge_detector, gt_sites, neuron_id)

    # Save results
    save_detections(detected_sites, gt_sites, neuron_id)


def load_data(swc_pointer):
    # Load graphs
    graph = SkeletonGraph(
        anisotropy=anisotropy,
        node_spacing=node_spacing,
        min_size=min_fragment_size,
        verbose=True
    )
    graph.load(swc_pointer)

    # Initialize dataset
    img_path = img_util.find_img_path("allen-nd-goog", "from_aind/", brain_id) + "/0"
    dataset = DenseGraphDataset(
        graph,
        img_path,
        patch_shape,
        batch_size=batch_size,
        is_multimodal=is_multimodal,
        min_search_size=min_search_size,
        step_size=step_size,
    )
    return dataset


# --- Performance Metrics ---
def compute_metrics(detected_sites, gt_sites, name):
    # Compute stats
    precision, recall, f1 = compute_stats(detected_sites, gt_sites)

    # Compile statistical results
    results = [f"\nName: {name}"]
    results.append(len(model_name) * ".")
    results.append(f"Precision: {precision}")
    results.append(f"Recall: {recall}")
    results.append(f"F1: {f1}\n")
    results.append(f"# Detected Merge Sites: {len(detected_sites)}")
    results.append(f"% Detected GT Merge Sites: {int(len(gt_sites) * recall)}/{len(gt_sites)}")
    results = "\n".join(results)
    print(results)

    # Save results
    log_path = os.path.join(output_dir, "results.txt")
    util.update_txt(log_path, results)


def plot_precision_vs_recall(merge_detector, gt_sites, neuron_id, dt=0.05):
    # Compute error rates
    precision_list = list()
    recall_list = list()
    for t in np.arange(dt, 1, dt):
        detected_sites = merge_detector.get_detected_sites(t)
        precision, recall, _ = compute_stats(detected_sites, gt_sites)
        precision_list.append(precision)
        recall_list.append(recall)

    # Visualize result
    plt.figure(figsize=(6, 4))
    plt.plot(
        precision_list,
        recall_list,
        marker="o",
        linestyle="-",
        color="steelblue"
    )
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Save result
    filename = f"precision_vs_recall-{neuron_id}.png"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")


# --- Helpers ---
def classify_detections(gt_sites, detected_sites):
    # Build KD-True of GT merge sites
    if len(gt_sites) > 0:
        kdtree = KDTree(gt_sites)
        gt_hits = set()
    else:
        return set(), set()

    # Separate detected sites into true and false positives
    true_positives = set()
    false_positives = set()
    for xyz in map(tuple, detected_sites):
        d, idx = kdtree.query(xyz)
        if d <= d_tp and idx not in gt_hits:
            true_positives.add(xyz)
            gt_hits.add(idx)
        elif d > d_tp:
            false_positives.add(xyz)
    return true_positives, false_positives


def compute_stats(gt_sites, detected_sites):
    # Compute metrics
    true_sites, false_sites = classify_detections(gt_sites, detected_sites)
    tp = len(true_sites)
    fp = len(false_sites)

    precision = tp / (tp + fp + 1e-5)
    recall = tp / (len(gt_sites) + 1e-5)
    f1 = 2 * precision * recall / (precision + recall + 1e-5)
    return precision, recall, f1


def print_experiment_details():
    # Initialize log path
    log_path = os.path.join(output_dir, "results.txt")
    util.mkdir(output_dir)
    if os.path.exists(log_path):
        os.remove(log_path)

    # Report experiment details
    util.update_txt(log_path, "\nExperiment Details")
    util.update_txt(log_path, "-" * (len(model_name) + 9))
    util.update_txt(log_path, f"Brain_ID: {brain_id}")
    util.update_txt(log_path, f"Segmentation_ID: {segmentation_id}")
    util.update_txt(log_path, f"Model Name: {model_name}\n")


def save_detections(gt_sites, detected_sites, neuron_id):
    # Get sites
    tp_sites, fp_sites = classify_detections(gt_sites, detected_sites)
    _, fn_sites = classify_detections(detected_sites, gt_sites)

    # Save sites
    sites_path = os.path.join(output_dir, f"sites-{neuron_id}.zip")
    save_points(sites_path, tp_sites, "0.0 1.0 0.0", "true_positive")
    save_points(sites_path, fp_sites, "1.0 0.0 0.0", "false_positive")
    save_points(sites_path, fn_sites, "1.0 1.0 1.0", "false_negative")


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
    brain_id = "802449"
    segmentation_id = "jin_masked_mean40_stddev105"

    exp_name = "V5"
    model_name = "MergeDetectorCNN3D-v4-run1-newmask=False-negativebias=0-20260109-135-0.8374"

    accept_threshold = 0.4
    anisotropy = (0.748, 0.748, 1.0)
    batch_size = 32
    d_tp = 32
    device = "cuda:0"
    min_fragment_size = 40
    min_search_size = 5000
    node_spacing = 5
    patch_shape = (128, 128, 128)
    step_size = 20

    # Paths
    dataset_prefix = f"automated_proofreading_dataset/merge_detection/whole_brain_fragments_dataset/{brain_id}/{segmentation_id}"
    img_prefix_lookup_path = "/root/capsule/data/exaspim_image_prefixes.json"
    model_path = f"/root/capsule/data/{exp_name}/{model_name}.pth"
    output_dir = f"/root/capsule/results"
    util.mkdir(output_dir)

    # Model
    if "VisionDGCNN" in model_name:
        is_multimodal = True
        model = VisionDGCNN(patch_shape)
    elif "CNN3D" in model_name:
        is_multimodal = False
        model = CNN3D(
            patch_shape,
            n_conv_layers=6,
            n_feat_channels=24,
        )

    # Run evaluation
    main()
