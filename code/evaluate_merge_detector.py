from aind_exaspim_dataset_utils import s3_util
from scipy.spatial import KDTree

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from neuron_proofreader.machine_learning.point_cloud_models import VisionDGCNN
from neuron_proofreader.machine_learning.vision_models import CNN3D, ViT3D
from neuron_proofreader.merge_proofreading import merge_dataloading as data_util
from neuron_proofreader.merge_proofreading.merge_inference import (
    DenseGraphDataset,
    MergeDetector,
)
from neuron_proofreader.merge_proofreading.merge_datasets import (
    MergeSiteDataset
)
from neuron_proofreader.utils import swc_util, util


def main():
    # Load data
    dataset = init_dataset()

    # Run evaluation for each brain
    print("\nModel Name:", model_name)
    for brain_id in data_util.get_brain_ids(dataset.merge_sites_df, is_test=is_test):
        if brain_id == "685221":
            continue
        evaluate(brain_id, dataset.graphs[brain_id], dataset.merge_sites_df)


def evaluate(brain_id, fragments_graph, merge_sites_df):
    # Paths
    img_path = s3_util.get_img_prefix(brain_id, image_prefixes_path)
    img_path = os.path.join(img_path, "0")

    # Dataset
    dataset = DenseGraphDataset(
        fragments_graph,
        img_path,
        patch_shape,
        batch_size=batch_size,
        brightness_clip=brightness_clip,
        is_multimodal=is_multimodal,
        step_size=step_size,
    )

    # Initialize merge detection object
    merge_detector = MergeDetector(
        dataset,
        model,
        model_path,
        threshold=accept_threshold,
    )

    # Run detection
    merge_detector.search_graph()
    fragments_graph.node_radius = 10 * merge_detector.node_preds
    detections_path = os.path.join(output_dir, f"detections-{brain_id}.zip")
    fragments_graph.to_zipped_swcs(detections_path, preserve_radius=True)

    # Evaluate
    print("-" * 100)
    print("Brain ID:", brain_id)
    gt_sites = data_util.get_brain_merge_sites(merge_sites_df, brain_id)
    detected_sites = merge_detector.get_detected_sites(accept_threshold)
    compute_performance_metrics(brain_id, detected_sites, gt_sites)
    plot_precision_vs_recall(brain_id, merge_detector, gt_sites)
    print("-" * 100)


# --- Performance Metrics ---
def compute_performance_metrics(brain_id, detected_sites, gt_sites):
    # Compute metrics
    precision, recall, f1 = compute_metrics(gt_sites, detected_sites)
    save_detections(brain_id, gt_sites, detected_sites)

    # Compile statistical results
    results = []
    results.append(f"Precision: {precision}")
    results.append(f"Recall: {recall}")
    results.append(f"F1: {f1}\n")
    results.append(f"# Detected Merge Sites: {len(detected_sites)}")
    results.append(f"% Detected GT Merge Sites: {int(len(gt_sites) * recall)}/{len(gt_sites)}")
    results = "\n".join(results)
    print(results)

    # Save results
    path = os.path.join(output_dir, f"results-{brain_id}.txt")
    with open(path, "w") as f:
        f.write(results)


def compute_metrics(gt_sites, detected_sites):
    true_sites, false_sites = classify_detections(gt_sites, detected_sites)
    tp = len(true_sites)
    fp = len(false_sites)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / len(gt_sites)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return precision, recall, f1


def classify_detections(gt_sites, detected_sites):
    # Build KD-True of GT merge sites
    if len(gt_sites) > 0:
        kdtree = KDTree(gt_sites)
        gt_hits = set()
    else:
        return list(), list()

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


def save_detections(brain_id, gt_sites, detected_sites):
    # Get sites
    tp_sites, fp_sites = classify_detections(gt_sites, detected_sites)
    _, fn_sites = classify_detections(detected_sites, gt_sites)

    # Save sites
    sites_path = os.path.join(output_dir, f"sites-{brain_id}.zip")
    save_points(sites_path, tp_sites, "0.0 1.0 0.0", "true_positive")
    save_points(sites_path, fp_sites, "1.0 0.0 0.0", "false_positive")
    save_points(sites_path, fn_sites, "1.0 1.0 1.0", "false_negative")


def plot_precision_vs_recall(brain_id, merge_detector, gt_sites, dt=0.05):
    # Compute error rates
    precision_list = list()
    recall_list = list()
    for t in np.arange(dt, 1, dt):
        detected_sites = merge_detector.get_detected_sites(t)
        precision, recall, _ = compute_metrics(gt_sites, detected_sites)
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
    filename = f"precision_vs_recall-{brain_id}.png"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")


# --- Data Loading ---
def init_dataset():
    # Load merge sites
    test_idxs = data_util.read_idxs(test_idxs_path)
    merge_sites_df = data_util.load_merge_sites_df(
        merge_sites_path, is_test=is_test
        )
    #merge_sites_df = merge_sites_df.iloc[test_idxs].reset_index(drop=True)

    # Initialize dataset
    dataset = MergeSiteDataset(
        merge_sites_df,
        anisotropy=anisotropy,
        node_spacing=node_spacing,
        patch_shape=patch_shape,
    )

    # Load data
    data_util.load_groundtruth(dataset, is_test=is_test)
    data_util.load_fragments(dataset, is_test=is_test)
    data_util.load_images(
        dataset,
        image_prefixes_path,
        segmentation_prefixes_path,
        is_test=is_test,
    )
    dataset.remove_isolated_sites()
    return dataset


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
    model_name = "MergeDetectorCNN3D-20251216-121-0.8177"
    exp_name = "V2"

    accept_threshold = 0.5
    anisotropy = (0.748, 0.748, 1.0)
    batch_size = 32
    brightness_clip = 300
    d_tp = 32
    device = "cuda"
    is_multimodal = False
    is_test = True
    node_spacing = 5
    patch_shape = (128, 128, 128)
    step_size = 20

    # Paths
    bucket_name = "allen-nd-goog"
    fragments_prefix = "automated_proofreading_dataset/raw_merge_sites"
    image_prefixes_path = "/root/capsule/data/exaspim_image_prefixes.json"
    segmentation_prefixes_path = "/root/capsule/data/exaspim_segmentation_prefixes.json"
    merge_sites_path = f"/root/capsule/data/{exp_name}/merge_sites_df.csv"
    test_idxs_path = f"/root/capsule/data/{exp_name}/test_idxs.csv"
    model_path = f"/root/capsule/data/{exp_name}/{model_name}.pth"
    output_dir = f"/root/capsule/results/{exp_name}-{model_name}"
    util.mkdir(output_dir)

    # Model
    model = CNN3D(
        patch_shape,
        n_conv_layers=6,
        n_feat_channels=24,
        use_double_conv=True
    )

    # Run evaluation
    main()
