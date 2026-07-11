from collections import defaultdict
from scipy.spatial import KDTree

import ast
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pandas as pd

from neuron_proofreader.skeleton_graph import SkeletonGraph
from neuron_proofreader.machine_learning.vision_models import CNN3D
from neuron_proofreader.merge_proofreading.merge_inference import (
    MergeDetector,
    DenseGraphDataset
)
from neuron_proofreader.utils import (
    graph_util as gutil,
    img_util,
    swc_util,
    util
)


def main():
    # Load merge sites
    #path = f"gs://{bucket_name}/{dataset_prefix}/merge_sites.csv"
    #merge_sites = pd.read_csv(path)

    merge_sites = load_merge_sites()

    # Run evaluation for each neuron
    print_experiment_details()
    swc_zip_paths = util.list_gcs_filenames(bucket_name, dataset_prefix, ".zip")
    for cnt, swc_zip_path in enumerate(sorted(swc_zip_paths)):
        # Set paths
        name = get_name(swc_zip_path)
        gt_path = f"gs://allen-nd-goog/ground_truth_tracings/{brain_id}/voxel/{name}.swc"
        fragments_path = f"gs://{bucket_name}/{swc_zip_path}"

        # Evaluate
        name = name[:-3]
        if name in merge_sites:
            print(f"\nFilename ({cnt + 1}/{len(swc_zip_paths)}):", name)
            neuron_merge_sites = merge_sites[name]
            evaluate_on_neuron(gt_path, fragments_path, neuron_merge_sites)


def evaluate_on_neuron(gt_path, fragments_path, merge_sites):
    # Parse name
    name = get_name(fragments_path)
    name = name.replace("_", "-")
    neuron_id = name.split("-")[0]

    # Run merge detection
    dataset = load_data(gt_path, fragments_path)
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
    gt_sites = np.array(merge_sites)
    compute_metrics(gt_sites, detected_sites, name)
    plot_precision_vs_recall(gt_sites, merge_detector, neuron_id)

    # Save results
    dataset.graph.node_radius = 10 * merge_detector.node_preds
    detections_path = os.path.join(output_dir, f"detections-{neuron_id}.zip")
    dataset.graph.to_zipped_swcs(detections_path, preserve_radius=True)
    save_detections(gt_sites, detected_sites, neuron_id)


def load_data(gt_path, fragments_path):
    # Initialize
    graph = load_graphs(gt_path, fragments_path)
    img_path = img_util.find_img_path("allen-nd-goog", "from_aind/", brain_id) + "/0"

    # Load
    dataset = DenseGraphDataset(
        graph,
        img_path,
        patch_shape,
        batch_size=batch_size,
        brightness_clip=brightness_clip,
        is_multimodal=is_multimodal,
        min_search_size=min_search_size,
        step_size=step_size,
        use_new_mask=use_new_mask
    )
    return dataset


def load_graphs(gt_path, fragments_path):
    # Ground Truth
    gt_graph = SkeletonGraph(anisotropy=anisotropy)
    gt_graph.load(gt_path)

    # Fragments
    graph = SkeletonGraph(
        anisotropy=anisotropy,
        node_spacing=node_spacing,
        min_size=min_fragment_size,
        use_anisotropy=False,
        verbose=True
    )
    graph.load(fragments_path)

    # Post process fragments
    name = get_name(gt_path)
    remove_groundtruth_component(graph, name)
    clip_to_groundtruth(gt_graph, graph)
    remove_far_components(gt_graph, graph)
    return graph


def load_merge_sites():
    # Load the file
    df = pd.read_excel("/root/capsule/data/802449-merge-locations.xlsx")

    # Parse merge locations
    df["merge_location"] = df["merge_location"].apply(parse_location)
    df = df[df["merge_location"].notna()]
    df = df.reset_index(drop=True)

    # Extract merge sites
    merge_sites = defaultdict(list)
    for i in range(len(df)):
        name = df["Neuron"][i]
        if isinstance(df["merge_location"][i], tuple):
            for xyz in df["merge_location"][i]:
                merge_sites[name].append(xyz)
        else:
            merge_sites[name].append(df["merge_location"][i])
    return merge_sites


# --- Helpers ---
def get_name(path):
    filename = os.path.basename(path)
    name, ext = os.path.splitext(filename)
    return name


def parse_location(x):
    if pd.isna(x):
        return None
    return ast.literal_eval(x)


def remove_far_components(gt_graph, graph, threshold=3):
    nodes = list()
    for component in nx.connected_components(graph):
        # Compute projections
        is_far = True
        for node in component:
            dist, _ = gt_graph.kdtree.query(graph.node_xyz[node])
            if dist < threshold:
                is_far = False
                break

        # Check whether to remove component
        if is_far:
            nodes.extend(component)
    graph.remove_nodes(nodes)


def remove_groundtruth_component(graph, name):
    swc_id = f"{name}.0"
    component_id = graph.get_component_id_from_swc_id(swc_id)
    gt_nodes = graph.get_nodes_with_component_id(component_id)
    graph.remove_nodes(gt_nodes)


# --- Performance Metrics ---
def compute_metrics(gt_sites, detected_sites, name):
    # Compute stats
    precision, recall, f1 = compute_stats(gt_sites, detected_sites)
    tp_sites, fp_sites = classify_detections(gt_sites, detected_sites)

    # Compile statistical results
    results = ["\n"]
    results.append(len(model_name) * ".")
    results.append(f"Name: {name}")
    results.append(f"Precision: {precision:.4f}")
    results.append(f"Recall: {recall:.4f}")
    results.append(f"F1: {f1:.4f}\n")
    results.append(f"# Detected Merge Sites: {len(detected_sites)}")
    results.append(f"% Detected GT Merge Sites: {len(tp_sites)}/{len(gt_sites)}")
    results.append(len(model_name) * ".")
    results = "\n".join(results)
    print(results)

    # Save results
    log_path = os.path.join(output_dir, "results.txt")
    util.update_txt(log_path, results)


def plot_precision_vs_recall(gt_sites, merge_detector, neuron_id, dt=0.05):
    # Compute error rates
    precision_list = list()
    recall_list = list()
    for t in np.arange(dt, 1, dt):
        detected_sites = merge_detector.get_detected_sites(t)
        precision, recall, _ = compute_stats(gt_sites, detected_sites)
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
    tp_sites = set()
    fp_sites = set()
    for xyz in map(tuple, detected_sites):
        d, idx = kdtree.query(xyz)
        if d <= d_tp and idx not in gt_hits:
            tp_sites.add(xyz)
            gt_hits.add(idx)
        elif d > d_tp:
            fp_sites.add(xyz)
    return tp_sites, fp_sites


def clip_to_groundtruth(gt_graph, graph, threshold=60):
    nodes = list()
    gt_graph.set_kdtree()
    for node in graph.nodes:
        dist, _ = gt_graph.kdtree.query(graph.node_xyz[node])
        if dist > threshold:
             nodes.append(node)
    graph.remove_nodes(nodes)


def compute_stats(gt_sites, detected_sites):
    # Compute metrics
    tp_sites, fp_sites = classify_detections(gt_sites, detected_sites)
    tp = len(tp_sites)
    fp = len(fp_sites)

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
    brain_id = "802449"
    segmentation_id = "jin_masked_mean40_stddev105"
    model_name = "MergeDetectorCNN3D-v5-run2-newmask=True-negativebias=0-20260204-16-0.8669"

    bucket_name = "allen-nd-goog"
    exp_name = "V5"

    accept_threshold = 0.4
    anisotropy = (0.748, 0.748, 1.0)
    batch_size = 24
    brightness_clip = 300
    d_tp = 32
    device = "cuda"
    min_fragment_size = 40
    min_search_size = 400
    node_spacing = 5
    patch_shape = (128, 128, 128)
    step_size = 20
    use_new_mask = True

    img_config = ImageConfig(
        brightness_clip=500,
        patch_shape=(128, 128, 128),
        percentiles=(1, 99.9),
    )

    # Paths
    dataset_prefix = f"automated_proofreading_dataset/merge_detection/whole_brain_fragments_dataset/{brain_id}/{segmentation_id}"
    img_prefix_lookup_path = "/root/capsule/data/exaspim_image_prefixes.json"
    model_path = f"/root/capsule/data/V5/{model_name}.pth"
    output_dir = f"/root/capsule/results/{model_name}"
    util.mkdir(output_dir, delete=True)

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
