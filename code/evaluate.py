"""
Created on Wed June 3 12:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code evaluating merge detection models.

"""

from pathlib import Path
from scipy.spatial import KDTree
from tqdm import tqdm

import argparse
import numpy as np
import os
import pandas as pd

from neuron_proofreader.configs import ImageConfig
from neuron_proofreader.skeleton_graph import SkeletonGraph
from neuron_proofreader.merge_proofreading.merge_detection import (
    MergeDetector,
)
from neuron_proofreader.merge_proofreading.search_datasets import (
    SparseSearchDataset,
    DenseSearchDataset,
)
from neuron_proofreader.models.new_vision_models import NewCNN3D
from neuron_proofreader.utils import ml_util, util

from neuron_proofreading_evaluation import utils
from neuron_proofreading_evaluation import visualization as viz
import neuron_proofreading_evaluation.proofread_merges.merge_evaluation as evaluate


def main():
    # Load data
    gt_df = utils.load_sites_df(gt_sites_path)
    gt_names = gt_df["cell_id"].unique()
    gt_neurons = load_gt_neurons(gt_names)  # name --> xyz coordinates
    fragments_datasets = load_datasets(gt_names)  # search_mode --> dataset

    # Extract models to evaluate
    print("\nStep 3: Evaluate Model Families")
    experiment_names = util.listdir(root_dir)
    print(f"Found {len(experiment_names)} model directories in {root_dir}")

    # Evaluate models
    for experiment_name in [e for e in experiment_names if "convnext" not in e]:
        # Create results directory for model family
        print("\nModel Family:", experiment_name)
        input_dir = f"{root_dir}/{experiment_name}"
        output_dir = f"/results/{experiment_name}"
        util.mkdir(output_dir)

        # Iterate over top k models
        for model_path in get_top_ckpts(input_dir, k=top_k_ckpts):
            # Report model name
            print(f"\n{'='*(len(model_path.name) + 4)}")
            print(f"  {model_path.name}")
            print(f"{'='*(len(model_path.name) + 4)}")

            # Create model
            config_path = os.path.join(input_dir, "model_config.json")
            config = util.read_json(config_path)

            model = NewCNN3D(**config)
            ml_util.load_model(model, model_path)

            # Compute batch size
            input_shape = config["input_shape"]
            batch_size = ml_util.find_max_eval_batch_size(model, input_shape)
            batch_size = int(batch_size * 0.85)
            print("Batch Size:", batch_size)

            # Evaluate model by search type
            for search_mode, dataset in fragments_datasets.items():
                # Create output dir
                dirname = f"{search_mode}-{model_path.stem}"
                model_output_dir = os.path.join(output_dir, dirname)
                os.makedirs(model_output_dir, exist_ok=True)

                # Model predictions
                print("\nSearch Mode:", search_mode)
                merge_detector = MergeDetector(
                    dataset,
                    model,
                    batch_size=batch_size,
                    threshold=threshold,
                )
                merge_detector.search_graph()
                merge_detector.save(model_output_dir, inplace=False)

                # Evaluate performance
                pred_df = load_pred_df(model_output_dir, gt_neurons)
                evaluate_model(gt_df, pred_df, model_output_dir)


# --- Evaluation ---
def evaluate_model(gt_df, pred_df, output_dir):
    # Plot model predictions
    output_path = os.path.join(output_dir, "model_predictions_histogram.png")
    viz.plot_predictions(pred_df["Prediction"].values, output_path=output_path)

    # Create results preamble
    preamble = (
        f"Brain ID: {brain_id}\n"
        f"Segmentation ID: {segmentation_id}\n"
        f"{'-' * (len(segmentation_id) + 17)}\n"
    )

    # Evaluation
    evaluate.prec_recall_curve(gt_df, pred_df, output_dir)
    evaluate.prec_recall_per_neuron(gt_df, pred_df, threshold, output_dir)
    evaluate.prec_recall_at_threshold(
        gt_df, pred_df, threshold, output_dir=output_dir, preamble=preamble
    )

    # Determine threshold for saving sites
    csv_path = os.path.join(output_dir, "results_varying_threshold.csv")
    prec_recall_df = pd.read_csv(csv_path)
    t_90 = evaluate.threshold_at_recall(prec_recall_df, target_recall=0.90)
    evaluate.save_sites(gt_df, pred_df, t_90, output_dir, max_dist=32)


# --- Load Data ---
def load_datasets(gt_names):
    # Load graph
    print("\nStep 2: Load Fragments")
    swc_paths = unique_fragment_paths(gt_names)
    graph = SkeletonGraph(
        anisotropy=(0.748, 0.748, 1.0),
        node_spacing=5,
        use_anisotropy=False,
        verbose=True,
    )
    graph.load(swc_paths)

    # Create datasets
    datasets = dict()
    datasets["Sparse"] = SparseSearchDataset(
        graph,
        img_config,
        min_search_size=min_search_size,
        prefetch=prefetch,
    )
    datasets["Dense"] = DenseSearchDataset(
        graph,
        img_config,
        min_search_size=min_search_size,
        step_size=step_size,
        prefetch=prefetch,
    )
    return datasets


def load_gt_neurons(gt_names):
    print("\nStep 1: Load Ground Truth")
    gt_prefix = f"gs://allen-nd-goog/ground_truth_tracings/{brain_id}"
    gt_neurons = dict()
    for name in tqdm(gt_names, desc="Read SWCs"):
        swc_path = f"{gt_prefix}/world/{name}.swc"
        gt_neurons[name] = utils.load_swc_points(swc_path)
    return gt_neurons


def load_pred_df(output_dir, gt_neurons, max_dist=10):
    # Reads sites csv
    csv_path = os.path.join(output_dir, "model_predictions.csv")
    pred_df = utils.load_sites_df(csv_path)

    # Remove sites far from GT
    gt_kdtree = KDTree(np.vstack(list(gt_neurons.values())))
    dd, _ = gt_kdtree.query(np.stack(pred_df["xyz"].values))
    pred_df = pred_df[dd < max_dist].copy()
    return pred_df


# --- Helpers ---
def get_img_path():
    prefixes_path = "/data/exaspim_image_prefixes.json"
    prefixes = util.read_json(prefixes_path)
    # return os.path.join(prefixes[brain_id], "0")
    return "gs://allen-nd-goog/from_aind/exaSPIM_794495_2026-01-21_14-25-07_training-data/whole-brain-denoised/fused.zarr/0"


def get_top_ckpts(input_dir, k=5):
    def get_f1(path):
        return float(path.stem.split("-")[-1])

    ckpt_paths = sorted(Path(input_dir).rglob("*.pth"))
    return sorted(ckpt_paths, key=get_f1, reverse=True)[:k]


def unique_fragment_paths(gt_names):
    swc_paths = list()
    visited_fragments = set()
    for name in gt_names:
        prefix = os.path.join(fragments_prefix, name)
        for swc_path in util.list_paths(prefix):
            swc_name = os.path.basename(swc_path)
            if swc_name not in visited_fragments:
                swc_paths.append(swc_path)
                visited_fragments.add(swc_name)
    return swc_paths


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Process named arguments")
    parser.add_argument("--brain_id", type=str, required=True)
    parser.add_argument("--segmentation_id", type=str, required=True)
    parser.add_argument("--version", type=str, required=True)
    parser.add_argument("--top_k_ckpts", type=int, required=True)

    parser.add_argument("--min_search_size", type=float, required=True)
    parser.add_argument("--step_size", type=float, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    args = parser.parse_args()

    # Parameters
    brain_id = args.brain_id.strip()
    segmentation_id = args.segmentation_id.strip()
    version = args.version.strip()
    top_k_ckpts = args.top_k_ckpts

    min_search_size = args.min_search_size
    prefetch = 16
    step_size = args.step_size
    threshold = args.threshold

    img_config = ImageConfig(
        brightness_clip=500,
        img_path=get_img_path(),
        patch_shape=(128, 128, 128),
        percentiles=(1, 99.9),
    )

    # Paths
    fragments_prefix = f"gs://allen-nd-goog/anna.grim/intersecting_neuron_fragments/{brain_id}/{segmentation_id}"
    gt_sites_path = f"/data/{brain_id}-{segmentation_id}.csv"
    root_dir = f"/data/train_{version}"

    # Run evaluation
    main()
