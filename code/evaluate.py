"""
Created on Wed June 3 12:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code evaluating merge detection models.

"""

from scipy.spatial import KDTree
from tqdm import tqdm

import argparse
import numpy as np
import os

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
from neuron_proofreader.models.vision_models import CNN3D, ViT3D
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
    print("\nStep 3: Evaluate Models")
    model_names = [f for f in util.list_subdirs(models_dir)]
    print(f"Found {len(model_names)} models in {models_dir}")

    # Evaluate models
    for model_name in model_names:
        # Initializations
        print(f"\n{'='*(len(model_name) + 4)}")
        print(f"  {model_name}")
        print(f"{'='*(len(model_name) + 4)}")

        # Create model
        model = create_model(model_name)
        model_path = get_model_path(model_name)
        ml_util.load_model(model, model_path)

        # Compute batch size
        input_shape = (2,) + img_config.patch_shape
        batch_size = ml_util.find_max_eval_batch_size(model, input_shape)
        print("Batch Size:", batch_size)

        # Evaluate model by search type
        for search_mode, dataset in fragments_datasets.items():
            # Create output dir
            dirname = f"{search_mode}-{os.path.splitext(model_name)[0]}"
            model_output_dir = os.path.join(output_dir, dirname)
            os.makedirs(model_output_dir, exist_ok=True)

            # Model predictions
            print("Search Mode:", search_mode)
            merge_detector = MergeDetector(
                dataset,
                model,
                batch_size=batch_size,
                threshold=args.threshold,
            )
            merge_detector.search_graph()
            merge_detector.save(model_output_dir, inplace=False)

            # Evaluate performance
            pred_df = load_pred_df(model_output_dir, gt_neurons)
            evaluate_model(gt_df, pred_df, model_output_dir)


# --- Evaluation
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
    evaluate.prec_recall_per_neuron(gt_df, pred_df, args.threshold, output_dir)
    evaluate.prec_recall_at_threshold(
        gt_df, pred_df, args.threshold, output_dir=output_dir, preamble=preamble
    )
    evaluate.save_sites(gt_df, pred_df, args.threshold, output_dir, max_dist=32)


# --- Load Data ---
def create_model(model_name):
    # Parse parameters from model name
    path = os.path.join(models_dir, model_name, "model_config.json")
    params = util.read_json(path)

    # Create model
    if "newcnn3d" in model_name.lower():
        input_shape = (2, 128, 128, 128)
        model = NewCNN3D(
            input_shape,
            base_channels=32,
            depth=params["depth"],
            max_channels=256,
            use_resblock=params["use_resblock"],
        )
    else:
        patch_shape = (128, 128, 128)
        model = CNN3D(
            patch_shape,
            n_conv_layers=params["depth"],
            n_feat_channels=24,
        )
    return model


def load_datasets(gt_names):
    # Load graph
    print("\nStep 2: Load Fragments")
    swc_paths = unique_fragment_paths(gt_names)
    graph = SkeletonGraph(
        anisotropy=(0.748, 0.748, 1.0),
        node_spacing=5,
        use_anisotropy=False,
        verbose=True
    )
    graph.load(swc_paths)

    # Create datasets
    datasets = dict()
    datasets["Sparse"] = SparseSearchDataset(
        graph,
        img_config,
        min_search_size=args.min_search_size,
    )
    datasets["Dense"] = DenseSearchDataset(
        graph,
        img_config,
        min_search_size=args.min_search_size,
        step_size=args.step_size,
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
    return pred_df[dd < max_dist]


# --- Helpers ---
def get_img_path():
    prefixes_path = "/root/capsule/data/exaspim_image_prefixes.json"
    prefixes = util.read_json(prefixes_path)
    #return os.path.join(prefixes[brain_id], "0")
    return "gs://allen-nd-goog/from_aind/exaSPIM_794495_2026-01-21_14-25-07_training-data/whole-brain-denoised/fused.zarr/0"


def get_model_path(model_name):
    src_dir = os.path.join(models_dir, model_name)
    paths = util.listdir(src_dir, extension=".pth")
    assert len(paths) == 1
    return os.path.join(src_dir, paths[0])


def unique_fragment_paths(gt_names):
    swc_paths = list()
    visited_fragments = set()
    n_visited = 0
    for name in gt_names:
        prefix = os.path.join(fragments_prefix, name)
        for swc_path in util.list_paths(prefix):
            n_visited += 1
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
    parser.add_argument("--model_version", type=str, required=True)
    parser.add_argument("--min_search_size", type=float, required=True)
    parser.add_argument("--step_size", type=float, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    args = parser.parse_args()

    # Parameters
    brain_id = args.brain_id.strip()
    segmentation_id = args.segmentation_id.strip()
    model_version = args.model_version.strip()

    img_config = ImageConfig(
        brightness_clip=500,
        img_path=get_img_path(),
        patch_shape=(128, 128, 128),
        percentiles=(1, 99.9),
    )

    # Paths
    fragments_prefix = f"gs://allen-nd-goog/anna.grim/intersecting_neuron_fragments/{brain_id}/{segmentation_id}"
    gt_sites_path = f"/root/capsule/data/{brain_id}-{segmentation_id}.csv"
    models_dir = f"/root/capsule/data/{model_version}"
    output_dir = f"/root/capsule/results"

    # Run evaluation
    main()
