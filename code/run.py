"""
Created on Fri Dec 5 12:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code for running merge detection on whole-brain datasets.

"""

from datetime import datetime

import argparse
import os

from neuron_proofreader.machine_learning.vision_models import CNN3D
from neuron_proofreader.merge_proofreading.merge_inference import MergeDetector, DenseGraphDataset
from neuron_proofreader.skeleton_graph import SkeletonGraph
from neuron_proofreader.utils import img_util, util

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/root/capsule/data/gcs-token.json"


def main():
    # Build dataset
    print("\nStep 1: Build Graph")
    dataset = load_data()

    # Run detection
    print("\nStep 2: Run Merge Detection")
    merge_detector = MergeDetector(
        dataset,
        model,
        model_path,
        device="cuda",
        remove_detected_sites=remove_detected_sites,
        threshold=accept_threshold,
    )
    merge_detector.search_graph()

    # Save results
    print("\nStep 3: Save Results")
    save_results(merge_detector)


def load_data():
    # Load graphs
    graph = SkeletonGraph(
        anisotropy=anisotropy,
        node_spacing=node_spacing,
        min_cable_length=min_cable_length,
        verbose=True
    )
    graph.load(swc_pointer)

    # Initialize dataset
    img_path = img_util.find_img_path("allen-nd-goog", "from_aind/", brain_id) + "/0"
    img_path = img_path.replace("whole-brain", "whole-brain-raw")
    #img_path = "gs://allen-nd-goog/from_aind/exaSPIM_789202_2026-02-03_15-09-59_training-data/whole-brain-raw/fused.zarr/0"

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


def save_results(merge_detector):
    # Save fragments
    fragments_dir = os.path.join(output_dir, "predictions")
    fragments_path = os.path.join(fragments_dir, "predictions.zip")
    util.mkdir(fragments_dir)

    # Save predicted merge sites
    date = datetime.today().strftime("%Y%m%d_%H%M")
    output_prefix_s3 = f"s3://aind-msma-morphology-data/anna.grim/merge_detection/{brain_id}/result_{segmentation_id}_{date}"
    merge_detector.save_train_dataset(output_dir)
    merge_detector.save_parameters(output_dir)


if __name__ == "__main__":
    # Parameters
    brain_id = "794495"
    segmentation_id = "raw.unet_449_792_202_splits_and_merges_831600"
    model_name = "MergeDetectorCNN3D-v5-run2-newmask=False-negativebias=0.15-20260201-137-0.8771"

    accept_threshold = 0.5
    anisotropy = (0.748, 0.748, 1.0)
    batch_size = 16
    device = "cuda"
    is_multimodal = False
    min_cable_length = 50
    min_search_size = 10**4
    node_spacing = 5
    patch_shape = (128, 128, 128)
    remove_detected_sites = False  # not implemented
    step_size = 20

    # Paths
    model_path = f"/root/capsule/data/V5/{model_name}.pth"
    output_dir = f"/root/capsule/results"
    swc_pointer = f"gs://allen-nd-goog/from_google/{brain_id}/whole_brain/{segmentation_id}/swcs"

    # Model
    model = CNN3D(
        patch_shape,
        n_conv_layers=6,
        n_feat_channels=24,
    )

    # Run detection
    main()
