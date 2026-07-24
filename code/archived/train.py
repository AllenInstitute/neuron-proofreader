"""
Created on Sat Sept 16 11:30:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code that trains a model that performs merge detection. This code assumes that
there is a local directory with the following files:
    - "merge_sites_df.csv"
          DataFrame containing merge sites, must contain the columns:
          "brain_id", "segmentation_id", "segment_id", and "xyz".
    - "train_idxs.csv"
          Row indices from merge_sites_df to use as training examples.
    - "val_idxs.csv"
          Row indices from merge_sites_df to use as validation examples.

Note: Use the following command to train with multiple GPUs
        torchrun --nproc_per_node=4 train_merge_detector.py
"""

from torch.utils.data import DistributedSampler

import numpy as np
import os

from neuron_proofreader.machine_learning.train import DistributedTrainer, Trainer
from neuron_proofreader.machine_learning.point_cloud_models import VisionDGCNN
from neuron_proofreader.machine_learning.vision_models import CNN3D, ViT3D
from neuron_proofreader.merge_proofreading import merge_dataloading as data_util
from neuron_proofreader.merge_proofreading.merge_datasets import (
    MergeSiteDataset,
    MergeSiteTrainDataset,
    MergeSiteValDataset,
    MergeSiteDataLoader,
)
from neuron_proofreader.utils import util

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/root/capsule/data/gcs-token.json"


def main():
    # Set data paths
    sites_path = os.path.join(dataset_path, "merge_sites_df.csv")
    train_idxs_path = os.path.join(dataset_path, "train_idxs.csv")
    val_idxs_path = os.path.join(dataset_path, "val_idxs.csv")

    # Load data
    merge_sites_df = data_util.load_merge_sites_df(sites_path, is_test=is_test)
    if is_test:
        train_idxs = np.arange((merge_sites_df["brain_id"] == "653159").sum())
        val_idxs = np.arange((merge_sites_df["brain_id"] == "653159").sum())
    else:
        train_idxs = data_util.read_idxs(train_idxs_path)
        val_idxs = data_util.read_idxs(val_idxs_path)

    print("# Train Examples:", len(train_idxs))
    print("# Validate Examples:", len(val_idxs))

    # Dataset
    dataset = init_dataset(merge_sites_df)
    train_dataset = MergeSiteTrainDataset(dataset, train_idxs, negative_bias)
    val_dataset = MergeSiteValDataset(dataset, val_idxs)
    print("len(train_dataset):", len(train_dataset))
    print("len(val_dataset):", len(val_dataset))
    del dataset

    # Dataloaders
    trainer = init_trainer()
    val_dataset.save_summary(trainer.log_dir)

    train_dataloader = MergeSiteDataLoader(
        train_dataset,
        batch_size=batch_size,
        is_multimodal=is_multimodal,
        modality="pointcloud",
    )
    val_dataloader = MergeSiteDataLoader(
        val_dataset,
        batch_size=2 * batch_size,
        is_multimodal=is_multimodal,
        modality="pointcloud",
        use_shuffle=False,
    )

    # Train
    print("\nModel Name:", model_name)
    save_experiment_parameters()
    trainer.run(train_dataloader, val_dataloader)


def init_dataset(merge_sites_df):
    # Initialize dataset
    dataset = MergeSiteDataset(
        merge_sites_df,
        anisotropy=anisotropy,
        brightness_clip=brightness_clip,
        node_spacing=5,
        patch_shape=patch_shape,
        use_new_mask=use_new_mask,
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
    return dataset


def init_trainer():
    trainer = Trainer(
        model,
        model_name,
        output_dir,
        lr=lr,
        max_epochs=max_epochs,
        min_recall=min_recall,
        save_mistake_mips=save_mistake_mips,
    )
    if model_path:
        trainer.load_pretrained_weights(model_path)
    return trainer


def save_experiment_parameters():
    parameters = {
        "batch_size": batch_size,
        "brightness_clip": brightness_clip,
        "lr": lr,
        "is_finetuned": model_path != None,
        "is_multimodal": is_multimodal,
        "min_recall": min_recall,
        "model_class": model_class,
        "model_path": model_path,
        "negative_bias": negative_bias,
        "patch_shape": patch_shape,
        "use_new_mask": use_new_mask,
    }
    path = os.path.join(output_dir, "experiment_parameters.json")
    util.write_json(path, parameters)


if __name__ == "__main__":
    # Parameters
    anisotropy = (0.748, 0.748, 1.0)
    batch_size = 16
    brightness_clip = 400
    is_test = False
    lr = 1e-4
    max_epochs = 100
    min_recall = 0.85
    model_class = "CNN3D"
    negative_bias = 0.1
    patch_shape = (128, 128, 128)
    save_mistake_mips = True
    use_new_mask = True

    # Paths
    dataset_path = "/root/capsule/data/V5"
    image_prefixes_path = "/root/capsule/data/exaspim_image_prefixes.json"
    segmentation_prefixes_path = "/root/capsule/data/exaspim_segmentation_prefixes.json"
    model_path = None
    output_dir = "/root/capsule/results"

    # Model
    model_name = f"MergeDetector{model_class}-v6-run2-newmask={use_new_mask}"
    if model_class == "CNN3D":
        print("Model Class: CNN3D")
        is_multimodal = False
        model = CNN3D(
            patch_shape,
            n_conv_layers=6,
            n_feat_channels=24,
        )
    elif model_class == "DGCNN":
        print("Model Class: VisionDGCNN")
        is_multimodal = True
        model = VisionDGCNN(patch_shape)
    else:
        raise ValueError(f"model_class={model_class} is not valid!")

    # Main
    main()
