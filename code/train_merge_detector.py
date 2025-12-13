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

from neuron_proofreader.machine_learning.train import (
    DistributedTrainer, Trainer
)
from neuron_proofreader.machine_learning.point_cloud_models import VisionDGCNN
from neuron_proofreader.machine_learning.vision_models import CNN3D, ViT3D
from neuron_proofreader.merge_proofreading import merge_dataloading as data_util
from neuron_proofreader.merge_proofreading.merge_datasets import (
    MergeSiteDataset,
    MergeSiteTrainDataset,
    MergeSiteValDataset,
    MergeSiteDataLoader
)


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
    train_dataset = MergeSiteTrainDataset(dataset, train_idxs)
    val_dataset = MergeSiteValDataset(dataset, val_idxs)
    print("len(train_dataset):", len(train_dataset))
    print("len(val_dataset):", len(val_dataset))
    del dataset

    # Dataloaders
    trainer = init_trainer()
    sampler = init_sampler(trainer, train_dataset)
    val_dataset.save_summary(trainer.log_dir)

    train_dataloader = MergeSiteDataLoader(
        train_dataset,
        batch_size=batch_size,
        is_multimodal=is_multimodal,
        sampler=sampler,
    )
    val_dataloader = MergeSiteDataLoader(
        val_dataset,
        batch_size=2*batch_size,
        is_multimodal=is_multimodal,
        use_shuffle=False
    )

    # Train
    trainer.run(train_dataloader, val_dataloader)


def init_dataset(merge_sites_df):
    # Initialize dataset
    dataset = MergeSiteDataset(
        merge_sites_df,
        anisotropy=anisotropy,
        node_spacing=5,
        patch_shape=patch_shape,
    )

    # Load data
    data_util.load_groundtruth(dataset, merge_sites_df, is_test=is_test)
    data_util.load_fragments(dataset, merge_sites_df, is_test=is_test)
    data_util.load_images(
        dataset,
        merge_sites_df,
        is_test=is_test,
        prefix_lookup_path=prefix_lookup_path
    )
    return dataset


def init_trainer():
    TrainerClass = DistributedTrainer if use_distributed else Trainer
    trainer = TrainerClass(
        model,
        model_name,
        output_dir,
        batch_size=batch_size,
        device=device,
        lr=lr,
        save_mistake_mips=save_mistake_mips
    )
    if model_path:
        trainer.load_pretrained_weights(model_path)
    return trainer


def init_sampler(trainer, dataset):
    if use_distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=trainer.world_size,
            rank=trainer.rank,
        )
    else:
        sampler = None
    return sampler


if __name__ == "__main__":
    # Paths
    dataset_path = "/root/capsule/data"
    prefix_lookup_path = "/root/capsule/data/exaspim_image_prefixes.json"
    model_path = None
    output_dir = "/root/capsule/results"

    # Parameters
    anisotropy = (0.748, 0.748, 1.0)
    device = "cuda"
    batch_size = 20
    is_multimodal = False
    is_test = False
    lr = 1e-4
    patch_shape = (128, 128, 128)
    save_mistake_mips = True
    use_distributed = False

    # Model
    model_name = "MergeDetectorVisionDGCNN"
    model = VisionDGCNN(patch_shape)

    # Main
    main()
