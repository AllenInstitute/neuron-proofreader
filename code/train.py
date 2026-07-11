"""
Created on Wed June 3 12:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code starting merge detection training session.

"""

import argparse
import torch

from neuron_proofreader.configs import GraphConfig, ImageConfig
from neuron_proofreader.models.new_vision_models import NewCNN3D
from neuron_proofreader.models.vision_models import CNN3D, ViT3D
from neuron_proofreader.machine_learning.train import Trainer
from neuron_proofreader.merge_proofreading.merge_datamodules import (
    create_dataset_collection,
    ThreadedDataLoader,
)
from neuron_proofreader.utils import ml_util


def main():
    # Save experiment parameters
    graph_config.save(output_dir)
    img_config.save(output_dir)

    # Determine batch size
    input_shape = (2,) + img_config.patch_shape
    batch_size = ml_util.find_max_batch_size(
        model,
        input_shape=input_shape,
        optimizer_cls=torch.optim.AdamW,
        device="cuda",
    )
    batch_size -= 1
    print("Batch Size:", batch_size)
    assert batch_size > 0

    # Create datasets
    train_dataset, val_dataset = [
        create_dataset_collection(
            brain_ids,
            ds_mode,
            img_prefixes_path,
            sites_root_path,
            swcs_root_path,
            graph_config=graph_config,
            img_config=img_config,
            random_nonmerge_site_prob=args.random_nonmerge_site_prob,
        )
        for ds_mode, brain_ids in zip(["Train", "Val"], [train_ids, val_ids])
    ]
    print("\nDataset Summary...")
    print("Train Dataset:", train_dataset)
    print("Val Dataset:", val_dataset)

    # Create dataloaders
    train_loader_kwargs = dict(
        batch_size=batch_size,
        is_multimodal=is_multimodal,
        modality=modality,
        prefetch=prefetch,
    )
    val_loader_kwargs = dict(
        batch_size=batch_size,
        is_multimodal=is_multimodal,
        modality=modality,
        prefetch=prefetch,
        shuffle=False
    )
    train_dataloader = ThreadedDataLoader(train_dataset, **train_loader_kwargs)
    val_dataloader = ThreadedDataLoader(val_dataset, **val_loader_kwargs)

    # Train Model
    trainer = init_trainer()
    trainer.run(train_dataloader, val_dataloader)


# --- Helpers ---
def init_trainer():
    trainer = Trainer(
        model,
        args.model_name,
        output_dir,
        device="cuda",
        min_recall=args.min_recall,
        lr=args.lr,
        verbose=True
    )
    if model_path:
        trainer.load_pretrained_weights(model_path)
    return trainer


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Process named arguments")
    parser.add_argument("--base_channels", type=int, required=True)
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--max_channels", type=int, required=True)
    parser.add_argument("--min_recall", type=float, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--random_nonmerge_site_prob", type=float, default=0.25)
    parser.add_argument("--use_resblock", type=bool, required=True)
    args = parser.parse_args()

    # Dataset
    train_ids = ["653159", "715345", "730902", "789202", "794491", "802449"]
    val_ids = ["751473", "794495"]

    # Paths
    img_prefixes_path = "/root/capsule/data/exaspim_image_prefixes.json"
    model_path = None
    output_dir = "/root/capsule/results"
    sites_root_path = "gs://allen-nd-goog/automated_proofreading_dataset/curated_sites_05202026/"
    swcs_root_path = sites_root_path  #"gs://allen-nd-goog/from_google/"

    # Parameters
    is_multimodal = False
    modality = None
    prefetch = 32

    graph_config = GraphConfig(
        anisotropy=(0.748, 0.748, 1.0),
        min_cable_length=0,
        node_spacing=5,
        use_anisotropy=False,
        verbose=True,
    )
    img_config = ImageConfig(
        brightness_clip=500,
        patch_shape=(128, 128, 128),
        percentiles=(1, 99.9),
    )

    # Model architecture
    print("Model Name:", args.model_name)
    if "new" in args.model_name.lower():
        input_shape = (2,) + img_config.patch_shape
        model = NewCNN3D(
            input_shape,
            base_channels=args.base_channels,
            depth=args.depth,
            max_channels=args.max_channels,
            use_double=True,
            use_resblock=args.use_resblock,
        )
    else:    
        model = CNN3D(
            img_config.patch_shape,
            n_conv_layers=args.depth,
            n_feat_channels=args.base_channels,
            use_double_conv=True,
        )

    # Run code
    main()
