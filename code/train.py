"""
Created on Wed June 3 12:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code starting merge detection training session.

"""

import resource
import torch

from neuron_proofreader.configs import GraphConfig, ImageConfig
from neuron_proofreader.machine_learning.vision_models import CNN3D, ViT3D
from neuron_proofreader.machine_learning.train import Trainer
from neuron_proofreader.merge_proofreading.merge_datamodules import (
    create_dataset_collection,
    ThreadedDataLoader,
)
from neuron_proofreader.utils import ml_util


# Raise the soft file descriptor limit to the hard limit to prevent
# "Too many open files" when many threads open GCS credentials concurrently
_soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (_hard, _hard))
print(f"File descriptor limit set to: {_hard}")


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
    print("Batch Size:", batch_size)

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
            random_nonmerge_site_prob=random_nonmerge_site_prob,
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
        model_name,
        output_dir,
        device="cuda",
        min_recall=min_recall,
        lr=lr,
        save_mistake_mips=save_mistake_mips,
        verbose=True
    )
    if model_path:
        trainer.load_pretrained_weights(model_path)
    return trainer


if __name__ == "__main__":
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
    lr = 1e-4
    min_recall = 0.90
    modality = None
    prefetch = 32
    random_nonmerge_site_prob = 0.25
    save_mistake_mips = False

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
    model_name = "MergeDetectorCNN3D"
    model = CNN3D(
        img_config.patch_shape,
        n_conv_layers=5,
        n_feat_channels=24,
        use_double_conv=True
    )

    # Run code
    main()
