"""
Created on Wed June 3 12:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code to train merge detection models.

"""

from neuron_proofreader.configs import GraphConfig, ImageConfig
from neuron_proofreader.machine_learning.vision_models import CNN3D, ViT3D
from neuron_proofreader.machine_learning.train import Trainer
from neuron_proofreader.merge_proofreading.merge_datamodules import (
    create_dataset_collection,
    ThreadedDataLoader,
)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/root/capsule/data/gcs-token.json"


def main():
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
        )
        for ds_mode, brain_ids in zip(["Train", "Val"], [train_ids, val_ids])
    ]
    print("Dataset Summary...")
    print("  Train Dataset:", train_dataset)
    print("  Val Dataset:", val_dataset)

    # Create dataloaders
    loader_kwargs = dict(
        batch_size=batch_size,
        is_multimodal=is_multimodal,
        modality=modality,
        prefetch=prefetch,
    )
    train_dataloader = ThreadedDataLoader(train_dataset, **loader_kwargs)
    val_dataloader = ThreadedDataLoader(val_dataset, **loader_kwargs)

    # Train Model
    trainer = init_trainer()
    trainer.run(train_dataloader, val_dataloader)


# --- Helpers ---
def init_trainer():
    trainer = Trainer(
        model,
        model_name,
        output_dir,
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
    output_dir = "/root/capsule/results/"
    sites_root_path = "gs://allen-nd-goog/automated_proofreading_dataset/curated_sites_05202026/"
    swcs_root_path = "gs://allen-nd-goog/from_google/"

    # Parameters
    batch_size = 16
    is_multimodal = False
    lr = 1e-4
    modality = None
    prefetch = 8
    save_mistake_mips = True

    graph_config = GraphConfig(
        anisotropy=(0.748, 0.748, 1.0),
        min_cable_length=600,
        min_swc_pts=160,
        node_spacing=6,
        use_anisotropy=True,
        verbose=True,
    )
    img_config = ImageConfig(
        brightness_clip=400,
        percentiles=(1, 99.5),
        patch_shape=(128, 128, 128),
    )

    graph_config.save(output_dir)
    img_config.save(output_dir)

    # Model architecture
    model_name = "MergeDetectorCNN3D"
    model = CNN3D(
        img_config.patch_shape,
        n_conv_layers=6,
        n_feat_channels=24,
        use_double_conv=True
    )

    # Run code
    main()
