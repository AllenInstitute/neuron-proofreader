"""
Created on Fri Aug 15 12:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Models used for merge and split proofreading tasks.
"""

import torch
import torch.nn as nn

from arborist.models.arborist import Arborist
from neuron_proofreader.models.vision_models import CNN3D, ViT3D
from neuron_proofreader.utils.ml_util import FeedForwardNet


class ArboristMergeDetector(nn.Module):
    """
    Multimodal merge detector combining a 3D vision backbone with the Arborist
    morphology encoder.
    """

    def __init__(
        self,
        vision_input_shape,
        output_dim=1,
        arborist_latent_dim=32,
        arborist_kwargs=None,
        vision_backbone="CNN3D",
        vision_latent_dim=128,
        dropout=0.2,
        **vision_backbone_kwargs,
    ):
        """
        Instantiates an ArboristMergeDetector object.

        Parameters
        ----------
        vision_input_shape : Tuple[int]
            Shape of one node's image patch: (C, D, H, W).
        output_dim : int, optional
            Output dimension of the classifier head. Default is 1.
        vision_latent_dim : int, optional
            Output dimension of the vision backbone. Default is 128.
        vision_backbone : str, optional
            Vision backbone to use: "CNN3D" or "ViT3D". Default is "CNN3D".
        arborist_latent_dim : int, optional
            Latent dimension passed to Arborist and the size of z_tree.
            Default is 32.
        arborist_kwargs : dict or None, optional
            Extra keyword arguments forwarded to Arborist.__init__. Uses
            Arborist defaults when not specified.
        **vision_backbone_kwargs
            Extra keyword arguments forwarded to the image backbone constructor.
        """
        super().__init__()

        _arborist_kwargs = arborist_kwargs or {}

        self.config = {
            "model_type": "ArboristMergeDetector",
            "vision_input_shape": tuple(vision_input_shape),
            "vision_latent_dim": vision_latent_dim,
            "output_dim": output_dim,
            "vision_backbone": vision_backbone,
            "arborist_latent_dim": arborist_latent_dim,
            "arborist_kwargs": _arborist_kwargs,
            "dropout": dropout,
            **vision_backbone_kwargs,
        }

        # Vision backbone — shared image-patch encoder
        if vision_backbone == "ViT3D":
            self.vision = ViT3D(
                vision_input_shape,
                output_dim=vision_latent_dim,
                **vision_backbone_kwargs,
            )
        else:
            self.vision = CNN3D(
                vision_input_shape,
                output_dim=vision_latent_dim,
                **vision_backbone_kwargs,
            )

        # Projection when the backbone returns raw pooled features
        self.vision_proj = None
        if isinstance(getattr(self.vision, "output", None), nn.Identity):
            self.vision_proj = nn.Sequential(
                nn.Linear(self.vision.feature_dim, vision_latent_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        # Arborist skeleton encoder — produces z_tree per subgraph
        self.arborist = Arborist(
            latent_dim=arborist_latent_dim, **_arborist_kwargs
        )

        # Fusion classification head
        self.drop = nn.Dropout(dropout)
        self.head = FeedForwardNet(
            vision_latent_dim + arborist_latent_dim, output_dim, 3
        )

    def forward(self, x):
        z_img = self.vision(x["img"])
        if self.vision_proj is not None:
            z_img = self.vision_proj(z_img)
        z_tree = self._encode_tree_samples(x["tree_sample"], z_img.device)
        return self.head(self.drop(torch.cat([z_img, z_tree], dim=1)))

    # --- Helpers
    def save(self, path):
        torch.save(
            {"config": self.config, "state_dict": self.state_dict()}, path
        )

    @classmethod
    def load(cls, path, map_location=None):
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
        config = {k: v for k, v in ckpt["config"].items() if k != "model_type"}
        if config.get("vision_backbone", "CNN3D") == "CNN3D":
            config = CNN3D.upgrade_config(config)
        model = cls(**config)
        model.load_state_dict(ckpt["state_dict"])
        return model

    @torch._dynamo.disable
    def _encode_tree_samples(self, tree_samples, device):
        with torch.amp.autocast("cuda", enabled=False):
            z_trees = [self.arborist.encode(s)[0] for s in tree_samples]
        return torch.stack(z_trees).to(device)
