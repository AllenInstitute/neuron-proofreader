"""
Created on Fri Aug 15 12:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Models used for merge and split proofreading tasks.
"""

import torch
import torch.nn as nn

from arborist.models.arborist import Arborist
from neuron_proofreader.models.new_vision_models import CNN3D, ViT3D
from neuron_proofreader.utils.ml_util import FeedForwardNet


class ArboristVisionMergeModel(nn.Module):
    """
    Multimodal merge detector combining a 3D vision backbone with the Arborist
    morphology encoder.
    """

    def __init__(
        self,
        input_shape,
        vision_latent_dim=128,
        output_dim=1,
        vision_backbone="CNN3D",
        arborist_latent_dim=32,
        arborist_kwargs=None,
        pretrained_curve_encoder_path=None,
        freeze_curve_encoder=False,
        **vision_backbone_kwargs,
    ):
        """
        Parameters
        ----------
        input_shape : Tuple[int]
            Shape of one node's image patch: (C, D, H, W).
        vision_latent_dim : int, optional
            Output dimension of the vision backbone. Default is 128.
        output_dim : int, optional
            Output dimension of the classifier head. Default is 1.
        vision_backbone : str, optional
            Vision backbone to use: "CNN3D" or "ViT3D". Default is "CNN3D".
        arborist_latent_dim : int, optional
            Latent dimension passed to Arborist and the size of z_tree.
            Default is 32.
        arborist_kwargs : dict or None, optional
            Extra keyword arguments forwarded to Arborist.__init__. Defaults
            match the graph-only MergeDetector: d_ff_curve=512, d_ff_graph=64.
        pretrained_curve_encoder_path : str or None, optional
            Path to a curve autoencoder checkpoint. If given, the encoder
            weights are extracted and loaded into Arborist.curve_encoder.
        freeze_curve_encoder : bool, optional
            If True, freeze the curve encoder after loading pretrained weights.
            Default is False.
        **vision_backbone_kwargs
            Extra keyword arguments forwarded to the image backbone constructor.
        """
        super().__init__()

        # Merge caller-supplied arborist kwargs on top of graph-model defaults
        _arborist_kwargs = {"d_ff_curve": 512, "d_ff_graph": 64}
        _arborist_kwargs.update(arborist_kwargs or {})

        self.config = {
            "model_type": "ArboristVisionMergeModel",
            "input_shape": tuple(input_shape),
            "vision_latent_dim": vision_latent_dim,
            "output_dim": output_dim,
            "vision_backbone": vision_backbone,
            "arborist_latent_dim": arborist_latent_dim,
            "arborist_kwargs": _arborist_kwargs,
            **vision_backbone_kwargs,
        }

        # Vision backbone — shared image-patch encoder
        if vision_backbone == "ViT3D":
            self.vision = ViT3D(
                input_shape,
                output_dim=vision_latent_dim,
                **vision_backbone_kwargs,
            )
        else:
            self.vision = CNN3D(
                input_shape,
                output_dim=vision_latent_dim,
                **vision_backbone_kwargs,
            )

        # Arborist skeleton encoder — produces z_tree per subgraph
        self.arborist = Arborist(
            latent_dim=arborist_latent_dim, **_arborist_kwargs
        )

        # Load pre-trained curve encoder weights if provided
        if pretrained_curve_encoder_path:
            self._load_curve_encoder(pretrained_curve_encoder_path)
        if freeze_curve_encoder:
            for p in self.arborist.curve_encoder.parameters():
                p.requires_grad_(False)

        # Fusion classification head
        self.head = FeedForwardNet(
            vision_latent_dim + arborist_latent_dim, output_dim, 3
        )

    def forward(self, x):
        z_img = self.vision(x["img"])
        z_tree = self._encode_tree_samples(x["tree_sample"], z_img.device)
        return self.head(torch.cat([z_img, z_tree], dim=1))

    # --- Helpers
    def _load_curve_encoder(self, path):
        ckpt = torch.load(path, map_location="cpu")
        encoder_state = {
            k[len("encoder."):]: v
            for k, v in ckpt.items()
            if k.startswith("encoder.")
        }
        self.arborist.curve_encoder.load_state_dict(encoder_state)

    def save(self, path):
        torch.save(
            {"config": self.config, "state_dict": self.state_dict()}, path
        )

    @classmethod
    def load(cls, path, map_location=None):
        ckpt = torch.load(path, map_location=map_location)
        config = {k: v for k, v in ckpt["config"].items() if k != "model_type"}
        model = cls(**config)
        model.load_state_dict(ckpt["state_dict"])
        return model

    @torch._dynamo.disable
    def _encode_tree_samples(self, tree_samples, device):
        z_trees = [self.arborist.encode(s)[0] for s in tree_samples]
        return torch.stack(z_trees).to(device)
