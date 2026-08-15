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


class MultimodalMergeModel(nn.Module):
    """
    Multimodal merge detector combining a 3D vision backbone with the Arborist
    morphology encoder.

    For each candidate site the vision backbone encodes its image patch and the
    Arborist model encodes its local skeleton subgraph into a graph-level
    embedding (z_tree). The two embeddings are concatenated and classified by a
    small MLP head.

    Inputs (forward)
    ----------------
    x : dict
        "img"         : (B, C, D, H, W) — image patches, one per candidate.
        "tree_sample" : List[TreeSample] of length B — Arborist subgraph
                        samples built by search_datasets.subgraph_to_tree_sample.
    """

    def __init__(
        self,
        input_shape,
        embed_dim=128,
        output_dim=1,
        backbone="CNN3D",
        arborist_latent_dim=64,
        arborist_kwargs=None,
        **backbone_kwargs,
    ):
        """
        Parameters
        ----------
        input_shape : tuple
            Shape of one node's image patch: (C, D, H, W).
        embed_dim : int, optional
            Output dimension of the vision backbone. Default is 128.
        output_dim : int, optional
            Output dimension of the classifier head. Default is 1.
        backbone : str, optional
            Vision backbone to use: "CNN3D" or "ViT3D". Default is "CNN3D".
        arborist_latent_dim : int, optional
            Latent dimension passed to Arborist and the size of z_tree.
            Default is 64.
        arborist_kwargs : dict or None, optional
            Extra keyword arguments forwarded to Arborist.__init__.
        **backbone_kwargs
            Extra keyword arguments forwarded to the backbone constructor.
        """
        super().__init__()

        self.config = {
            "model_type": "MultimodalMergeModel",
            "input_shape": tuple(input_shape),
            "embed_dim": embed_dim,
            "output_dim": output_dim,
            "backbone": backbone,
            "arborist_latent_dim": arborist_latent_dim,
            "arborist_kwargs": arborist_kwargs or {},
            **backbone_kwargs,
        }

        # Vision backbone — shared image-patch encoder
        if backbone == "ViT3D":
            self.vision = ViT3D(input_shape, output_dim=embed_dim, **backbone_kwargs)
        else:
            self.vision = CNN3D(input_shape, output_dim=embed_dim, **backbone_kwargs)

        # Arborist skeleton encoder — produces z_tree per subgraph
        self.arborist = Arborist(
            latent_dim=arborist_latent_dim, **(arborist_kwargs or {})
        )

        # Fusion classification head
        self.head = FeedForwardNet(embed_dim + arborist_latent_dim, output_dim, 3)

    def save(self, path):
        torch.save({"config": self.config, "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path, map_location=None):
        ckpt = torch.load(path, map_location=map_location)
        config = {k: v for k, v in ckpt["config"].items() if k != "model_type"}
        model = cls(**config)
        model.load_state_dict(ckpt["state_dict"])
        return model

    def forward(self, x):
        imgs = x["img"]                  # (B, C, D, H, W)
        tree_samples = x["tree_sample"]  # List[TreeSample], length B

        # Vision embeddings — single batched forward pass
        z_img = self.vision(imgs)        # (B, embed_dim)

        # Arborist embeddings — sequential; model processes one sample at a time
        z_trees = [self.arborist.encode(s)[0] for s in tree_samples]
        z_tree = torch.stack(z_trees).to(z_img.device)  # (B, arborist_latent_dim)

        return self.head(torch.cat([z_img, z_tree], dim=1))
