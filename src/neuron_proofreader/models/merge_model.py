"""
Created on Tue Nov 24 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Merge detection model: MultimodalEncoder with a FeedForwardNet head.
"""

import torch
import torch.nn as nn

from neuron_proofreader.models.encoders import MultimodalEncoder
from neuron_proofreader.utils.ml_util import FeedForwardNet


class MergeModel(nn.Module):
    """
    Merge detector: a MultimodalEncoder embedding of the candidate site
    classified by a FeedForwardNet. Keyword arguments other than
    ``dropout`` are those of MultimodalEncoder.

    Parameters
    ----------
    patch_shape : Tuple[int]
        Image patch shape (C, D, H, W).
    dropout : float, optional
        Dropout in the vision projection and before the head. Default is
        0.1.
    """

    def __init__(self, patch_shape, dropout=0.1, **encoder_kwargs):
        """Instantiates a MergeModel object."""
        super().__init__()
        self.config = {
            "patch_shape": tuple(patch_shape),
            "dropout": dropout,
            **encoder_kwargs,
        }
        self.encoder = MultimodalEncoder(
            patch_shape, dropout=dropout, **encoder_kwargs
        )
        self.drop = nn.Dropout(dropout)
        self.head = FeedForwardNet(self.encoder.embed_dim, 1, 3)

    def forward(self, x):
        """
        Parameters
        ----------
        x : dict
            Batch with "img" (B, C, D, H, W) and "tree_sample"
            (List[TreeSample]).

        Returns
        -------
        torch.Tensor
            Logits of shape (B, 1).
        """
        return self.head(self.drop(self.encoder(x["img"], x["tree_sample"])))

    def half_vision(self):
        """Runs the image path in fp16; see MultimodalEncoder.half_vision."""
        self.encoder.half_vision()
        return self

    def save(self, path):
        """Saves the config and weights to a checkpoint."""
        state = {"config": self.config, "state_dict": self.state_dict()}
        torch.save(state, path)

    @classmethod
    def load(cls, path, map_location=None):
        """
        Reconstructs a MergeModel from a checkpoint written by ``save``.

        Parameters
        ----------
        path : str
            Path to the checkpoint.
        map_location : str or torch.device, optional
            Passed through to torch.load. Default is None.

        Returns
        -------
        MergeModel
        """
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
        model = cls(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        return model
