"""
Created on Tue Nov 24 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Merge detection model: MultimodalEncoder with a FeedForwardNet head.
"""

import torch.nn as nn

from neuron_proofreader.models.proofreading_models import ProofreadingModel
from neuron_proofreader.utils.ml_util import FeedForwardNet


class MergeModel(ProofreadingModel):
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
        super().__init__(patch_shape, dropout=dropout, **encoder_kwargs)
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
