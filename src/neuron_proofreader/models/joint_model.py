"""
Created on Fri Sep 25 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Joint model for simultaneous split and merge correction.

"""

import torch
import torch.nn as nn

from arborist.models.arborist import Arborist
from neuron_proofreader.models.new_gnn_models import _build_hetero_gat
from neuron_proofreader.models.new_vision_models import CNN3D
from neuron_proofreader.split_proofreading import split_feature_extraction
from neuron_proofreader.utils.ml_util import FeedForwardNet


class JointProofreader(nn.Module):
    """
    Joint model for split and merge proofreading with a shared 3D vision
    backbone.

    Architecture
    ------------
    Shared
        CNN3D vision backbone — encodes a 2-channel image patch to a
        fixed-size embedding z_img for every example regardless of task.

    Merge head  (all merge examples; MLP only)
        Arborist skeleton encoder produces z_tree; a linear classifier then
        acts on cat(z_img, z_tree).  Merge sites are treated independently
        because the correction decision at one site does not depend on other
        sites — an MLP is sufficient.

    Split head  (split examples only; HGAT with message passing)
        Proposal and branch node geometry features are embedded and fused with
        z_img from the shared backbone.  Two rounds of heterogeneous GAT
        message-passing let proposals exchange context through the fragment
        graph before a linear layer classifies each proposal.  Message passing
        is necessary because split proposals are correlated: whether to merge
        fragment A with B affects the A-C decision when A, B, C are nearby
        — proposals must communicate to make globally consistent choices.
    """

    def __init__(
        self,
        vision_input_shape,
        img_embed_dim=64,
        arborist_latent_dim=64,
        arborist_kwargs=None,
        geometry_embed_dim=64,
        split_heads=4,
        dropout=0.1,
        **vision_backbone_kwargs,
    ):
        """
        Instantiates a JointProofreader object.

        Parameters
        ----------
        vision_input_shape : Tuple[int]
            Shape of one image patch: (C, D, H, W).  Both tasks share this
            format — 2 channels (raw image + segmentation mask).
        img_embed_dim : int, optional
            Output dimension of the shared vision backbone.  Default is 64.
        arborist_latent_dim : int, optional
            Latent dimension for the Arborist skeleton encoder used by the
            merge head.  Default is 64.
        arborist_kwargs : dict or None, optional
            Extra keyword arguments forwarded to Arborist.__init__.
        geometry_embed_dim : int, optional
            Hidden dimension for split node geometry embeddings.  Default is 64.
        split_heads : int, optional
            Number of GAT attention heads in the split head.  Default is 4.
        dropout : float, optional
            Dropout probability applied before the merge classification head.
            Default is 0.1.
        **vision_backbone_kwargs
            Forwarded to CNN3D (e.g. base_channels, depth, block_type).
        """
        super().__init__()

        # Shared 3D vision backbone.
        self.vision = CNN3D(
            vision_input_shape,
            output_dim=img_embed_dim,
            use_output_head=True,
            **vision_backbone_kwargs,
        )

        # Merge head: Arborist morphology encoder + fusion MLP.
        _arborist_kwargs = arborist_kwargs or {}
        self.arborist = Arborist(latent_dim=arborist_latent_dim, **_arborist_kwargs)
        self.drop = nn.Dropout(dropout)
        self.merge_head = FeedForwardNet(img_embed_dim + arborist_latent_dim, 1, 3)

        # Split head: node geometry embedding + heterogeneous GAT.
        feats = split_feature_extraction.get_feature_dict()
        self.split_node_embed = nn.ModuleDict({
            "branch": FeedForwardNet(feats["branch"], geometry_embed_dim, 3),
            "proposal": FeedForwardNet(
                feats["proposal"], geometry_embed_dim // 2, 3
            ),
        })
        self.gat1 = _build_hetero_gat(geometry_embed_dim, split_heads)
        self.gat2 = _build_hetero_gat(geometry_embed_dim * split_heads, split_heads)
        self.split_head = nn.Linear(geometry_embed_dim * split_heads ** 2, 1)

    def forward(self, x, task):
        """
        Parameters
        ----------
        x : dict
            Batch dictionary.  For "merge": must contain "img" and
            "tree_sample".  For "split": must contain "img", "x_dict", and
            "edge_index_dict".
        task : str
            "merge" or "split".

        Returns
        -------
        torch.Tensor
            Logits with shape (N, 1).
        """
        if task == "merge":
            return self._forward_merge(x)
        return self._forward_split(x)

    def _forward_merge(self, x):
        z_img = self.vision(x["img"])
        z_tree = self._encode_arborist(x["tree_sample"], z_img.device)
        return self.merge_head(self.drop(torch.cat([z_img, z_tree], dim=1)))

    def _forward_split(self, x):
        x_dict = x["x_dict"]
        edge_index_dict = x["edge_index_dict"]

        # Encode patches through the shared backbone.  Patches may arrive as
        # float16; cast when not inside autocast so convolutions stay in the
        # parameter dtype.
        x_img = x["img"]
        if not torch.is_autocast_enabled():
            x_img = x_img.to(next(self.vision.parameters()).dtype)
        z_img = self.vision(x_img)

        # Embed geometry features and fuse image embeddings into proposals.
        for key, f in self.split_node_embed.items():
            x_dict[key] = f(x_dict[key])
        x_dict["proposal"] = torch.cat((x_dict["proposal"], z_img), dim=1)

        # Heterogeneous message passing.
        x_dict = self.gat1(x_dict, edge_index_dict)
        x_dict = self.gat2(x_dict, edge_index_dict)
        return self.split_head(x_dict["proposal"])

    @torch._dynamo.disable
    def _encode_arborist(self, tree_samples, device):
        with torch.amp.autocast("cuda", enabled=False):
            z_trees = [self.arborist.encode(s)[0] for s in tree_samples]
        return torch.stack(z_trees).to(device)
