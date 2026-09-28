"""
Created on Fri Sep 25 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Split-correction GNN models using the updated CNN3D vision backbone.

"""

from torch_geometric import nn as nn_geometric

import torch
import torch.nn as nn

from arborist.models.arborist import Arborist
from neuron_proofreader.models.new_vision_models import CNN3D
from neuron_proofreader.split_proofreading import split_feature_extraction
from neuron_proofreader.utils.ml_util import FeedForwardNet


def proposal_length_feature(tree_samples, device, dtype=torch.float32):
    """
    Log-scaled proposal length, one scalar per sample. Length is a property
    of the candidate edge that neither the rooted-subgraph curves nor the
    resized image patch expose, so it is fed to models explicitly.

    Parameters
    ----------
    tree_samples : List[TreeSample]
        Samples produced by ProposalTreeFeatureExtractor.

    Returns
    -------
    torch.Tensor
        Shape (n_samples, 1), values log1p(length_in_microns).
    """
    lengths = torch.tensor(
        [s.proposal_length for s in tree_samples], device=device, dtype=dtype
    )
    return torch.log1p(lengths).unsqueeze(1)


class ProposalTreeEncoder(nn.Module):
    """
    Encodes proposal-rooted TreeSamples into fixed-size embeddings using
    Arborist. The two curves incident to the root (one per side of the
    proposal) are mean-pooled after the GraphTransformer has contextualized
    them against the rest of the subgraph.

    Parameters
    ----------
    latent_dim : int, optional
        Arborist latent dimension and output embedding size. Default is 64.
    pretrained_curve_encoder_path : str or None, optional
        Path to a CurveAutoencoder checkpoint whose encoder weights are loaded
        into the CurveEncoder. Default is None.
    freeze_curve_encoder : bool, optional
        If True and a pretrained path is provided, the CurveEncoder is frozen.
        Default is True.
    **arborist_kwargs
        Forwarded to the Arborist constructor.
    """

    def __init__(
        self,
        latent_dim=64,
        pretrained_curve_encoder_path=None,
        freeze_curve_encoder=True,
        **arborist_kwargs,
    ):
        super().__init__()
        self.config = {"latent_dim": latent_dim, **arborist_kwargs}
        self.arborist = Arborist(latent_dim=latent_dim, **arborist_kwargs)
        if pretrained_curve_encoder_path is not None:
            self._load_curve_encoder(pretrained_curve_encoder_path)
            if freeze_curve_encoder:
                for p in self.arborist.curve_encoder.parameters():
                    p.requires_grad_(False)

    def _load_curve_encoder(self, path):
        ckpt = torch.load(path, weights_only=False)
        state = ckpt.get("model_state", ckpt)
        encoder_state = {
            k[len("encoder."):]: v
            for k, v in state.items()
            if k.startswith("encoder.")
        }
        self.arborist.curve_encoder.load_state_dict(encoder_state)

    @torch._dynamo.disable
    def forward(self, tree_samples):
        """
        Parameters
        ----------
        tree_samples : List[TreeSample]
            One sample per proposal, in proposal-index order.

        Returns
        -------
        torch.Tensor
            Shape (n_proposals, latent_dim).
        """
        device = next(self.arborist.parameters()).device
        latent_dim = self.arborist.config["latent_dim"]

        # Collect unique curves across all proposals. CurveEncoder encodes
        # each curve independently, so shared curves (same skeleton path in
        # multiple proposals' subgraphs) only need to be encoded once.
        curve_to_idx = {}
        unique_curves = []
        sample_curve_indices = []
        for sample in tree_samples:
            local_indices = []
            for curve in sample.curves:
                key = (curve.shape, curve.tobytes())
                if key not in curve_to_idx:
                    curve_to_idx[key] = len(unique_curves)
                    unique_curves.append(curve)
                local_indices.append(curve_to_idx[key])
            sample_curve_indices.append(local_indices)

        if not unique_curves:
            return torch.zeros(len(tree_samples), latent_dim, device=device)

        # Single CurveEncoder pass over all unique curves.
        min_len = self.arborist.curve_encoder.segment_len
        lengths = [len(c) for c in unique_curves]
        t_max = max(max(lengths), min_len)
        diffs = torch.zeros(len(unique_curves), t_max, 3, device=device)
        mask = torch.ones(len(unique_curves), t_max, dtype=torch.bool, device=device)
        for i, (c, length) in enumerate(zip(unique_curves, lengths)):
            diffs[i, :length] = torch.as_tensor(c, dtype=torch.float32, device=device)
            mask[i, :length] = False

        with torch.amp.autocast("cuda", enabled=False):
            z_all, _ = self.arborist.curve_encoder(diffs, mask)

        # Per-sample GraphTransformer pass. GraphTransformer applies cross-curve
        # attention using each proposal's unique tree topology, so it must run
        # per sample. Curve embeddings are looked up from the shared z_all.
        zs = []
        for sample, local_indices in zip(tree_samples, sample_curve_indices):
            if not sample.curves:
                zs.append(torch.zeros(latent_dim, device=device))
                continue

            z = z_all[torch.tensor(local_indices, dtype=torch.long, device=device)]
            edge_index = torch.as_tensor(
                sample.edge_index, dtype=torch.long, device=device
            )
            with torch.amp.autocast("cuda", enabled=False):
                z_curves = self.arborist.graph_transformer(z, edge_index)

            root_idx = sample.root_curve_indices or range(len(z_curves))
            zs.append(z_curves[list(root_idx)].mean(dim=0))

        return torch.stack(zs)


class NewVisionHGAT(nn.Module):
    """
    Heterogeneous graph attention network for split correction.

    Uses the updated CNN3D backbone from new_vision_models for image patch
    encoding. Branch and proposal nodes are embedded from geometry features;
    proposal nodes are then augmented with per-patch image embeddings before
    two rounds of heterogeneous message passing classify each proposal.
    """

    _RELATIONS = [
        ("branch", "to", "branch"),
        ("proposal", "to", "proposal"),
        ("branch", "to", "proposal"),
        ("proposal", "to", "branch"),
    ]

    def __init__(
        self,
        patch_shape,
        gnn_hidden_dim=64,
        img_embed_dim=64,
        heads=4,
    ):
        """
        Instantiates a NewVisionHGAT object.

        Parameters
        ----------
        patch_shape : Tuple[int]
            Spatial shape of image patches (D, H, W) — no channel dimension.
        gnn_hidden_dim : int, optional
            Hidden dimension for branch and proposal geometry embeddings.
            Default is 64.
        img_embed_dim : int, optional
            Output dimension of the image patch encoder. Default is 64.
        heads : int, optional
            Number of attention heads per GAT layer. Default is 4.
        """
        super().__init__()

        feats = split_feature_extraction.get_feature_dict()

        # Node feature embeddings: branch→gnn_hidden_dim,
        # proposal→gnn_hidden_dim//2 (img features fill the other half).
        self.node_embedding = nn.ModuleDict({
            "branch": FeedForwardNet(feats["branch"], gnn_hidden_dim, 3),
            "proposal": FeedForwardNet(
                feats["proposal"], gnn_hidden_dim // 2, 3
            ),
        })

        # Image patch embedding: 2-channel input (raw image + segmentation mask).
        self.patch_embedding = CNN3D(
            (2,) + tuple(patch_shape),
            output_dim=img_embed_dim,
            use_output_head=True,
        )

        # Heterogeneous GAT layers.
        # After node embedding + image fusion, proposal nodes have dim
        # gnn_hidden_dim//2 + img_embed_dim; branch nodes have dim
        # gnn_hidden_dim.  GATv2Conv handles mismatched dims via lazy
        # (-1) input initialisation.
        self.gat1 = _build_hetero_gat(gnn_hidden_dim, heads)
        self.gat2 = _build_hetero_gat(gnn_hidden_dim * heads, heads)
        self.output = nn.Linear(gnn_hidden_dim * heads ** 2, 1)

        self.config = {
            "patch_shape": tuple(patch_shape),
            "gnn_hidden_dim": gnn_hidden_dim,
            "img_embed_dim": img_embed_dim,
            "heads": heads,
        }

        self._init_weights()

    def _init_weights(self):
        for module in [self.node_embedding, self.output]:
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.kaiming_normal_(p)
                else:
                    nn.init.zeros_(p)

    def forward(self, input_dict):
        """
        Parameters
        ----------
        input_dict : dict
            Must contain:
              "img"            – (N_proposals, 2, D, H, W) image patches
              "x_dict"         – dict mapping node type → feature tensor
              "edge_index_dict" – dict mapping edge type → (2, E) index tensor

        Returns
        -------
        torch.Tensor
            Per-proposal logits with shape (N_proposals, 1).
        """
        x_dict = input_dict["x_dict"]
        x_img = input_dict["img"]
        edge_index_dict = input_dict["edge_index_dict"]

        # Patches may arrive as float16 from the dataloader; convolutions need
        # them in the parameter dtype when not running inside autocast.
        if not torch.is_autocast_enabled():
            x_img = x_img.to(next(self.patch_embedding.parameters()).dtype)
        x_img = self.patch_embedding(x_img)

        # Embed node geometry features, then fuse image embeddings into proposals.
        for key, f in self.node_embedding.items():
            x_dict[key] = f(x_dict[key])
        x_dict["proposal"] = torch.cat((x_dict["proposal"], x_img), dim=1)

        # Two rounds of heterogeneous message passing.
        x_dict = self.gat1(x_dict, edge_index_dict)
        x_dict = self.gat2(x_dict, edge_index_dict)
        return self.output(x_dict["proposal"])

    def save(self, path):
        torch.save({"config": self.config, "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path, map_location=None):
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
        model = cls(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        return model


def _build_hetero_gat(out_dim, heads):
    """
    Builds a HeteroConv with GATv2Conv for every relation in NewVisionHGAT.
    Same-type relations use a single input-dim spec; cross-type relations use
    a pair so GATv2Conv can handle bipartite input.
    """
    gat_dict = {}
    for rel in NewVisionHGAT._RELATIONS:
        src, _, dst = rel
        if src == dst:
            conv = nn_geometric.GATv2Conv(-1, out_dim, dropout=0.1, heads=heads)
        else:
            conv = nn_geometric.GATv2Conv(
                (-1, -1),
                out_dim,
                add_self_loops=False,
                dropout=0.1,
                heads=heads,
            )
        gat_dict[rel] = conv
    return nn_geometric.HeteroConv(gat_dict)
