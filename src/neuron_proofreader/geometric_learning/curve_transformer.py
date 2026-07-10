"""
Created on Wed June 10 12:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

...

"""

import numpy as np
import torch
import torch.nn as nn

from neuron_proofreader.utils import util


class CurveEncoder(nn.Module):
    """
    Transformer encoder that maps 3D first-order finite differences to a
    latent vector. The curve is tokenized into fixed-length segments with a
    class token. Each segment token is built from its flattened finite
    differences concatenated with its normalized arc-length position in
    [0, 1], letting the segment projection learn the geometry-position
    relationship directly rather than injecting a separate positional
    encoding. This makes the encoder robust to varying numbers of points
    and path lengths.
    """

    def __init__(
        self,
        segment_len=10,
        d_token=128,
        n_heads=4,
        n_layers=4,
        d_ff=256,
        latent_dim=64,
        dropout=0.1,
    ):
        """
        Instantiates a CurveEncoder object.

        Parameters
        ----------
        segment_len : int, optional
            Number of points per segment token. Default is 10.
        d_token : int, optional
            Token dimension. Default is 128.
        n_heads : int, optional
            Number of attention heads. Default is 4.
        n_layers : int, optional
            Number of transformer encoder layers. Default is 4.
        d_ff : int, optional
            Feed-forward hidden dimension. Default is 256.
        latent_dim : int, optional
            Dimension of the output latent vector. Default is 64.
        dropout : float, optional
            Dropout probability. Default is 0.1.
        """
        # Call parent class
        super().__init__()

        # Instance attributes
        self.segment_len = segment_len
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))
        self.segment_proj = nn.Linear(segment_len * 3 + 1, d_token)

        # Architecture
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )
        self.to_latent = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Linear(d_token, latent_dim),
        )

    def forward(self, diffs, mask=None):
        """
        Parameters
        ----------
        diffs : torch.Tensor
            Shape (B, N, 3), normalized to the unit sphere, with
            diffs[:, 0] == [0, 0, 0]. N can vary across calls.
        mask : torch.Tensor, optional
            True where padding (to be ignored). Default is None.

        Returns
        -------
        z : torch.Tensor
            Latent vector of shape (B, latent_dim).
        tokens : torch.Tensor
            Per-token encodings of shape (B, n_segments + 1, d_token).
        """
        B, N, _ = diffs.shape
        n_segments = N // self.segment_len

        # Arc length: cumulative step norm, normalized to [0, 1] per sequence.
        step_len = diffs.norm(dim=-1)
        arc_pos = torch.cumsum(step_len, dim=1)
        arc_pos = arc_pos / arc_pos[:, -1:].clamp_min(1e-8)

        # Segment features: flattened diffs + arc position
        segments = diffs[:, :n_segments * self.segment_len, :]
        segments = segments.reshape(B, n_segments, self.segment_len, 3)
        seg_pos = arc_pos[:, :n_segments * self.segment_len]
        seg_pos = seg_pos.reshape(B, n_segments, self.segment_len)[:, :, -1:]

        seg_input = torch.cat(
            [segments.reshape(B, n_segments, -1), seg_pos], dim=-1
        )
        seg_tokens = self.segment_proj(seg_input)

        # Concatenate: [CLS | segments]
        cls_token = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_token, seg_tokens], dim=1)

        # Token-level mask — CLS is always unmasked
        token_mask = None
        if mask is not None:
            seg_mask = mask[:, ::self.segment_len][:, :n_segments]
            token_mask = torch.cat([
                torch.zeros(B, 1, dtype=torch.bool, device=mask.device),
                seg_mask,
            ], dim=1)

        tokens = self.transformer(tokens, src_key_padding_mask=token_mask)

        # Aggregate via CLS token only
        z = self.to_latent(tokens[:, 0, :])
        return z, tokens


class AutoregressiveINRDecoder(nn.Module):
    """
    Autoregressive implicit neural representation decoder. At each step,
    predicts the i-th segment's differences conditioned on the global latent
    z, arc position t_i, and previous segment's predicted diffs T_{i-1}.

        T_i = F(z, t_i | T_{i-1})

    The GRU hidden state carries long-range context from all previous
    segments, while T_{i-1} provides an explicit local continuity signal.
    """

    def __init__(
        self,
        segment_len=10,
        latent_dim=32,
        d_hidden=256,
        n_layers=4,
        n_frequencies=16,
        dropout=0.1,
    ):
        """
        Parameters
        ----------
        segment_len : int
            Number of points per segment.
        latent_dim : int
            Dimension of the input latent vector z.
        d_hidden : int
            GRU hidden dimension.
        n_layers : int
            Number of GRU layers.
        n_frequencies : int
            Number of Fourier frequency bands for positional encoding of t.
        dropout : float
            Dropout probability.
        """
        # Call parent class
        super().__init__()

        # Instance attributes
        self.latent_proj = nn.Linear(latent_dim, d_hidden)
        self.n_frequencies = n_frequencies
        self.segment_len = segment_len

        # Input at each step: [T_{i-1} | pe(t_i) | z]
        d_seg = segment_len * 3
        d_input = d_seg + 1 + latent_dim
        self.gru = nn.GRU(
            input_size=d_input,
            hidden_size=d_hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.output_proj = nn.Sequential(
            nn.LayerNorm(d_hidden),
            nn.Linear(d_hidden, d_seg),
        )

    def forward(self, z, n_segments=None):
        """
        Parameters
        ----------
        z : torch.Tensor
            Latent vector of shape (B, latent_dim).
        n_segments : int, optional
            Number of output segments to decode.

        Returns
        -------
        torch.Tensor
            Predicted finite differences.
        """
        B = z.shape[0]
        n_segments = n_segments or 10
        d_seg = self.segment_len * 3

        # Seed hidden state from z: (n_layers, B, d_hidden)
        h = (
            self.latent_proj(z)
            .unsqueeze(0)
            .expand(self.gru.num_layers, -1, -1)
            .contiguous()
        )

        # Arc positions and Fourier encodings: (n_segments, 2 * n_freq)
        t = torch.linspace(0, 1, n_segments, device=z.device)
        pe = t.unsqueeze(-1)

        # Autoregressive loop
        outputs = []
        T_prev = torch.zeros(B, d_seg, device=z.device)
        for s in range(n_segments):
            # Create GRU input
            pe_s = pe[s].unsqueeze(0).expand(B, -1)
            x_s = torch.cat([T_prev, pe_s, z], dim=-1).unsqueeze(1)

            # Predict next token and update hidden state
            out, h = self.gru(x_s, h)
            T_i = self.output_proj(out.squeeze(1))

            # Update previous token
            outputs.append(T_i)
            T_prev = T_i

        out = torch.stack(outputs, dim=1)
        return out.reshape(B, n_segments * self.segment_len, 3)


class CurveAutoencoder(nn.Module):

    def __init__(
        self,
        segment_len=10,
        d_token=128,
        n_heads=4,
        n_layers=4,
        d_ff=256,
        latent_dim=32,
        dropout=0.1,
    ):
        # Call parent class
        super().__init__()

        # Config
        self.config = {
            "segment_len": segment_len,
            "d_token": d_token,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "d_ff": d_ff,
            "latent_dim": latent_dim,
            "dropout": dropout,
        }

        # Architecture
        self.encoder = CurveEncoder(
            segment_len=segment_len,
            d_token=d_token,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            latent_dim=latent_dim,
            dropout=dropout,
        )
        self.decoder = AutoregressiveINRDecoder(
            segment_len=segment_len,
            latent_dim=latent_dim,
            d_hidden=256,
            n_layers=4,
            n_frequencies=16,
            dropout=dropout,
        )

    def forward(self, diffs, mask=None):
        """
        Parameters
        ----------
        diffs : torch.Tensor
            Shape (B, N, 3), normalized to the unit sphere, diffs[:, 0] == 0.

        Returns
        -------
        reconstruction : torch.Tensor
            Shape (B, N, 3).
        z : torch.Tensor
            Latent vector of shape (B, latent_dim).
        """
        z, encoder_tokens = self.encoder(diffs, mask)
        n_segments = diffs.shape[1] // self.decoder.segment_len
        recon = self.decoder(z, n_segments=n_segments)
        return recon, z

    def encode(self, diffs):
        z, _ = self.encoder(diffs)
        return z

    # --- Helpers ---
    def save_config(self, path):
        util.write_json(path, self.config)

    @classmethod
    def load(cls, path):
        checkpoint = torch.load(path)
        model = cls(**checkpoint["config"])
        model.load_state_dict(checkpoint["model_state_dict"])
        return model
