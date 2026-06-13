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
    Transformer encoder that maps a 3D space curve, normalized to the unit
    sphere, to a fixed-size latent vector. The curve is tokenized into fixed-
    length segments with a fixed learned token for the start point at the
    origin and a projected token for the end point. Positional encodings are
    sinusoidal over the normalized arc position [0, 1], making the encoder
    robust to varying numbers of points and path lengths.
    """

    def __init__(
        self,
        segment_len=10,
        d_token=64,
        n_heads=4,
        n_layers=4,
        d_ff=64,
        latent_dim=32,
        dropout=0.1,
    ):
        """
        Parameters
        ----------
        segment_len : int
            Number of points per segment token.
        d_token : int
            Dimension of each token.
        n_heads : int
            Number of attention heads.
        n_layers : int
            Number of transformer encoder layers.
        d_ff : int
            Feed-forward hidden dimension.
        latent_dim : int
            Dimension of the output latent vector.
        dropout : float
            Dropout probability.
        """
        # Call parent class
        super().__init__()

        # Instance attributes
        self.segment_len = segment_len
        self.start_token = nn.Parameter(torch.randn(1, 1, d_token))
        self.end_token_proj = nn.Linear(3, d_token)
        self.segment_proj = nn.Linear(segment_len * 3, d_token)

        # Archictecture
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

    def forward(self, offsets, mask=None):
        """
        Parameters
        ----------
        offsets : torch.Tensor
            Shape (B, N, 3), normalized to the unit sphere, with
            offsets[:, 0] == [0, 0, 0]. N can vary across calls.
        mask : torch.Tensor, optional
            Shape (B, N), True where padding (to be ignored). Default is None.

        Returns
        -------
        z : torch.Tensor
            Latent vector of shape (B, latent_dim).
        tokens : torch.Tensor
            Per-token encodings of shape (B, n_segments + 2, d_token).
        """
        B, N, _ = offsets.shape
        n_segments = N // self.segment_len

        # Start and end tokens
        start_tok = self.start_token.expand(B, -1, -1)  # (B, 1, d_token)
        end_tok = self.end_token_proj(offsets[:, -1, :]).unsqueeze(
            1
        )  # (B, 1, d_token)

        # Segment tokens
        segments = offsets[:, : n_segments * self.segment_len, :]
        segments = segments.reshape(B, n_segments, self.segment_len * 3)
        seg_tokens = self.segment_proj(segments)  # (B, n_seg, d_token)

        # Concatenate: [start | segments | end]
        tokens = torch.cat(
            [start_tok, seg_tokens, end_tok], dim=1
        )  # (B, n_seg+2, d_token)

        # Convert point-level mask to token-level mask
        token_mask = None
        if mask is not None:
            seg_mask = mask[:, :: self.segment_len][
                :, :n_segments
            ]  # (B, n_seg)
            token_mask = torch.cat(
                [
                    torch.zeros(B, 1, dtype=torch.bool, device=mask.device),
                    seg_mask,
                    torch.zeros(B, 1, dtype=torch.bool, device=mask.device),
                ],
                dim=1,
            )  # (B, n_seg+2)

        # Sinusoidal positional encoding, zeroed out for padding tokens
        pe = sinusoidal_encoding(
            tokens.shape[1], tokens.shape[2], tokens.device
        )
        if token_mask is not None:
            pe = pe * (~token_mask).unsqueeze(-1).float()
        tokens = tokens + pe

        tokens = self.transformer(tokens, src_key_padding_mask=token_mask)

        # Mean pool over non-padding tokens only
        if token_mask is not None:
            valid = (~token_mask).unsqueeze(-1).float()
            z = self.to_latent((tokens * valid).sum(dim=1) / valid.sum(dim=1))
        else:
            z = self.to_latent(tokens.mean(dim=1))

        return z, tokens


class CurveDecoder(nn.Module):
    """
    Transformer decoder that reconstructs a 3D space curve from a latent
    vector and the encoder's token representations. Positional queries are
    sinusoidally encoded over arc position and biased by the global latent.
    The output resolution can differ from the encoder input, allowing
    decoding at arbitrary granularity.
    """

    def __init__(
        self,
        n_points=100,
        segment_len=10,
        d_token=64,
        n_heads=4,
        n_layers=4,
        d_ff=64,
        latent_dim=32,
        dropout=0.1,
    ):
        """
        Parameters
        ----------
        n_points : int
            Default number of output curve points.
        segment_len : int
            Number of points per segment token (must match encoder).
        d_token : int
            Dimension of each token throughout the transformer.
        n_heads : int
            Number of attention heads.
        n_layers : int
            Number of transformer decoder layers.
        d_ff : int
            Feed-forward hidden dimension.
        latent_dim : int
            Dimension of the input latent vector.
        dropout : float
            Dropout probability.
        """
        super().__init__()
        self.segment_len = segment_len
        self.n_segments = n_points // segment_len

        # Project latent to d_token to bias the positional queries
        self.latent_proj = nn.Linear(latent_dim, d_token)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(
            decoder_layer, num_layers=n_layers
        )

        self.to_points = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Linear(d_token, segment_len * 3),
        )

    def forward(self, z, encoder_tokens, encoder_mask=None, n_segments=None):
        """
        Parameters
        ----------
        z : torch.Tensor
            Latent vector of shape (B, latent_dim).
        encoder_tokens : torch.Tensor
            Per-token encoder outputs of shape (B, n_segments + 2, d_token).
        encoder_mask : torch.Tensor, optional
            Shape (B, n_segments + 2), True where padding. Passed as
            memory_key_padding_mask to cross-attention. Default is None.
        n_segments : int, optional
            Number of output segments. Inferred from encoder tokens if not
            provided.

        Returns
        -------
        curve : torch.Tensor
            Reconstructed curve of shape (B, n_segments * segment_len, 3).
        """
        B = z.shape[0]
        d_token = encoder_tokens.shape[2]
        n_segments = n_segments or self.n_segments

        # Sinusoidal queries over arc position, biased by global latent
        pe = sinusoidal_encoding(n_segments, d_token, encoder_tokens.device)
        latent = self.latent_proj(z).unsqueeze(1)  # (B, 1, d_token)
        queries = pe.expand(B, -1, -1) + latent  # (B, n_seg, d_token)

        out = self.transformer(
            queries,
            encoder_tokens,
            memory_key_padding_mask=encoder_mask,
        )  # (B, n_seg, d_token)

        segments = self.to_points(out)  # (B, n_seg, seg_len*3)
        offsets = segments.reshape(B, n_segments * self.segment_len, 3)
        return offsets


class CurveAutoencoder(nn.Module):

    def __init__(
        self,
        n_points=100,
        segment_len=10,
        d_token=64,
        n_heads=4,
        n_layers=4,
        d_ff=64,
        latent_dim=32,
        dropout=0.1,
    ):
        # Call parent class
        super().__init__()

        # Config
        self.config = {
            "n_points": n_points,
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
        self.decoder = CurveDecoder(
            n_points=n_points,
            segment_len=segment_len,
            d_token=d_token,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            latent_dim=latent_dim,
            dropout=dropout,
        )

    def forward(self, offsets, token_mask):
        """
        Parameters
        ----------
        offsets : torch.Tensor
            Shape (B, N, 3), normalized to the unit sphere, offsets[:, 0] == 0.

        Returns
        -------
        reconstruction : torch.Tensor
            Shape (B, N, 3).
        z : torch.Tensor
            Latent vector of shape (B, latent_dim).
        """
        z, encoder_tokens = self.encoder(offsets, token_mask)
        n_segments = offsets.shape[1] // self.decoder.segment_len
        reconstruction = self.decoder(z, encoder_tokens, n_segments=n_segments)
        return reconstruction, z

    def encode(self, offsets):
        z, _ = self.encoder(offsets)
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


# --- Helpers ---
def sinusoidal_encoding(n_tokens, d_token, device):
    """
    Sinusoidal positional encoding over normalised arc position [0, 1].

    Parameters
    ----------
    n_tokens : int
        Number of tokens (segments + 2 endpoint tokens).
    d_token : int
        Model dimension.
    device : torch.device
        Device to create the encoding on.

    Returns
    -------
    torch.Tensor
        Encoding of shape (1, n_tokens, d_token).
    """
    position = torch.linspace(0, 1, n_tokens, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_token, 2, device=device)
        * (-np.log(10000.0) / d_token)
    )
    pe = torch.zeros(n_tokens, d_token, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)
