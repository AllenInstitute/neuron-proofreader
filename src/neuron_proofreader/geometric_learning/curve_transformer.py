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


class NewCurveEncoder(nn.Module):

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
        super().__init__()

        self.segment_len = segment_len
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))
        self.segment_proj = nn.Linear(segment_len * 3, d_token)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.to_latent = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Linear(d_token, latent_dim),
        )

    def forward(self, offsets, mask=None):
        B, N, _ = offsets.shape
        n_segments = N // self.segment_len

        # CLS and segment tokens
        cls_token = self.cls_token.expand(B, -1, -1)
        segments = offsets[:, :n_segments * self.segment_len, :]
        seg_tokens = self.segment_proj(
            segments.reshape(B, n_segments, self.segment_len * 3)
        )

        # Concatenate: [CLS | segments | end]
        tokens = torch.cat([cls_token, seg_tokens], dim=1)

        # Token-level mask — CLS is always unmasked
        token_mask = None
        if mask is not None:
            seg_mask = mask[:, ::self.segment_len][:, :n_segments]
            token_mask = torch.cat([
                torch.zeros(B, 1, dtype=torch.bool, device=mask.device),
                seg_mask,
            ], dim=1)

        # Sinusoidal positional encoding — skip CLS (position 0 reserved)
        pe = sinusoidal_encoding(tokens.shape[1], tokens.shape[2], tokens.device)
        pe[:, 0, :] = 0.0
        if token_mask is not None:
            pe = pe * (~token_mask).unsqueeze(-1).float()
        tokens = tokens + pe

        tokens = self.transformer(tokens, src_key_padding_mask=token_mask)

        # Aggregate via CLS token only
        z = self.to_latent(tokens[:, 0, :])
        return z, tokens


class CurveEncoder(nn.Module):
    """
    Transformer encoder that maps a 3D first order finite differences, to a
    fixed-size latent vector. The curve is tokenized into fixed-length
    segments with a class token. Positional encodings are sinusoidal over the
    normalized arc position [0, 1], making the encoder robust to varying
    numbers of points and path lengths.
    """

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
        """
        Parameters
        ----------
        segment_len : int
            Number of points per segment token.
        d_token : int
            Token dimension.
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
            Per-token encodings of shape (B, n_segments + 1, d_token).
        """
        B, N, _ = offsets.shape
        n_segments = N // self.segment_len

        # Start and end tokens
        start_tok = self.start_token.expand(B, -1, -1)  # (B, 1, d_token)
        end_token = self.end_token_proj(offsets[:, -1, :]).unsqueeze(
            1
        )  # (B, 1, d_token)

        # Segment tokens
        segments = offsets[:, : n_segments * self.segment_len, :]
        segments = segments.reshape(B, n_segments, self.segment_len * 3)
        seg_tokens = self.segment_proj(segments)  # (B, n_seg, d_token)

        # Concatenate: [start | segments | end]
        tokens = torch.cat(
            [start_tok, seg_tokens, end_token], dim=1
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
            )

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


class AutoregressiveINRDecoder(nn.Module):
    """
    Autoregressive implicit neural representation decoder. At each step,
    predicts the i-th segment's offsets conditioned on the global latent z,
    the arc position t_i, and the previous segment's predicted offsets T_{i-1}.

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
        use_fourier=False
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
        self.segment_len = segment_len
        self.n_frequencies = n_frequencies
        self.use_fourier = use_fourier
        d_seg = segment_len * 3

        # Seed the GRU hidden state from z
        self.latent_proj = nn.Linear(latent_dim, d_hidden)

        # Input at each step: [T_{i-1} | pe(t_i) | z]
        d_pe = 1 if not use_fourier else 2 * n_frequencies
        d_input = d_seg + d_pe + latent_dim
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

    def positional_encoding(self, t):
        """
        Encodes scalar arc positions as either raw arc length or Fourier features.
    
        Parameters
        ----------
        t : torch.Tensor
            Arc positions of shape (n_segments,) in [0, 1].
    
        Returns
        -------
        torch.Tensor
            Shape (n_segments, 1) if use_fourier=False,
            else (n_segments, 2 * n_frequencies).
        """
        if not self.use_fourier:
            return t.unsqueeze(-1)

        freqs = 2 ** torch.arange(self.n_frequencies, device=t.device).float()
        x = t.unsqueeze(-1) * freqs * torch.pi
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)

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
            Reconstructed differences sequence.
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
        pe = self.positional_encoding(t)

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
        self.encoder = NewCurveEncoder(
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


# --- Helpers ---
def sinusoidal_encoding(n_tokens, d_token, device):
    """
    Sinusoidal positional encoding over normalised arc position [0, 1].

    Parameters
    ----------
    n_tokens : int
        Number of tokens.
    d_token : int
        Token dimension.
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
