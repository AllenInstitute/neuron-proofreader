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


class INRDecoder(nn.Module):
    """
    Implicit neural representation decoder. Models the curve as F(z, t) ->
    offsets where t in [0, 1] is the normalized arc position of each segment.
    Supports three MLP variants: plain, residual (ResNet-style), and SIREN.
    """

    def __init__(
        self,
        segment_len=10,
        latent_dim=32,
        d_hidden=256,
        n_layers=4,
        n_frequencies=16,
        mlp_type="residual",
        dropout=0.1,
    ):
        """
        Parameters
        ----------
        segment_len : int
            Number of points per segment.
        latent_dim : int
            Dimension of the input latent vector.
        d_hidden : int
            Hidden dimension of the MLP.
        n_layers : int
            Number of hidden layers.
        n_frequencies : int
            Number of Fourier frequency bands for positional encoding of t.
            Not used when mlp_type is 'siren' (SIREN handles PE internally).
        mlp_type : str
            One of 'plain', 'residual', or 'siren'.
        dropout : float
            Dropout probability. Not used in SIREN.
        """
        super().__init__()
        self.segment_len = segment_len
        self.n_frequencies = n_frequencies
        self.mlp_type = mlp_type

        d_out = segment_len * 3

        if mlp_type == "siren":
            # SIREN handles position encoding internally via sin activations —
            # just concatenate raw t (scalar) with z
            d_input = latent_dim + 1
            layers = [SirenLayer(d_input, d_hidden, is_first=True)]
            for _ in range(n_layers - 1):
                layers.append(SirenLayer(d_hidden, d_hidden))
            layers.append(nn.Linear(d_hidden, d_out))
            self.mlp = nn.Sequential(*layers)

        elif mlp_type == "residual":
            d_input = latent_dim + 2 * n_frequencies
            self.input_proj = nn.Sequential(
                nn.Linear(d_input, d_hidden),
                nn.LayerNorm(d_hidden),
                nn.GELU(),
            )
            self.blocks = nn.ModuleList(
                [ResidualBlock(d_hidden, dropout) for _ in range(n_layers)]
            )
            self.output_proj = nn.Linear(d_hidden, d_out)

        elif mlp_type == "plain":
            d_input = latent_dim + 2 * n_frequencies
            layers = [
                nn.Linear(d_input, d_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            for _ in range(n_layers - 1):
                layers += [
                    nn.Linear(d_hidden, d_hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            layers.append(nn.Linear(d_hidden, d_out))
            self.mlp = nn.Sequential(*layers)

        else:
            raise ValueError(f"Unknown mlp_type: {mlp_type}")

    def positional_encoding(self, t):
        """
        Fourier positional encoding for scalar arc positions.

        Parameters
        ----------
        t : torch.Tensor
            Arc positions of shape (n_segments,) in [0, 1].

        Returns
        -------
        torch.Tensor
            Shape (n_segments, 2 * n_frequencies).
        """
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
            Number of output segments.

        Returns
        -------
        torch.Tensor
            Reconstructed offset sequence of shape (B, n_segments * segment_len, 3).
        """
        B = z.shape[0]
        n_segments = n_segments or 10
        t = torch.linspace(0, 1, n_segments, device=z.device)

        if self.mlp_type == "siren":
            # Raw scalar t concatenated with z — SIREN encodes frequency internally
            t_exp = t.unsqueeze(-1).unsqueeze(0).expand(B, -1, -1)
            z_exp = z.unsqueeze(1).expand(-1, n_segments, -1)
            x = torch.cat([z_exp, t_exp], dim=-1)
            out = self.mlp(x)

        elif self.mlp_type == "residual":
            pe = self.positional_encoding(t)
            z_exp = z.unsqueeze(1).expand(-1, n_segments, -1)
            pe_exp = pe.unsqueeze(0).expand(B, -1, -1)
            x = torch.cat([z_exp, pe_exp], dim=-1)
            x = self.input_proj(x)
            for block in self.blocks:
                x = block(x)
            out = self.output_proj(x)

        else:  # plain
            pe = self.positional_encoding(t)
            z_exp = z.unsqueeze(1).expand(-1, n_segments, -1)
            pe_exp = pe.unsqueeze(0).expand(B, -1, -1)
            x = torch.cat([z_exp, pe_exp], dim=-1)
            out = self.mlp(x)

        return out.reshape(B, n_segments * self.segment_len, 3)


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
        super().__init__()
        self.segment_len = segment_len
        self.n_frequencies = n_frequencies
        d_seg = segment_len * 3

        # Seed the GRU hidden state from z
        self.latent_proj = nn.Linear(latent_dim, d_hidden)

        # Input at each step: [T_{i-1} | pe(t_i) | z]
        # Including z directly at every step (alongside the hidden state) lets
        # the model re-attend to the global code at each position, rather than
        # relying purely on it surviving through the hidden state
        d_input = d_seg + 2 * n_frequencies + latent_dim
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
        Fourier positional encoding for scalar arc positions.

        Parameters
        ----------
        t : torch.Tensor
            Arc positions of shape (n_segments,) in [0, 1].

        Returns
        -------
        torch.Tensor
            Shape (n_segments, 2 * n_frequencies).
        """
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
            Reconstructed offset sequence of shape (B, n_segments * segment_len, 3).
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
        T_prev = torch.zeros(B, d_seg, device=z.device)  # T_{-1}: start token
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
        decoder,
        decoder_name=None,
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
            "decoder_name": decoder_name,
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
        self.decoder = decoder

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
        if mask is not None:
            valid_lengths = (~mask).sum(dim=1)  # (B,)
            n_segments = (
                valid_lengths.min() // self.decoder.segment_len
            ).item()
        else:
            n_segments = diffs.shape[1] // self.decoder.segment_len
        reconstruction = self.decoder(z, n_segments=n_segments)
        return reconstruction, z

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
class ResidualBlock(nn.Module):
    def __init__(self, d_hidden, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
            nn.LayerNorm(d_hidden),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))


class SirenLayer(nn.Module):
    def __init__(self, d_in, d_out, omega=30.0, is_first=False):
        """
        Parameters
        ----------
        omega : float
            Frequency scaling factor. Default 30 following the original paper.
        is_first : bool
            First layer uses a different initialization scheme.
        """
        super().__init__()
        self.omega = omega
        self.linear = nn.Linear(d_in, d_out)
        self._init_weights(is_first, d_in)

    def _init_weights(self, is_first, d_in):
        with torch.no_grad():
            if is_first:
                bound = 1 / d_in
            else:
                bound = np.sqrt(6 / d_in) / self.omega
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega * self.linear(x))


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
