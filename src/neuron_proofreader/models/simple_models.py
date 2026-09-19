"""
Created on Fri July 10 12:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Graph neural network architectures that classify edge proposals.

"""

from torch import nn


class FeedForwardNet(nn.Module):
    """
    A feed-forward neural network for encoding feature vectors into a latent
    representation.

    Architecture
    ------------
        Linear(input_dim -> hidden_dim)
            ↓
        MLP Block × n_blocks
            ↓
        LayerNorm
            ↓
        Linear(hidden_dim -> output_dim)
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        n_blocks=2,
        hidden_dim=128,
        expansion=4,
        dropout=0.1,
    ):
        """
        Instantiates a FeedForwardNet object.

        Parameters
        ----------
        input_dim : int
            Dimension of the input feature vector.
        output_dim : int
            Dimension of the output feature vector.
        n_blocks : int, optional
            Number of residual MLP blocks. Default is 2.
        hidden_dim : int, optional
            Dimension of the hidden representation. Default is 128.
        expansion : int, optional
            Expansion factor used within each MLP block. Default is 4.
        dropout : float, optional
            Dropout probability applied within each MLP block. Default is 0.1.
        """
        # Call parent class
        super().__init__()

        # Create encoder layers
        layers = [nn.Linear(input_dim, hidden_dim)]
        for _ in range(n_blocks):
            layers.append(
                MLPBlock(
                    hidden_dim,
                    expansion=expansion,
                    dropout=dropout,
                )
            )

        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.Linear(hidden_dim, output_dim))

        # Instance attributes
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        Encodes the input feature vectors.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, input_dim), where N is the batch size.
        Returns
        -------
        torch.Tensor
            Encoded feature tensor of shape (N, output_dim).
        """
        return self.net(x)


class MLPBlock(nn.Module):
    """
    A residual feed-forward block consisting of a Layer Normalization layer
    followed by a two-layer multilayer perceptron (MLP).

    Architecture
    ------------
        LayerNorm
            ↓
        Linear(dim -> expansion * dim)
            ↓
        GELU
            ↓
        Dropout
            ↓
        Linear(expansion * dim -> dim)
            ↓
        Dropout
            ↓
        Residual Addition
    """

    def __init__(self, dim, expansion=4, dropout=0.1):
        """
        Instantiates an MLPBlock object.

        Parameters
        ----------
        dim : int
            Dimension of the input and output feature vectors.
        expansion : int, optional
            Expansion factor for the hidden layer of the MLP. Default is 4.
        dropout : float, optional
            Dropout probability applied after each linear layer. Default is
            0.1.
        """
        # Call parent class
        super().__init__()

        # Instance attributes
        hidden_dim = dim * expansion
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """
        Encodes the input feature vectors.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, input_dim), where N is the batch size.

        Returns
        -------
        torch.Tensor
            Encoded feature tensor of shape (N, output_dim).
        """
        return x + self.net(self.norm(x))
