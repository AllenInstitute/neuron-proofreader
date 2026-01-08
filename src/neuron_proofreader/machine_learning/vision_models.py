"""
Created on Sat July 15 12:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code for vision models that perform image classification tasks within
NeuronProofreading pipelines.

"""

from einops import rearrange

import torch
import torch.nn as nn

from neuron_proofreader.utils import ml_util


# --- CNNs ---
class CNN3D(nn.Module):
    """
    Convolutional neural network for 3D images.
    """

    def __init__(
        self,
        patch_shape,
        output_dim=1,
        dropout=0.1,
        n_conv_layers=5,
        n_feat_channels=16,
        use_double_conv=True,
    ):
        """
        Instantiates a CNN3D object.

        Parameters
        ----------
        patch_shape : Tuple[int]
            Shape of input image patch.
        output_dim : int, optional
            Dimension of output. Default is 1.
        dropout : float, optional
            Fraction of values to randomly drop during training. Default is
            0.1.
        n_conv_layers : int, optional
            Number of convolutional layers. Default is 5.
        use_double_conv : bool, optional
            Indication of whether to use double convolution. Default is True.
        """
        # Call parent class
        nn.Module.__init__(self)

        # Class attributes
        self.dropout = dropout
        self.patch_shape = patch_shape

        # Convolutional layers
        self.conv_layers = init_cnn3d(
            2, n_feat_channels, n_conv_layers, use_double_conv=use_double_conv
        )

        # Output layer
        flat_size = self._get_flattened_size()
        self.output = ml_util.init_feedforward(flat_size, output_dim, 3)

        # Initialize weights
        self.apply(self.init_weights)

    def _get_flattened_size(self):
        """
        Compute the flattened feature vector size after applying a sequence
        of convolutional and pooling layers on an input tensor with the given
        shape.

        Returns
        -------
        int
            Length of the flattened feature vector after the convolutions and
            pooling.
        """
        with torch.no_grad():
            x = torch.zeros(1, 2, *self.patch_shape)
            x = self.conv_layers(x)
            return x.view(1, -1).size(1)

    @staticmethod
    def init_weights(m):
        """
        Initializes the weights and biases of a given PyTorch layer.

        Parameters
        ----------
        m : nn.Module
            PyTorch layer or module.
        """
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(
                m.weight, mode="fan_in", nonlinearity="leaky_relu"
            )
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Passes the given input through this neural network.

        Parameters
        ----------
        x : torch.Tensor
            Input vector of features.

        Returns
        -------
        x : torch.Tensor
            Output of the neural network.
        """
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x


# --- Transformers ---
class ViT3D(nn.Module):
    """
    A class that implements a 3D Vision transformer.

    Attributes
    ----------
    """

    def __init__(
        self,
        img_shape=(128, 128, 128),
        emb_dim=512,
        depth=6,
        heads=8,
        mlp_dim=1024,
        output_dim=1,
    ):
        """
        Instantiates a ViT3D object.

        Parameters
        ----------
        img_shape : Tuple[int], optional
            Shape of the input image. Default is (128, 128, 128).
        emb_dim : int, optional
            Dimension of the embedding space. Default is 512.
        depth : int, optional
            Number of transformer blocks. Default is 6.
        heads : int, optional
            Number of attention heads in each transformer block. Default 8.
        mlp_dim : int, optional
            Dimension of MLP embedding space. Default is 1024.
        output_dim : int, optional
            Dimension of output. Default is 1.
        """
        # Call parent class
        super().__init__()

        # Token embedding
        self.cls_token = nn.Parameter(torch.empty(1, 1, emb_dim))
        self.img_tokenizer = ImageTokenizer3D(emb_dim, img_shape)

        # Position embedding
        n_tokens = self.img_tokenizer.count_tokens() + 1
        self.pos_embedding = nn.Parameter(torch.empty(1, n_tokens, emb_dim))
        print("# Tokens:", n_tokens)

        # Transformer Blocks
        self.transformer = nn.Sequential(
            *[
                TransformerEncoderBlock(emb_dim, heads, mlp_dim)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(emb_dim)

        # Output layer
        self.output = ml_util.init_feedforward(emb_dim, output_dim, 2)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Initializes the model's weights.
        """
        # Initialize token embedding
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        # Initialize Transformer and output layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        Passes the given input through this neural network.

        Parameters
        ----------
        x : torch.Tensor
            Input vector of features.

        Returns
        -------
        x : torch.Tensor
            Output of the neural network.
        """
        # Tokenize input -> (b, n_tokens, emb_dim)
        x = self.img_tokenizer(x)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Transformer
        x = x + self.pos_embedding[:, : x.size(1)]
        x = self.transformer(x)
        x = self.norm(x[:, 0])

        # Output layer
        x = self.output(x)
        return x


class ImageTokenizer3D(nn.Module):
    """
    A class for learning image token embeddings for transformer-based
    architectures.

    Attributes
    ----------
    dropout : nn.Dropout
        Dropout layer.
    emb_dim : int
        Dimension of the embedding space.
    img_shape : Tuple[int]
        Shape of the input image (D, H, W).
    pos_embedding : nn.Parameter
        Learnable position encoding.
    proj : nn.Conv3d
        Convolutional layer that generates a learnable projection of the
        tokens.
    """

    def __init__(
        self,
        emb_dim,
        img_shape,
        dropout=0.05,
        n_cnn_layers=3,
        n_cnn_channels=32,
    ):
        """
        Instantiates a ImageTokenizer3D object.

        Parameters
        ----------
        emb_dim : int
            Dimension of the embedding space.
        img_shape : Tuple[int]
            Shape of the input image (D, H, W).
        dropout : float, optional
            Dropout probability applied after adding positional embeddings.
            Default is 0.05.
        n_cnn_layers : int, optional
            Number of layers in the CNN that generates the initial token
            embedding. Default is 3.
        """
        # Call parent class
        super().__init__()

        # Class attributes
        self.emb_dim = emb_dim
        self.img_shape = img_shape

        # Image embedding
        cnn_out_channels = n_cnn_channels * (2 ** (n_cnn_layers - 1))
        self.proj = nn.Conv3d(cnn_out_channels, emb_dim, kernel_size=1)
        self.tokenizer = init_cnn3d(
            2, n_cnn_channels, n_cnn_layers, use_double_conv=False
        )

        # Positional embedding
        n_tokens = self.count_tokens()
        self.pos_embedding = nn.Parameter(torch.randn(1, n_tokens, emb_dim))
        self.dropout = nn.Dropout(p=dropout)

    def count_tokens(self):
        """
        Counts the number of tokens that are generated given the patch shape,
        CCN3D architecture, and embedding dimension.

        Returns
        -------
        int
            Number of tokens generated by tokenizer.
        """
        with torch.no_grad():
            dummy = torch.zeros(1, 2, *self.img_shape)
            feats = self.tokenizer(dummy)
            feats = self.proj(feats)
            return feats.flatten(2).shape[-1]

    def forward(self, x):
        """
        Forward pass that converts an input image into a sequence of token
        embeddings.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, D, H, W).

        Returns
        -------
        torch.Tensor
            Token embeddings of shape (B, N, E).
        """
        # Embed image
        x = self.tokenizer(x)
        x = self.proj(x)

        # Tokenize
        x = rearrange(x, "b c d h w -> b (d h w) c")
        x = x + self.pos_embedding
        return self.dropout(x)


class TransformerEncoderBlock(nn.Module):
    """
    Single transformer encoder block.

    Attributes
    ----------
    attn : nn.MultiheadAttention
        Multihead attention block.
    dropout : nn.Dropout
        Dropout layer.
    mlp : nn.Sequential
        Multi-layer perceptron.
    norm1 : nn.LayerNorm
        Applies layer normalization over a mini-batch of the inputs.
    norm2 : nn.LayerNorm
        Applied layer normalization over a mini-batch of the outputs of the
        multihead attention block.
    """

    def __init__(self, emb_dim, heads, mlp_dim, dropout=0):
        """
        Instantiates a TransformerEncoderBlock object.

        Parameters
        ----------
        emb_dim : int
            Dimension of the embedding space.
        heads : int
            Number of attention heads.
        mlp_dim : int
            Dimensionality of the hidden layer in the MLP.
        dropout : float, optional
            Dropout probability applied after attention and MLP layers.
            Default is 0.
        """
        # Call parent class
        super().__init__()

        # Attention head
        self.norm1 = nn.LayerNorm(emb_dim)
        self.attn = nn.MultiheadAttention(
            emb_dim, heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(emb_dim)
        self.mlp = init_mlp(emb_dim, mlp_dim, emb_dim, dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        Forward pass of the encoder block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, N, E).

        Returns
        -------
        x : torch.Tensor
            Output tensor of shape (B, N, E), same as input.
        """
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        x = self.dropout(x)
        return x


# --- Build Simple Neural Networks ---
def init_cnn3d(in_channels, n_feat_channels, n_layers, use_double_conv=True):
    """
    Initializes a convolutional neural network.

    Parameters
    ----------
    in_channels : int
        Number of channels that are input to this convolutional layer.
    out_channels : int
        Number of channels that are output from this convolutional layer.
    n_layers : int
        Number of layers in the network.
    use_double_conv : bool, optional
        Indication of whether to use double convolution. Default is True.

    Returns
    -------
    layers : torch.nn.Sequential
        Sequence of operations that define the network.
    """
    layers = list()
    in_channels = in_channels
    out_channels = n_feat_channels
    for i in range(n_layers):
        # Build layer
        layers.append(
            init_conv_layer(in_channels, out_channels, 3, use_double_conv)
        )

        # Update channel sizes
        in_channels = out_channels
        out_channels *= 2
    return nn.Sequential(*layers)


def init_conv_layer(in_channels, out_channels, kernel_size, use_double_conv):
    """
    Initializes a convolutional layer.

    Parameters
    ----------
    in_channels : int
        Number of channels that are input to this convolutional layer.
    out_channels : int
        Number of channels that are output from this convolutional layer.
    kernel_size : int
        Size of kernel used on convolutional layers.
    use_double_conv : bool
        Indication of whether to use double convolution.

    Returns
    -------
    layers : torch.nn.Sequential
        Sequence of operations that define this convolutional layer.
    """
    # Convolution
    layers = [
        nn.Conv3d(
            in_channels, out_channels, kernel_size, padding=kernel_size // 2
        ),
        nn.BatchNorm3d(out_channels),
        nn.GELU(),
    ]
    if use_double_conv:
        layers += [
            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size,
                padding=kernel_size // 2,
            ),
            nn.BatchNorm3d(out_channels),
            nn.GELU(),
        ]
    # Pooling
    layers.append(nn.MaxPool3d(kernel_size=2))
    return nn.Sequential(*layers)
