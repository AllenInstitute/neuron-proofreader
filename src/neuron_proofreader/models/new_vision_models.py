"""
Created on Thu July 2 13:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code for vision models that perform image classification tasks within
NeuronProofreader pipelines.

"""

import torch.nn as nn
import torch.nn.functional as F

from neuron_proofreader.utils.ml_util import FeedForwardNet


class NewCNN3D(nn.Module):
    """
    Convolutional neural network for 3D images.
    """

    def __init__(
        self,
        input_shape,
        base_channels=16,
        channel_multiplier=2,
        depth=5,
        dropout=0.1,
        max_channels=256,
        num_single_blocks=2,
        output_dim=1,
        use_double=True,
    ):
        """
        Instantiates a CNN3D object.

        Parameters
        ----------
        output_dim : int, optional
            Dimension of output. Default is 1.
        dropout : float, optional
            Fraction of values to randomly drop during training. Default is
            0.1.
        depth : int, optional
            Number of convolutional blocks. Default is 5.
        use_double : bool, optional
            Indication of whether to use double convolution. Default is True.
        """
        # Call parent class
        nn.Module.__init__(self)

        # Instance attributes
        self.dropout = dropout
        self.encode = Encoder3D(
            input_shape[0],
            base_channels,
            depth,
            channel_multiplier=channel_multiplier,
            use_double=use_double,
            max_channels=max_channels,
            num_single_blocks=num_single_blocks,
        )
        self.output = FeedForwardNet(self.encode.out_channels, output_dim, 3)

        # Initialize weights
        self.apply(self.init_weights)

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
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.GroupNorm):
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
        x = self.encode(x)
        x = F.adaptive_avg_pool3d(x, 1).flatten(1)
        x = self.output(x)
        return x


class Encoder3D(nn.Module):
    """
    Sequence of ConvBlock3D blocks with growing (capped) channel width.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        depth,
        channel_multiplier=2,
        use_double=True,
        max_channels=256,
        num_single_blocks=2,
    ):
        """
        Instantiates an Encoder3D object.

        Parameters
        ----------
        in_channels : int
            Number of channels input to the first block.
        out_channels : int
            Number of channels output by the first block.
        depth : int
            Number of conv blocks in the encoder.
        channel_multiplier : float, optional
            Multiplicative channel growth factor per layer. Default is 2.
        use_double : bool, optional
            Indication of whether blocks use double convolution. Default is True.
        max_channels : int, optional
            Cap on channel growth across layers. Default is 128.
        """
        # Call parent class
        super().__init__()

        # Create encoding blocks
        blocks = list()
        for i in range(depth):
            use_double = i > num_single_blocks
            block = ConvBlock3D(
                in_channels, out_channels, use_double=use_double
            )
            blocks.append(block)
            in_channels = block.out_channels
            out_channels = int(min(out_channels * channel_multiplier, max_channels))

        # Instance attributes
        self.blocks = nn.ModuleList(blocks)
        self.out_channels = self.blocks[-1].out_channels

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class ConvBlock3D(nn.Module):

    def __init__(
        self, in_channels, out_channels, kernel_size=3, use_double=True
    ):
        """
        Instantiates a ConvBlock3D object.

        Parameters
        ----------
        in_channels : int
            Number of input channels to this block.
        out_channels : int
            Number of output channels from this block.
        kernel_size : int, optional
            Size of kernel used on convolutional layers. Default is 3.
        use_double : bool, optional
            Indication of whether to apply a second conv+norm+act before
            pooling. Default is True.
        """
        # Call parent class
        super().__init__()

        # Instance attributes
        self.out_channels = out_channels

        # Create encoding layers
        layers = self.create_unit(in_channels, out_channels, kernel_size)
        if use_double:
            layers.extend(
                self.create_unit(out_channels, out_channels, kernel_size)
            )

        self.net = nn.Sequential(*layers, nn.MaxPool3d(kernel_size=2))

    def forward(self, x):
        return self.net(x)

    # --- Helpers ---
    def create_unit(self, in_channels, out_channels, kernel_size):
        padding = kernel_size // 2
        n_groups = self.get_num_groups(out_channels, 8)
        unit = [
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
            ),
            nn.GroupNorm(n_groups, out_channels),
            nn.GELU()
        ]
        return unit

    @staticmethod
    def get_num_groups(num_channels, max_groups=8):
        for g in reversed(range(1, max_groups + 1)):
            if num_channels % g == 0:
                return g
        return 1
