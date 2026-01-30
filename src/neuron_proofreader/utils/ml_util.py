"""
Created on Sat November 04 15:30:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Helper routines for training and performing inference with machine learning
models.

"""

import numpy as np
import torch
import torch.nn as nn


# --- Architectures ---
class FeedForwardNet(nn.Module):
    """
    A class that implements a feed forward neural network.
    """

    def __init__(self, input_dim, output_dim, n_layers):
        """
        Instantiates a FeedFowardNet object.

        Parameters
        ----------
        input_dim : int
            Dimension of the input.
        output_dim : int
            Dimension of the output of the network.
        n_layers : int
            Number of layers in the network.
        """
        # Call parent class
        super().__init__()

        # Instance attributes
        self.net = self.build_network(input_dim, output_dim, n_layers)

    def build_network(self, input_dim, output_dim, n_layers):
        # Set input/output dimensions
        input_dim_i = input_dim
        output_dim_i = input_dim // 2

        # Build architecture
        layers = []
        for i in range(n_layers):
            mlp = init_mlp(input_dim_i, input_dim_i * 2, output_dim_i)
            layers.append(mlp)

            input_dim_i = input_dim_i // 2
            output_dim_i = (
                output_dim_i // 2 if i < n_layers - 2 else output_dim
            )

        # Initialize weights
        net = nn.Sequential(*layers)
        net.apply(self._init_weights)
        return net

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

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
        return self.net(x)


def init_mlp(input_dim, hidden_dim, output_dim, dropout=0.1):
    """
    Initializes a multi-layer perceptron (MLP).

    Parameters
    ----------
    input_dim : int
        Dimension of input.
    hidden_dim : int
        Dimension of the hidden layer.
    output_dim : int
        Dimension of output.
    dropout : float, optional
        Fraction of values to randomly drop during training. Default is 0.1.

    Returns
    -------
    mlp : nn.Sequential
        Multi-layer perception network.
    """
    mlp = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LeakyReLU(),
        nn.Dropout(p=dropout),
        nn.Linear(hidden_dim, output_dim),
    )
    return mlp


# --- Data Structures ---
class TensorDict(dict):

    def to(self, device):
        return TensorDict({k: self.move(v, device) for k, v in self.items()})

    def move(self, v, device):
        if torch.is_tensor(v):
            if v.dtype == torch.float64:
                v = v.float()
            return v.to(device, non_blocking=False)
        elif hasattr(v, "to"):
            v = v.to(device)
            if hasattr(v, "pos") and isinstance(v.pos, torch.Tensor):
                if v.pos.dtype == torch.float64:
                    v.pos = v.pos.float()
            return v
        elif isinstance(v, dict):
            return {kk: self.move(vv, device) for kk, vv in v.items()}
        elif isinstance(v, tuple):
            return tuple([self.move(vv, device) for vv in v])
        else:
            return v


# --- Miscellaneous ---
def load_model(model, model_path, device="cuda"):
    """
    Loads a PyTorch model checkpoint, moves the model to the speficied device,
    and sets it to evaluation mode.

    Parameters
    ----------
    model : torch.nn.Module
        Instantiated model architecture into which the weights will be loaded.
    model_path : str
        Path to the saved PyTorch checkpoint.
    device : str, optional
        Device to load the model onto. Default is "cuda".
    """
    state_dict = torch.load(model_path, map_location=device)
    fixed_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("output.") and not k.startswith("output.net."):
            k = k.replace("output.", "output.net.", 1)
        fixed_state_dict[k] = v

    model.load_state_dict(fixed_state_dict, strict=False)
    model.to(device)
    model.eval()


def tensor_to_list(tensor):
    """
    Converts a tensor to a list.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor with shape Nx1 to be converted.

    Returns
    -------
   tensor : List[float]
        Tensor converted to a list.
    """
    return to_cpu(tensor).flatten().tolist()


def to_cpu(tensor, to_numpy=False):
    """
    Move PyTorch tensor to the CPU and optionally convert it to a NumPy array.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor to be moved to CPU.
    to_numpy : bool, optional
        If True, converts the tensor to a NumPy array. Default is False.

    Returns
    -------
    torch.Tensor or np.ndarray
        Tensor or array on CPU.
    """
    if to_numpy:
        return np.array(tensor.detach().cpu())
    else:
        return tensor.detach().cpu()


def to_tensor(arr):
    """
    Converts a numpy array to a tensor.

    Parameters
    ----------
    arr : numpy.ndarray
        Array to be converted.

    Returns
    -------
    torch.Tensor
        Array converted to tensor.
    """
    return torch.tensor(arr, dtype=torch.float32)
