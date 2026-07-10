"""
Created on Sat November 04 15:30:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Helper routines for training and performing inference with machine learning
models.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        assert n_layers > 1
        self.net = self.build_network(input_dim, output_dim, n_layers)

    def build_network(self, input_dim, output_dim, n_layers):
        # Set input/output dimensions
        input_dim_i = input_dim
        output_dim_i = max(input_dim // 2, 4)

        # Build architecture
        layers = []
        for i in range(n_layers):
            mlp = init_mlp(input_dim_i, input_dim_i * 2, output_dim_i)
            layers.append(mlp)

            input_dim_i = output_dim_i
            output_dim_i = (
                max(output_dim_i // 2, 4) if i < n_layers - 2 else output_dim
            )

        # Initialize weights
        net = nn.Sequential(*layers)
        net.apply(self._init_weights)
        return net

    @staticmethod
    def _init_weights(m):
        """
        Initializes weights for linear layers using Kaiming initialization.

        Parameters
        ----------
        m : torch.nn.Module
            Module to initialize.
        """
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


# --- Performance Metric Class ---
class BinaryMetricAccumulator:

    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.loss = 0.0
        self.n = 0

    @torch.no_grad()
    def update(self, pred, y, loss):
        pred = pred.bool()
        y = y.bool()

        self.tp += (pred & y).sum().item()
        self.fp += (pred & ~y).sum().item()
        self.fn += (~pred & y).sum().item()

        self.loss += loss.item()
        self.n += y.numel()

        del pred, y, loss

    def compute(self):
        precision = self.tp / (self.tp + self.fp + 1e-8)
        recall = self.tp / (self.tp + self.fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "loss": self.loss / max(self.n, 1),
        }


# --- Data Structures ---
class TensorDict(dict):
    """
    A class for model inputs in a dictionary.
    """

    def to(self, device):
        """
        Moves dictionary values to the specified GPU device.

        Parameters
        ----------
        device : str
            Name of GPU device to move inputs to.

        Returns
        -------
        TensorDict
            Dictionary with values moved to the specified GPU device.
        """
        return TensorDict({k: self.move(v, device) for k, v in self.items()})

    def move(self, v, device):
        """
        Moves the given dictionary value to the speficied GPU device.

        Parameters
        ----------
        v : object
            Value to be moved to GPU device.
        device : str
            Name of GPU device to move value to.

        Returns
        -------
        object
            Value moved to the specified GPU device.
        """
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
def find_max_batch_size(
    model, input_shape, optimizer_cls, device="cuda", start=1, max_bs=32
):
    model.to(device)
    lo, hi = start, max_bs
    best = 0
    while lo <= hi:
        bs = (lo + hi) // 2
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            x = torch.randn(bs, *input_shape, device=device)
            y = torch.randn(bs, 1, device=device)
            opt = optimizer_cls(model.parameters())
            scaler = torch.cuda.amp.GradScaler()

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model(x)
                loss = F.mse_loss(out, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            best = bs
            lo = bs + 1
            del x, y, out, loss, opt
        except torch.cuda.OutOfMemoryError:
            hi = bs - 1
        finally:
            torch.cuda.empty_cache()
    return best


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
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()


def tensor_to_list(tensor):
    """
    Converts the given tensor to a list.

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
    Moves PyTorch tensor to CPU and optionally converts it to a NumPy array.

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
        return tensor.detach().cpu().numpy()
    else:
        return tensor.detach().cpu()


def to_tensor(arr):
    """
    Converts a NumPy array to a tensor.

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
