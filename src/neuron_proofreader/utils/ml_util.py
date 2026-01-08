"""
Created on Sat November 04 15:30:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Helper routines for training and performing inference with machine learning
models.

"""

import networkx as nx
import numpy as np
import torch
import torch.nn as nn


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
def get_inputs(data, device="cpu"):
    """
    Extracts input data for a graph-based model and optionally moves it to a
    GPU.

    Parameters
    ----------
    data : torch_geometric.data.HeteroData
        A data object with the following attributes:
            - x_dict: Dictionary of node features for different node types.
            - edge_index_dict: Dictionary of edge indices for edge types.
            - edge_attr_dict: Dictionary of edge attributes for edge types.
    device : str, optional
        Target device for the data, 'cuda' for GPU and 'cpu' for CPU. The
        default is "cpu".

    Returns
    --------
    tuple:
        Tuple containing the following:
            - x (dict): Node features dictionary.
            - edge_index (dict): Edge indices dictionary.
            - edge_attr (dict): Edge attributes dictionary.
    """
    data.to(device)
    return data.x_dict, data.edge_index_dict, data.edge_attr_dict


def line_graph(edges):
    """
    Initializes a line graph from a list of edges.

    Parameters
    ----------
    edges : list
        List of edges.

    Returns
    -------
    networkx.Graph
        Line graph generated from a list of edges.
    """
    graph = nx.Graph()
    graph.add_edges_from(edges)
    return nx.line_graph(graph)


def load_model(model, model_path, device="cuda"):
    """
    Loads a PyTorch model checkpoint, moves the model to the speficied device,
    and sets it to evaluation mode.

    Parameters
    ----------
    model : torch.nn.Module
        Instantiated model architecture into which the weights will be loaded.
    model_path : str or Path
        Path to the saved PyTorch checkpoint.
    device : str, optional
        Device to load the model onto. Default is "cuda".
    """
    model.load_state_dict(torch.load(model_path, map_location=device))
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
    List[float]
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
