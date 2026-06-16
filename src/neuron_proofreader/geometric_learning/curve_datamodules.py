"""
Created on Mon June 8 17:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

...

"""

from copy import deepcopy
from torch.utils.data import Dataset, DataLoader, Sampler

import networkx as nx
import numpy as np
import pandas as pd
import torch

from neuron_proofreader.skeleton_graph import SkeletonGraph


# --- Dataset Classes ---
class PathsDataset(Dataset):

    def __init__(
        self,
        brain_id,
        swcs_path,
        graph_config=None,
        max_length=np.inf,
        transform=None,
    ):
        # Instance attributes
        self.brain_id = brain_id
        self.max_length = max_length
        self.transform = transform

        # Core data structures
        self.graph = self.load_skeletons(graph_config, swcs_path)
        self.paths = self.get_valid_paths()

    def load_skeletons(self, config, swcs_path):
        graph = SkeletonGraph(
            anisotropy=config.anisotropy,
            min_cable_length=config.min_cable_length,
            node_spacing=config.node_spacing,
            use_anisotropy=config.use_anisotropy,
            verbose=config.verbose,
        )
        graph.load(swcs_path)
        return graph

    def get_valid_paths(self):
        paths = list()
        for p in self.irreducible_paths():
            if self.path_length(p) < self.max_length:
                paths.append(p)
        return paths

    # --- Get Examples ---
    def __getitem__(self, i):
        # Get path
        curve = deepcopy(self.node_xyz[self.paths[i]])
        if self.transform:
            curve = self.transform(curve)

        # Normalize
        curve -= curve[0]
        curve[1:] -= curve[:-1]
        return curve

    # --- Helpers ---
    def path_lengths(self):
        return np.array([self.path_length(p) for p in self.paths])

    def __getattr__(self, name):
        return getattr(self.graph, name)

    def __len__(self):
        return len(self.paths)

    def __repr__(self):
        lengths = self.path_lengths()
        num_neurons = nx.number_connected_components(self.graph)
        return (
            f"BrainDataset("
            f"\n   brain_id={self.brain_id}, "
            f"\n   num_neurons={num_neurons}, "
            f"\n   num_paths={len(self)}, "
            f"\n   min_length={np.min(lengths):.2f}, "
            f"\n   mean_length={np.mean(lengths):.2f}, "
            f"\n   max_length={np.max(lengths):.2f},"
            f"\n)"
        )


class PathsDatasetCollection(Dataset):

    def __init__(self, datasets, is_val=False, n_val_examples=1000, seed=42):
        """
        Parameters
        ----------
        datasets : List[PathsDataset]
            List of PathsDataset instances, one per brain.
        is_val : bool, optional
            If True, precomputes a fixed set of examples at construction time.
            Default is False.
        n_val_examples : int, optional
            Number of fixed validation examples to precompute. Default is 1000.
        seed : int, optional
            Random seed for reproducible val set. Default is 42.
        """
        # Instance attributes
        self.datasets = datasets
        self.is_val = is_val
        self.set_examples_df()

        # Check whether to set validation examples
        if is_val:
            self.val_examples = self.set_val_examples(n_val_examples, seed)

    def set_examples_df(self):
        rows = []
        for ds_idx, dataset in enumerate(self.datasets):
            ds_idxs = np.full(len(dataset), ds_idx)
            p_idxs = np.arange(len(dataset))
            ds_lengths = dataset.path_lengths()
            rows.append(
                pd.DataFrame(
                    {
                        "ds_idx": ds_idxs,
                        "path_idx": p_idxs,
                        "length": ds_lengths,
                    }
                )
            )
        self.examples_df = pd.concat(rows, ignore_index=True)

    def set_val_examples(self, n, seed):
        """
        Samples n examples with fixed seed, strips transforms, and caches
        the resulting examples.
        """
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(self.examples_df), size=n, replace=False)
        examples = []
        for i in indices:
            ds_idx = self.examples_df["ds_idx"][i]
            path_idx = self.examples_df["path_idx"][i]
            dataset = self.datasets[ds_idx]
            examples.append(dataset[path_idx])
        return examples

    # --- Data Fetching ---
    def __getitem__(self, i):
        # Case 1: validation example
        if self.is_val:
            return self.val_examples[i]

        # Case 2: train example
        ds_idx = self.examples_df["ds_idx"][i]
        path_idx = self.examples_df["path_idx"][i]
        return self.datasets[ds_idx][path_idx]

    def __len__(self):
        if self.is_val:
            return len(self.val_examples)
        return len(self.examples_df)

    def __repr__(self):
        return (
            f"PathsDatasetCollection("
            f"num_brains={len(self.datasets)}, "
            f"num_paths={len(self.examples_df)}) "
        )


# --- DataLoader Classes ---
class PathSampler(Sampler):

    def __init__(self, dataset, examples_per_epoch):
        """
        Parameters
        ----------
        dataset : PathsDatasetCollection
            Dataset to sample from.
        """
        self.dataset = dataset
        self.examples_per_epoch = examples_per_epoch

    def __iter__(self):
        idxs = self.dataset.examples_df.sample(
            self.examples_per_epoch, replace=True, weights="length"
        ).index
        return iter(np.array(idxs))

    def __len__(self):
        return self.examples_per_epoch


def collate_curves(curves):
    """
    Pads a list of curves to the longest in the batch and generates an
    attention mask.

    Parameters
    ----------
    curves : List[numpy.ndarray]
        Each of shape (N_i, 3), where N_i can vary.

    Returns
    -------
    padded : torch.Tensor
        Shape (B, N_max, 3), zero-padded.
    mask : torch.Tensor
        Shape (B, N_max), True where padding.
    """
    lengths = [len(c) for c in curves]
    n_max = max(lengths)
    B = len(curves)

    padded = torch.zeros(B, n_max, 3)
    mask = torch.ones(B, n_max, dtype=torch.bool)
    for i, (c, l) in enumerate(zip(curves, lengths)):
        padded[i, :l] = torch.tensor(c)
        mask[i, :l] = False
    return padded, mask


def build_dataloader(
    dataset,
    batch_size=32,
    examples_per_epoch=5000,
    num_workers=0,
    use_sampler=True,
):
    """
    Builds a DataLoader for a PathsDatasetCollection that samples uniformly
    across all non-empty bins.

    Parameters
    ----------
    dataset : PathsDatasetCollection
        Dataset to load from.
    examples_per_epoch : int, optional
        Number of examples per epoch. Default is 5000.
    batch_size : int, optional
        Number of curves per batch. Default is 32.
    num_workers : int, optional
        Number of worker processes for data loading. Default is 0.

    Returns
    -------
    DataLoader
    """
    sampler = PathSampler(dataset, examples_per_epoch) if use_sampler else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_curves,
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler,
    )
