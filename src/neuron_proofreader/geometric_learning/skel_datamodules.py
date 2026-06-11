"""
Created on Mon June 8 17:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

...

"""

from random import shuffle
from torch.utils.data import Dataset, DataLoader, Sampler

import networkx as nx
import numpy as np
import torch

from neuron_proofreader.skeleton_graph import SkeletonGraph
from neuron_proofreader.utils import util


# --- Dataset Classes ---
class PathsDataset(Dataset):

    def __init__(
        self,
        brain_id,
        swcs_path,
        bin_width=400,
        graph_config=None,
        max_path_length=10e4,
        transform=None,
    ):
        # Instance attributes
        self.brain_id = brain_id
        self.bin_width = bin_width
        self.max_path_length = max_path_length

        # Core data structures
        self.graph = self.load_skeletons(graph_config, swcs_path)
        self.paths = self.irreducible_paths()
        self.transform = transform

        self.set_bins()

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

    def set_bins(self):
        # Initialize bins
        db = self.bin_width
        n_bins = int(np.ceil(self.max_path_length / db))
        self.bins = {(i * db, (i + 1) * db): [] for i in range(n_bins)}

        # Store path indices in bins
        for idx, p in enumerate(self.paths):
            i = min(int(self.path_length(p) / db), n_bins - 1)
            self.bins[(i * db, (i + 1) * db)].append(idx)

    # --- Get Examples ---
    def __getitem__(self, bin_id):
        # Get path
        idx = util.sample_once(self.bins[bin_id])
        path = self.paths[idx].copy()

        # Check whether to subsample
        if self.path_length(path) > self.max_path_length:
            new_length = np.random.uniform(*bin_id)
            node = util.sample_once(path)
            path = self.path_thru_node(node, max_depth=new_length)

        # Check whether to transform
        curve = self.node_xyz[path]
        if self.transform:
            curve = self.transform(curve)

        # Normalize
        curve = (curve - curve[0]) / self.max_path_length
        return curve

    # --- Helpers ---
    def __getattr__(self, name):
        return getattr(self.graph, name)

    def __len__(self):
        return len(self.paths)

    def __repr__(self):
        lengths = [self.path_length(p) for p in self.paths]
        num_neurons = nx.number_connected_components(self.graph)
        return (
            f"BrainDataset("
            f"\n   brain_id={self.brain_id}, "
            f"\n   num_neurons={num_neurons}, "
            f"\n   num_paths={len(self)}, "
            f"\n   min_path_length={np.min(lengths):.2f}, "
            f"\n   mean_path_length={np.mean(lengths):.2f}, "
            f"\n   max_path_length={np.max(lengths):.2f},"
            f"\n)"
        )


class PathsDatasetCollection(Dataset):

    def __init__(self, datasets, examples_per_bin=32):
        """
        Collection of PathsDataset instances, one per brain, with a unified
        bin structure for sampling uniformly across brains and path lengths.

        Parameters
        ----------
        datasets : List[PathsDataset]
            List of PathsDataset instances, one per brain.
        """
        self.datasets = datasets
        self.examples_per_bin = examples_per_bin
        self.set_bins()

    def set_bins(self):
        """
        Builds a unified bin structure across all datasets. Each bin key is a
        (lower, upper) tuple and maps to a list of (dataset_idx, path_idx)
        pairs.
        """
        # Collect all bin keys across datasets
        all_keys = set()
        for ds in self.datasets:
            all_keys.update(ds.bins.keys())

        self.bins = {k: [] for k in sorted(all_keys)}
        for ds_idx, ds in enumerate(self.datasets):
            for bin_id, path_indices in ds.bins.items():
                for path_idx in path_indices:
                    self.bins[bin_id].append((ds_idx, path_idx))

    def __getitem__(self, bin_id):
        """
        Samples a random (dataset, path) pair from the given bin.

        Parameters
        ----------
        bin_id : Tuple[float, float]
            The (lower, upper) bin key.

        Returns
        -------
        numpy.ndarray
            Normalized curve of shape (N, 3).
        """
        ds_idx, path_idx = util.sample_once(self.bins[bin_id])
        return self.datasets[ds_idx][bin_id]

    # --- Helpers ---
    def all_path_lengths(self):
        path_lengths = list()
        for dataset in self.datasets:
            path_lengths.extend(
                [dataset.path_length(p) for p in dataset.paths]
            )
        return np.array(path_lengths)

    def generate_bin_ids(self):
        nonempty_keys = [k for k, v in self.bins.items() if v]
        bin_ids = list(nonempty_keys) * self.examples_per_bin
        shuffle(bin_ids)
        return bin_ids

    def __len__(self):
        return sum(len(ds) for ds in self.datasets)

    def __repr__(self):
        n_brains = len(self.datasets)
        n_paths = len(self)
        non_empty = sum(1 for v in self.bins.values() if len(v) > 0)
        return (
            f"PathsDatasetCollection("
            f"num_brains={n_brains}, "
            f"num_paths={n_paths}, "
            f"num_bins={non_empty})"
        )


# --- DataLoader Classes ---
class BinSampler(Sampler):
    """
    Sampler that yields bin IDs uniformly across all non-empty bins,
    with a fixed number of examples per bin per epoch.
    """
    def __init__(self, dataset, examples_per_bin=10):
        """
        Parameters
        ----------
        dataset : PathsDatasetCollection
            Dataset to sample from.
        examples_per_bin : int, optional
            Number of examples to draw from each bin per epoch. Default is 10.
        """
        self.examples_per_bin = examples_per_bin
        self.non_empty_bins = [k for k, v in dataset.bins.items() if len(v) > 0]

    def __iter__(self):
        bin_ids = self.non_empty_bins * self.examples_per_bin
        shuffle(bin_ids)
        return iter(bin_ids)

    def __len__(self):
        return len(self.non_empty_bins) * self.examples_per_bin


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
    dataset, batch_size=32, examples_per_bin=32, num_workers=0
):
    """
    Builds a DataLoader for a PathsDatasetCollection that samples uniformly
    across all non-empty bins.

    Parameters
    ----------
    dataset : PathsDatasetCollection
        Dataset to load from.
    examples_per_bin : int, optional
        Number of examples per bin per epoch. Default is 10.
    batch_size : int, optional
        Number of curves per batch. Default is 32.
    num_workers : int, optional
        Number of worker processes for data loading. Default is 4.

    Returns
    -------
    DataLoader
    """
    sampler = BinSampler(dataset, examples_per_bin=examples_per_bin)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_curves,
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler,
    )
