"""
Created on Tue June 26 12:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Dataset and dataloader utilities for processing merge site data to train a
model to detect merge errors.

Architecture
------------
BrainDataset
    Owns all data for a single brain: fragment graph, GT graph, image/
    segmentation readers, merge-site KD-tree, and the slice of
    merge_sites_df that belongs to this brain. Handles all per-brain
    positive/negative site retrieval.

BrainDatasetCollection
    Holds an ordered list of BrainDataset objects. Routes global indices to
    the correct brain, exposes split() for train/val partitioning, and is
    the object handed to MergeSiteDataLoader.

ThreadedDataLoader
    Custom DataLoader that uses multithreading to fetch image patches from
    cloud storage and assemble batches.
"""

from scipy.spatial import KDTree
from concurrent.futures import as_completed, ThreadPoolExecutor
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import networkx as nx
import numpy as np
import os
import pandas as pd
import queue
import threading
import torch

from neuron_proofreader.machine_learning.image_dataloader import (
    DetectionPatchLoader as PatchLoader,
)
from neuron_proofreader.machine_learning.geometric_gnn_models import (
    subgraph_to_data,
)
from neuron_proofreader.machine_learning.point_cloud_models import (
    subgraph_to_point_cloud,
)
from neuron_proofreader.skeleton_graph import SkeletonGraph
from neuron_proofreader.utils import ml_util, swc_util, util


# -- Datasets ---
class BrainDataset:
    """
    All data and retrieval logic for a single whole-brain dataset.

    Parameters
    ----------
    brain_id : str
        Unique identifier for this brain.
    subgraph_depth : float
        Radius (in microns) used when extracting rooted subgraphs.
    """

    giant_component_cable_length = 30000
    random_branching_site_probability = 0.7

    def __init__(
        self,
        brain_id,
        img_path,
        sites_prefix,
        swcs_path,
        class_ratios=(0.5, 0.5),
        graph_config=None,
        img_config=None,
        random_nonmerge_site_prob=0.5,
        rebalance_classes=True,
        subgraph_depth=100,
    ):
        # Instance attributes
        self.brain_id = brain_id
        self.class_ratios = class_ratios
        self.ignore_fragments = set()
        self.random_nonmerge_site_prob = random_nonmerge_site_prob
        self.rebalance_classes = rebalance_classes
        self.subgraph_depth = subgraph_depth

        # Core data structures
        self.graph = self.load_fragments(graph_config, swcs_path)
        self.merge_sites = self.load_sites(
            os.path.join(sites_prefix, "merge_sites")
        )
        self.nonmerge_sites = self.load_sites(
            os.path.join(sites_prefix, "nonmerge_sites")
        )
        self.patch_loader = PatchLoader(self.graph, img_config, img_path)

        # Store dataset info
        self.set_giant_components()
        self.set_merge_site_info()
        self.set_valid_branching_nodes()

    def load_fragments(self, config, swcs_path):
        graph = SkeletonGraph(
            anisotropy=config.anisotropy,
            min_cable_length=config.min_cable_length,
            node_spacing=config.node_spacing,
            use_anisotropy=config.use_anisotropy,
            verbose=config.verbose,
        )
        graph.load(swcs_path)
        return graph

    def load_sites(self, sites_prefix):
        sites = list()
        swc_reader = swc_util.Reader(verbose=False)
        for swc_dict in swc_reader(sites_prefix):
            xyz = swc_dict["xyz"][0]
            dd, ii = self.graph.kdtree.query(xyz)
            sites.append(
                {"xyz": xyz, "node": ii, "filename": swc_dict["swc_name"]}
            )
        return pd.DataFrame(sites)

    def set_merge_site_info(self):
        if len(self.merge_sites) > 0:
            # Build kdtree of merge sites
            xyz_arr = np.vstack(self.merge_sites["xyz"])
            self.merge_sites_kdtree = KDTree(xyz_arr)

            # Store fragment IDs corresponding to merge sites
            for xyz in self.merge_sites["xyz"]:
                _, ii = self.graph.kdtree.query(xyz)
                self.ignore_fragments.add(self.graph.node_component_id[ii])

    def set_giant_components(self):
        for nodes in map(list, nx.connected_components(self.graph)):
            # Compute cable length
            root = util.sample_once(nodes)
            cable_length = self.graph.cable_length(
                max_depth=self.giant_component_cable_length, root=root
            )

            # Check if giant component
            if cable_length > self.giant_component_cable_length:
                self.ignore_fragments.add(self.graph.node_component_id[root])

    def set_valid_branching_nodes(self):
        self.valid_branching_nodes = set()
        for node in self.graph.branching_nodes():
            # Reject if high-degree
            if self.graph.degree(node) > 3:
                continue

            # Reject if node belongs to fragment with merge
            # if self.graph.node_component_id[node] in self.ignore_fragments:
            #    continue

            # Reject if branching and near another branching node
            if self._has_nearby_branching(node):
                continue

            # Reject if near merge site
            dd, _ = self.merge_sites_kdtree.query(self.graph.node_xyz[node])
            if dd < 50:
                continue

            # Node is valid
            self.valid_branching_nodes.add(node)

    # --- Site Retrieval ---
    def __getitem__(self, idx):
        node, label = self.get_site(idx)
        subgraph = self.graph.rooted_subgraph(node, self.subgraph_depth)
        patches = self.patch_loader(node)
        return patches, subgraph, label

    def get_site(self, idx):
        if idx > 0:
            return self.merge_sites["node"][idx], 1
        elif np.random.random() < self.random_nonmerge_site_prob:
            return self.get_random_nonmerge_site()
        elif abs(idx) < len(self.nonmerge_sites):
            return self.nonmerge_sites["node"][abs(idx)], 0
        else:
            return self.get_random_nonmerge_site()

    def get_random_nonmerge_site(self):
        if self.use_branching():
            node = util.sample_once(self.valid_branching_nodes)
        else:
            node = util.sample_once(self.graph.nodes)
        return node, 0

    # --- Helpers ---
    def add_nonmerge_sites(self, num_sites):
        # Generate sites
        new_sites = list()
        for _ in range(num_sites):
            node, _ = self.get_random_nonmerge_site()
            site = {
                "node": node,
                "xyz": self.graph.node_xyz[node],
                "filename": "random",
            }
            new_sites.append(site)

        # Add sites to existing
        if len(self.nonmerge_sites) > 0:
            df = pd.DataFrame(new_sites)
            self.nonmerge_sites = pd.concat(
                [df, self.nonmerge_sites], ignore_index=True
            )
        else:
            self.nonmerge_sites = pd.DataFrame(new_sites)

    def use_branching(self):
        outcome = np.random.random() < self.random_branching_site_probability
        return outcome and self.valid_branching_nodes

    def _has_nearby_branching(self, root, max_depth=60):
        queue = [(root, 0)]
        visited = {root}
        while queue:
            # Visit node
            i, d_i = queue.pop()
            if self.graph.degree[i] > 2 and d_i > 0:
                return True

            # Update queue
            for j in self.graph.neighbors(i):
                d_j = d_i + self.graph.dist(i, j)
                if j not in visited and d_j < max_depth:
                    queue.append((j, d_j))
                    visited.add(j)
        return False

    def _list_indices(self):
        # Compute target class counts
        n_pos = len(self.merge_sites)
        n_neg = len(self.nonmerge_sites) or n_pos
        pos_ratio, neg_ratio = self.class_ratios
        n_target_neg = min(int(n_pos * neg_ratio / pos_ratio), n_neg)

        # Check whether to rebalance negative examples
        size = n_target_neg if self.rebalance_classes else n_neg
        neg_idxs = np.random.choice(n_neg, size=size, replace=False)
        return np.concatenate((-neg_idxs, np.arange(n_pos)))

    def __len__(self):
        """
        Counts the number of examples in the dataset.

        Returns
        -------
        int
            Number of examples in the dataset.
        """
        return len(self._list_indices())

    def __repr__(self):
        return (
            f"BrainDataset("
            f"brain_id={self.brain_id}, "
            f"n_examples={len(self)}, "
            f"n_pos_examples={len(self.merge_sites)}, "
            f"n_neg_examples={len(self.nonmerge_sites)})"
        )


class BrainDatasetCollection(Dataset):
    """
    An ordered collection of BrainDataset objects that presents a unified
    Dataset interface to MergeSiteDataLoader.

    Global indices are mapped to (brain_idx, local_idx) pairs via a flat
    index table built from each brain's _list_indices(). The BrainDataset
    handles all site dispatch and retrieval internally.

    Parameters
    ----------
    datasets : List[BrainDataset]
        One BrainDataset per brain.
    """

    def __init__(self, datasets):
        self.datasets = datasets
        self._index_table = self._build_index_table()

    def _build_index_table(self):
        """
        Builds a flat list of (brain_idx, local_idx) pairs by concatenating
        each brain's _list_indices(). Rebuilt whenever the collection changes.

        Returns
        -------
        table : List[Tuple[int, int]]
        """
        table = []
        for b_idx, bd in enumerate(self.datasets):
            for local_idx in bd._list_indices():
                table.append((b_idx, int(local_idx)))
        return table

    # --- Dataset Interface ---
    def __len__(self):
        return len(self._index_table)

    def __getitem__(self, idx):
        """
        Gets one example: (patches, subgraph, label).

        Parameters
        ----------
        idx : int
            Index into the flat index table.

        Returns
        -------
        ...
        """
        b_idx, local_idx = self._index_table[idx]
        return self.datasets[b_idx][local_idx]

    def __repr__(self):
        return (
            f"BrainDatasetCollection("
            f"n_brains={len(self.datasets)}, "
            f"n_examples={len(self)})"
        )

    def get_idxs(self):
        """
        Returns shuffleable indices over the full index table.

        Returns
        -------
        numpy.ndarray
            Indices over the full index table.
        """
        return np.arange(len(self._index_table))


# --- Dataloader ---
class ThreadedDataLoader(DataLoader):

    _VALID_MODALITIES = {None, "graph", "pointcloud"}

    def __init__(
        self,
        dataset,
        batch_size=32,
        is_multimodal=False,
        modality=None,
        sampler=None,
        use_shuffle=True,
        prefetch_batches=8,
    ):
        # Check that modality is valid
        if modality not in self._VALID_MODALITIES:
            raise ValueError(
                f"modality must be one of {self._VALID_MODALITIES}, "
                f"got {modality!r}."
            )

        # Call parent class
        super().__init__(dataset, batch_size=batch_size, sampler=sampler)

        # Instance attributes
        self.is_multimodal = is_multimodal
        self.modality = modality
        self.use_shuffle = use_shuffle
        self.prefetch_batches = prefetch_batches
        self.img_shape = (2,) + dataset.datasets[0].patch_loader.patch_shape

        # Set batch loader
        if self.is_multimodal and self.modality == "graph":
            self._load_batch = self._load_image_graph_batch
        elif self.is_multimodal and self.modality == "pointcloud":
            self._load_batch = self._load_image_pc_batch
        else:
            self._load_batch = self._load_image_batch

    def __iter__(self):
        # Extract indices
        self.dataset._index_table = self.dataset._build_index_table()
        idxs = self.dataset.get_idxs()
        if self.use_shuffle:
            np.random.shuffle(idxs)

        # Split into batches upfront
        batch_idx_groups = [
            idxs[start: min(start + self.batch_size, len(idxs))]
            for start in range(0, len(idxs), self.batch_size)
        ]

        # Sentinel signalling the prefetch thread is done
        _DONE = object()
        buffer = queue.Queue(maxsize=self.prefetch_batches)

        def prefetch_worker():
            try:
                for batch_idxs in batch_idx_groups:
                    buffer.put(self._load_batch(batch_idxs))
            except Exception as e:
                buffer.put(e)
            finally:
                buffer.put(_DONE)

        thread = threading.Thread(target=prefetch_worker, daemon=True)
        thread.start()

        while True:
            item = buffer.get()
            if item is _DONE:
                break
            if isinstance(item, Exception):
                raise item
            yield item

        thread.join()

    def _load_image_batch(self, batch_idxs):
        """
        Loads a batch of samples from the dataset using multithreading.

        Parameters
        ----------
        batch_idxs : List[int]
            Indices of the dataset items to include in the batch.

        Returns
        -------
        patches : torch.Tensor
            Image patches for the batch.
        targets : torch.Tensor
            Target labels corresponding to each patch.
        """
        with ThreadPoolExecutor() as executor:
            # Assign threads
            pending = dict()
            for i, idx in enumerate(batch_idxs):
                thread = executor.submit(self.dataset.__getitem__, idx)
                pending[thread] = i

            # Store results
            patches = np.zeros((len(batch_idxs),) + self.img_shape)
            targets = np.zeros((len(batch_idxs), 1))
            for thread in as_completed(pending.keys()):
                i = pending.pop(thread)
                patches[i], _, targets[i] = thread.result()
        return ml_util.to_tensor(patches), ml_util.to_tensor(targets)

    def _load_image_pc_batch(self, batch_idxs):
        """
        Loads a batch of samples from the dataset using multithreading.

        Parameters
        ----------
        batch_idxs : List[int]
            Indices of the dataset items to include in the batch.

        Returns
        -------
        batch : Dict[str, torch.Tensor]
            Dictionary that maps modality names to batch features.
        targets : torch.Tensor
            Target labels corresponding to each patch.
        """
        with ThreadPoolExecutor() as executor:
            # Assign threads
            pending = dict()
            for i, idx in enumerate(batch_idxs):
                thread = executor.submit(self.dataset.__getitem__, idx)
                pending[thread] = i

            # Store results
            patches = np.zeros((len(batch_idxs),) + self.shape)
            targets = np.zeros((len(batch_idxs), 1))
            point_clouds = np.zeros((len(batch_idxs), 3, 3600))
            for thread in as_completed(pending.keys()):
                i = pending.pop(thread)
                patches[i], subgraph, targets[i] = thread.result()
                point_clouds[i] = subgraph_to_point_cloud(subgraph)

        # Set batch dictionary
        batch = ml_util.TensorDict(
            {
                "img": ml_util.to_tensor(patches),
                "point_cloud": ml_util.to_tensor(point_clouds),
            }
        )
        return batch, ml_util.to_tensor(targets)

    def _load_image_graph_batch(self, idxs):
        """
        Loads a batch of samples from the dataset using multithreading.

        Parameters
        ----------
        idxs : List[int]
            Indices of the dataset items to include in the batch.

        Returns
        -------
        batch : Dict[str, torch.Tensor]
            Dictionary that maps modality names to batch features.
        targets : torch.Tensor
            Target labels corresponding to each patch.
        """
        with ThreadPoolExecutor() as executor:
            # Assign threads
            threads = list()
            for idx in idxs:
                threads.append(executor.submit(self.dataset.__getitem__, idx))

            # Store results
            targets = np.zeros((len(idxs), 1))
            patches = np.zeros((len(idxs),) + self.img_shape)
            h, x, edge_index, batches = list(), list(), list(), list()
            node_offset = 0
            for i, thread in enumerate(as_completed(threads)):
                patches[i], subgraph, targets[i] = thread.result()
                h_i, x_i, edge_index_i = subgraph_to_data(subgraph)
                n_i = h_i.size(0)

                edge_index_i += node_offset
                h.append(h_i)
                x.append(x_i)
                edge_index.append(edge_index_i)
                batches.append(torch.full((n_i,), i, dtype=torch.long))

                node_offset += n_i

        # Combine subgraph batches
        h = torch.cat(h, dim=0)
        x = torch.cat(x, dim=0)
        edge_index = torch.cat(edge_index, dim=1)
        batches = torch.cat(batches, dim=0)

        # Set batch dictionary
        batch = ml_util.TensorDict(
            {
                "img": ml_util.to_tensor(patches),
                "graph": (h, x, edge_index, batches),
            }
        )
        return batch, ml_util.to_tensor(targets)


# --- Sites Loading ---
def create_dataset_collection(
    brain_ids,
    dataset_mode,
    img_prefixes_path,
    sites_root_path,
    swcs_root_path,
    class_ratios=(0.5, 0.5),
    graph_config=None,
    img_config=None,
    subgraph_depth=100,
):
    # Set parameters based on mode
    assert dataset_mode in ["Train", "Val"]
    if dataset_mode == "Train":
        img_config.set_train_mode()
        random_nonmerge_site_prob = 0.5
        rebalance_classes = True
    else:
        random_nonmerge_site_prob = 0
        rebalance_classes = False

    # Load image prefixes
    bucket, root_prefix = util.parse_cloud_path(sites_root_path)
    img_prefixes = util.read_json(img_prefixes_path)

    # Iterate over brains
    datasets = list()
    for brain_id in tqdm(brain_ids, desc=f"Load {dataset_mode} Dataset"):
        # Extract dataset info
        img_path = os.path.join(img_prefixes[brain_id], "0")
        segmentation_id = get_segmentation_id(sites_root_path, brain_id)
        sites_path = os.path.join(sites_root_path, brain_id, segmentation_id)
        swcs_path = util.get_google_swcs_prefix(
            swcs_root_path, brain_id, segmentation_id
        )

        # Add dataset
        dataset = BrainDataset(
            brain_id,
            img_path,
            sites_path,
            swcs_path,
            class_ratios=class_ratios,
            graph_config=graph_config,
            img_config=img_config,
            subgraph_depth=subgraph_depth,
            random_nonmerge_site_prob=random_nonmerge_site_prob,
            rebalance_classes=rebalance_classes,
        )

        # Check whether to generate examples for validation
        if dataset_mode == "Val":
            num_target_neg = 10 * len(dataset.merge_sites)
            num_added_neg = num_target_neg - len(dataset.nonmerge_sites)
            dataset.add_nonmerge_sites(num_added_neg)

        # Add dataset to collection
        datasets.append(dataset)
    return BrainDatasetCollection(datasets)


def get_segmentation_id(sites_path, brain_id):
    brain_sites_path = os.path.join(sites_path, brain_id)
    subprefixes = util.list_gcs_subprefixes(brain_sites_path)
    assert len(subprefixes) == 1
    return subprefixes[0].split("/")[-2]
