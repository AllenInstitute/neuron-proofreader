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

MergeSiteDataLoader
    Custom DataLoader that uses multithreading to fetch image patches from
    cloud storage and assemble batches.
"""

from scipy.spatial import KDTree
from concurrent.futures import as_completed, ThreadPoolExecutor
from torch.utils.data import Dataset, DataLoader

import networkx as nx
import numpy as np
import os
import queue
import random
import threading
import torch

from neuron_proofreader.machine_learning.augmentation import ImageTransforms
from neuron_proofreader.machine_learning.geometric_gnn_models import (
    subgraph_to_data,
)
from neuron_proofreader.machine_learning.point_cloud_models import (
    subgraph_to_point_cloud,
)
from neuron_proofreader.skeleton_graph import SkeletonGraph
from neuron_proofreader.utils import (
    geometry_util,
    img_util,
    ml_util,
    swc_util,
    util,
)


# ---------------------------------------------------------------------------
# BrainDataset
# ---------------------------------------------------------------------------


class BrainDataset:
    """
    All data and retrieval logic for a single whole-brain dataset.

    Parameters
    ----------
    brain_id : str
        Unique identifier for this brain.
    anisotropy : Tuple[float]
        Voxel-to-physical scaling factors.
    brightness_clip : int
        Maximum raw image intensity before normalisation.
    subgraph_radius : int
        Radius (um) used when extracting rooted subgraphs.
    node_spacing : int
        Spacing (um) between neighbouring graph nodes.
    patch_shape : Tuple[int]
        Shape of the 3D image patches to extract.
    use_segmentation_mask : bool
        Whether to overlay a volumetric segmentation when building the
        segment mask.
    """

    def __init__(
        self,
        brain_id,
        sites_prefix,
        pos_site_paths,
        neg_site_paths,
        img_path,
        anisotropy=(1.0, 1.0, 1.0),
        brightness_clip=500,
        subgraph_radius=100,
        node_spacing=5,
        patch_shape=(128, 128, 128),
        probability_random_nonmerge_site=0.5,
        use_transform=False,
        use_segmentation_mask=False,
    ):
        # Instance attributes
        self.anisotropy = anisotropy
        self.brain_id = brain_id
        self.brightness_clip = brightness_clip
        self.subgraph_radius = subgraph_radius
        self.node_spacing = node_spacing
        self.patch_shape = patch_shape
        self.probability_random_nonmerge_site = (
            probability_random_nonmerge_site
        )
        self.use_segmentation_mask = use_segmentation_mask

        # Core data structures
        self.graph = self.load_fragments(sites_prefix)
        self.img = img_util.TensorStoreImage(img_path)
        self.segmentation_reader = None

        self.nonmerge_sites = self.load_sites(neg_site_paths)
        self.merge_sites = self.load_sites(pos_site_paths)
        self.set_merge_site_info()

        # Image augmentation for training
        self.transform = ImageTransforms() if use_transform else None

    def load_fragments(self, sites_prefix):
        graph = SkeletonGraph(
            anisotropy=self.anisotropy,
            node_spacing=self.node_spacing,
            use_anisotropy=False,
            verbose=True,
        )
        graph.load(os.path.join(sites_prefix, "fragments"))
        return graph

    def load_sites(self, site_paths):
        sites = list()
        swc_reader = swc_util.Reader(verbose=False)
        for swc_dict in swc_reader(site_paths):
            sites.append(swc_dict["xyz"])
        return np.vstack(sites)

    def set_merge_site_info(self):
        # Build kdtree of merge sites
        self.merge_sites_kdtree = KDTree(self.merge_sites)

        # Store fragment IDs corresponding to merge sites
        self.fragments_with_merge = set()
        for xyz in self.merge_sites:
            _, ii = self.graph.kdtree.query(xyz)
            self.fragments_with_merge.add(self.graph.node_component_id[ii])

    # --- Site retrieval ---
    def __getitem__(self, idx):
        # Get example
        node, label = self.get_site(idx)
        subgraph = self.graph.rooted_subgraph(node, self.subgraph_radius)

        # Get voxel coordinate
        voxel = subgraph.node_voxel(0)
        if self.transform:
            voxel += np.random.randint(-6, 6 + 1, size=3)

        # Extract subgraph and image patches centered at site
        img_patch = self.get_img_patch(voxel)
        segment_mask = self.get_segment_mask(voxel, subgraph)

        # Stack image channels
        try:
            patches = np.stack([img_patch, segment_mask], axis=0)
        except ValueError:
            img_patch = img_util.pad_to_shape(img_patch, self.patch_shape)
            patches = np.stack([img_patch, segment_mask], axis=0)
        return patches, subgraph, label

    def get_site(self, idx):
        if idx > 0:
            return self.get_merge_site(idx)
        elif np.random.random() < self.probability_random_nonmerge_site:
            return self.get_random_nonmerge_site()
        elif abs(idx) < len(self.nonmerge_sites):
            return self.get_indexed_nonmerge_site(abs(idx))
        else:
            return self.get_random_nonmerge_site()

    def get_merge_site(self, idx):
        _, node = self.graph.kdtree.query(self.merge_sites[idx])
        return node, 1

    def get_indexed_nonmerge_site(self, idx):
        _, node = self.graph.kdtree.query(self.nonmerge_sites[idx])
        return node, 0

    def get_random_nonmerge_site(self):
        # Search for valid nonmerge site
        branching_nodes = self.graph.branching_nodes()
        use_branching = branching_nodes and random.random() < 0.5
        for cnt in range(10**4):
            # Sample node
            if use_branching:
                node = util.sample_once(branching_nodes)
            else:
                node = util.sample_once(self.graph.nodes)

            # Reject if high-degree
            if self.graph.degree(node) > 3:
                continue

            # Reject if branching and near another branching node
            if use_branching and self._has_nearby_branching(node):
                continue

            # Reject if near merge site
            dd, _ = self.merge_sites_kdtree.query(self.graph.node_xyz[node])
            if dd < 100:
                continue

            # Site is valid
            break

        return node, 0

    # --- Image / Mask Extraction ---
    def get_img_patch(self, center):
        """
        Extracts, clips, and normalises a 3D image patch centred at center.

        Parameters
        ----------
        center : numpy.ndarray
            Voxel coordinates of the patch centre.

        Returns
        -------
        numpy.ndarray
        """
        patch = self.img.read(center, self.patch_shape)
        patch = np.minimum(patch, self.brightness_clip)
        return img_util.normalize(patch)

    def get_segment_mask(self, center, subgraph):
        """
        Builds the segment mask for subgraph, optionally incorporating a
        volumetric segmentation read.

        Parameters
        ----------
        center : numpy.ndarray
            Voxel coordinates of the patch centre.
        subgraph : SkeletonGraph

        Returns
        -------
        numpy.ndarray
        """
        if self.use_segmentation_mask:
            return self._segment_mask_with_segmentation(center, subgraph)
        return self._segment_mask_skeleton_only(subgraph)

    def _segment_mask_skeleton_only(self, subgraph):
        mask = np.zeros(self.patch_shape)
        center = subgraph.node_voxel(0)
        offset = img_util.get_offset(center, self.patch_shape)
        for node1, node2 in subgraph.edges:
            v1 = subgraph.node_local_voxel(node1, offset)
            v2 = subgraph.node_local_voxel(node2, offset)
            img_util.annotate_voxels(
                mask, geometry_util.make_digital_line(v1, v2)
            )
        return mask

    def _segment_mask_with_segmentation(self, center, subgraph):
        mask = self.segmentation_reader.read(center, self.patch_shape)
        mask = img_util.remove_small_segments(mask, 1000)
        mask = 0.5 * (mask > 0).astype(float)
        offset = img_util.get_offset(center, self.patch_shape)
        for node1, node2 in subgraph.edges:
            v1 = subgraph.node_local_voxel(node1, offset)
            v2 = subgraph.node_local_voxel(node2, offset)
            img_util.annotate_voxels(
                mask, geometry_util.make_digital_line(v1, v2)
            )
        return mask

    # --- Private helpers ---
    def _list_indices(self):
        # Set idxs
        pos_idxs = np.arange(len(self.merge_sites))
        neg_idxs = np.arange(len(self.nonmerge_sites))

        # Check for class imbalance
        if len(neg_idxs) < len(pos_idxs):
            neg_idxs = -pos_idxs
        else:
            neg_idxs = -np.random.choice(
                neg_idxs, size=len(pos_idxs), replace=False
            )
        return np.concatenate((pos_idxs, neg_idxs))

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

    def __len__(self):
        return len(self._list_indices())


# ---------------------------------------------------------------------------
# BrainDatasetCollection
# ---------------------------------------------------------------------------


class BrainDatasetCollection(Dataset):
    """
    An ordered collection of BrainDataset objects that presents a unified
    Dataset interface to MergeSiteDataLoader.

    Global indices are mapped to (brain_idx, local_idx) pairs via a flat
    index table built from each brain's _list_indices(). The BrainDataset
    handles all site dispatch and retrieval internally.

    Parameters
    ----------
    brain_datasets : List[BrainDataset]
        One BrainDataset per brain.
    augmentation : callable or None
        Applied to (2, D, H, W) patch arrays in-place during __getitem__.
        Pass None when augmentation is not needed (e.g. validation).
    """

    def __init__(self, brain_datasets):
        self.brain_datasets = brain_datasets
        self._index_table = self._build_index_table()

    def _build_index_table(self):
        """
        Builds a flat list of (brain_idx, local_idx) pairs by concatenating
        each brain's _list_indices(). Rebuilt whenever the collection changes.

        Returns
        -------
        List[Tuple[int, int]]
        """
        table = []
        for b_idx, bd in enumerate(self.brain_datasets):
            for local_idx in bd._list_indices():
                table.append((b_idx, int(local_idx)))
        return table

    # --- Dataset interface ---
    def __len__(self):
        return len(self._index_table)

    def __getitem__(self, idx):
        """
        Returns one example: (patches, subgraph, label).

        Parameters
        ----------
        idx : int
            Index into the flat index table.

        Returns
        -------
        patches : numpy.ndarray  shape (2, D, H, W)
        subgraph : SkeletonGraph
        label : int
        """
        b_idx, local_idx = self._index_table[idx]
        return self.brain_datasets[b_idx][local_idx]

    def get_idxs(self):
        """
        Returns shuffleable indices over the full index table.

        Returns
        -------
        numpy.ndarray
        """
        return np.arange(len(self._index_table))

    # --- Helpers ---

    def brain_ids(self):
        """Returns the list of brain IDs in this collection."""
        return [bd.brain_id for bd in self.brain_datasets]

    def n_merge_sites(self):
        """Returns the total number of merge sites across all brains."""
        return sum(len(bd.merge_sites) for bd in self.brain_datasets)

    def count_fragments(self):
        """Returns the total number of fragments across all brains."""
        return sum(
            nx.number_connected_components(bd.graph)
            for bd in self.brain_datasets
            if bd.graph is not None
        )

    def __repr__(self):
        return (
            f"BrainDatasetCollection("
            f"n_brains={len(self.brain_datasets)}, "
            f"n_examples={len(self)})"
        )


# ---------------------------------------------------------------------------
# MergeSiteDataLoader
# ---------------------------------------------------------------------------


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
        self.patches_shape = (2,) + dataset.brain_datasets[0].patch_shape

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
        batch_index_groups = [
            idxs[start: min(start + self.batch_size, len(idxs))]
            for start in range(0, len(idxs), self.batch_size)
        ]

        # Sentinel signalling the prefetch thread is done
        _DONE = object()
        buffer = queue.Queue(maxsize=self.prefetch_batches)

        def prefetch_worker():
            try:
                for batch_idxs in batch_index_groups:
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
            patches = np.zeros((len(batch_idxs),) + self.patches_shape)
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
            patches = np.zeros((len(batch_idxs),) + self.patches_shape)
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
            patches = np.zeros((len(idxs),) + self.patches_shape)
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
                batches.append(
                    torch.full((n_i,), i, dtype=torch.long)
                )

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
                "graph": (h, x, edge_index, batches)
            }
        )
        return batch, ml_util.to_tensor(targets)
