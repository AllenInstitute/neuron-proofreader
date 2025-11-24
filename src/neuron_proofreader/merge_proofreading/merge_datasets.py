"""
Created on Wed July 2 11:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Dataset and dataloader utilities for processing merge site data to train model
to detect merge errors.

"""

from concurrent.futures import as_completed, ThreadPoolExecutor
from scipy.spatial import KDTree
from torch.utils.data import Dataset, DataLoader

import copy
import networkx as nx
import numpy as np
import pandas as pd
import random

from neuron_proofreader.machine_learning.augmentation import ImageTransforms
from neuron_proofreader.machine_learning.point_clouds import (
    subgraph_to_point_cloud,
)
from neuron_proofreader.merge_proofreading.merge_dataloading import (
    get_brain_merge_sites
)
from neuron_proofreader.skeleton_graph import SkeletonGraph
from neuron_proofreader.utils import (
    geometry_util,
    img_util,
    ml_util,
    swc_util,
    util,
)


# --- Datasets ---
class MergeSiteDataset(Dataset):
    """
    Dataset class for loading and processing merge site data. The core data
    structure is the attribute "merge_sites_df" which contains metadata about
    each merge site.

    Attributes
    ----------
    anisotropy : Tuple[float], optional
        Image to physical coordinates scaling factors to account for the
        anisotropy of the microscope.
    subgraph_radius : int, optional
        Radius (in microns) around merge sites used to extract rooted
        subgraphs.
    gt_graphs : Dict[str, SkeletonGraph]
        Dictionary that maps brain IDs to a graph containing ground truth
        skeletons.
    graphs : Dict[str, SkeletonGraph]
        Dictionary that maps brain IDs to graphs containing fragments that
        have merge mistakes.
    img_readers : Dict[str, ImageReader]
        Image readers used to read raw images from cloud bucket.
    merge_sites_df : pandas.DataFrame
        DataFrame containing merge sites, must contain the columns: "brain_id"
        "segmentation_id", "segment_id", and "xyz".
    node_spacing : int, optional
        Spacing (in microns) between neighboring nodes in graphs.
    patch_shape : Tuple[int], optional
        Shape of the 3D image patches to extract.
    """

    def __init__(
        self,
        merge_sites_df,
        anisotropy=(1.0, 1.0, 1.0),
        subgraph_radius=100,
        node_spacing=5,
        patch_shape=(128, 128, 128),
    ):
        """
        Instantiates a MergeSiteDataset object.

        Parameters
        ----------
        merge_sites_df : pandas.DataFrame
            DataFrame containing merge sites, must contain the columns:
            "brain_id", "segmentation_id", "segment_id", and "xyz".
        anisotropy : Tuple[float], optional
            Image to physical coordinates scaling factors to account for the
            anisotropy of the microscope. Default is (1.0, 1.0, 1.0).
        subgraph_radius : int, optional
            Radius (in microns) around merge sites used to extract rooted
            subgraph. Default is 100μm.
        node_spacing : int, optional
            Spacing between nodes in the graph. Default is 5μm.
        patch_shape : Tuple[int], optional
            Shape of the 3D patches to extract. Default is (128, 128, 128).
        """
        # Instance attributes
        self.anisotropy = anisotropy
        self.node_spacing = node_spacing
        self.merge_sites_df = merge_sites_df
        self.patch_shape = patch_shape
        self.subgraph_radius = subgraph_radius

        # Data structures
        self.img_readers = dict()
        self.graphs = dict()
        self.gt_graphs = dict()
        self.merge_site_kdtrees = dict()

    # --- Load Data ---
    def load_fragment_graphs(self, brain_id, swc_pointer):
        """
        Loads fragments containing merge mistakes for a whole-brain dataset,
        then stores them in the "graphs" attribute.

        Parameters
        ----------
        brain_id : str
            Unique identifier for a whole-brain dataset.
        swc_pointer : str
            Pointer to SWC files to be loaded into a graph.
        """
        # Load graphs
        graph = SkeletonGraph(node_spacing=self.node_spacing)
        graph.load(swc_pointer)

        # Filter non-merge components
        idxs = self.merge_sites_df["brain_id"] == brain_id
        merged_segment_ids = self.merge_sites_df["segment_id"][idxs].values
        for swc_id in graph.get_swc_ids():
            segment_id = swc_util.get_segment_id(swc_id)
            if str(segment_id) not in merged_segment_ids:
                component_id = util.find_key(
                    graph.component_id_to_swc_id, swc_id
                )
                nodes = graph.get_nodes_with_component_id(component_id)
                graph.remove_nodes(nodes, relabel_nodes=False)
        graph.relabel_nodes()

        # Post process fragments
        self.clip_fragments_to_groundtruth(brain_id, graph)
        self.graphs[brain_id] = graph

        # Build merge site kdtrees
        pts = get_brain_merge_sites(self.merge_sites_df, brain_id)
        self.merge_site_kdtrees[brain_id] = KDTree(pts)

    def load_gt_graphs(self, brain_id, swc_pointer):
        """
        Loads ground truth skeletons for a whole-brain dataset, then stores
        them in the "gt_graphs" attribute.

        Parameters
        ----------
        brain_id : str
            Unique identifier for a whole-brain dataset.
        swc_pointer : str
            Pointer to SWC files to be loaded into graph.
        """
        node_spacing = 2 * self.node_spacing
        self.gt_graphs[brain_id] = SkeletonGraph(node_spacing=node_spacing)
        self.gt_graphs[brain_id].load(swc_pointer)
        self.gt_graphs[brain_id].set_kdtree()

    def load_image(self, brain_id, img_path):
        """
        Loads image reader for a whole-brain dataset, then stores it in the
        "img_readers" attribute.

        Parameters
        ----------
        brain_id : str
            Unique identifier for a whole-brain dataset.
        swc_pointer : str
            Pointer to SWC files to be loaded into graph.
        """
        self.img_readers[brain_id] = img_util.TensorStoreReader(img_path)

    # --- Create Subclass Dataset ---
    def subset(self, cls, idxs):
        """
        Creates a derived dataset keeping only specified indices.

        Parameters
        ----------
        cls : class
            Class of the new dataset.
        idxs : List[int]
            Indices of merge sites to keep.

        Returns
        -------
        new_dataset : cls
            New dataset instance containing only the specified subset.
        """
        new_dataset = cls.__new__(cls)
        new_dataset.__dict__ = copy.deepcopy(self.__dict__)
        new_dataset.remove_nonindexed_fragments(idxs)
        return new_dataset

    def remove_nonindexed_fragments(self, idxs):
        """
        Removes fragments that do not correspond to the given site indices.

        Parameters
        ----------
        idxs : List[int]
            Indices of merge sites to keep. Fragments associated with all
            other sites are removed.
        """
        # Remove other fragments
        for idx in [i for i in self.merge_sites_df.index if i not in idxs]:
            # Extract site info
            brain_id = self.merge_sites_df["brain_id"][idx]
            xyz = self.merge_sites_df["xyz"][idx]

            # Find fragment containing site
            dist, node = self.graphs[brain_id].kdtree.query(xyz)
            if dist < 20 and node in self.graphs[brain_id]:
                nodes = self.graphs[brain_id].get_connected_nodes(node)
                self.graphs[brain_id].remove_nodes(nodes, False)

        # Relabel nodes
        for brain_id in self.graphs:
            self.graphs[brain_id].relabel_nodes()

        # Update merge sites df
        self.merge_sites_df = self.merge_sites_df.iloc[idxs]
        self.merge_sites_df = self.merge_sites_df.reset_index(drop=True)

    # --- Getters ---
    def __getitem__(self, idx):
        """
        Gets the example corresponding to the given index, which consists of
        an image patch, label mask, and rooted subgraph.

        Parameters
        ----------
        idx : int
            Index of example to retrieve. Positive indices correspond to merge
            sites, while non-positive indices correspond to non-merge sites.

        Returns
        -------
        patches : numpy.ndarray
            Array containing the image patch and segment mask with shape
            (2, D, H, W).
        subgraph : networkx.Graph
            Rooted subgraph centered at the site node.
        label : int
            1 if the example is positive and 0 otherwise.
        """
        # Get example
        brain_id, subgraph, label = self.get_site(idx)
        voxel = img_util.to_voxels(subgraph.node_xyz[0], self.anisotropy)

        # Extract subgraph and image patches centered at site
        img_patch = self.get_img_patch(brain_id, voxel)
        segment_mask = self.get_segment_mask(subgraph)

        # Stack image channels
        try:
            patches = np.stack([img_patch, segment_mask], axis=0)
        except ValueError:
            img_patch = img_util.pad_to_shape(img_patch, self.patch_shape)
            patches = np.stack([img_patch, segment_mask], axis=0)
        return patches, subgraph, label

    def get_indexed_negative_site(self, idx):
        """
        Gets the negative example corresponding to the given index.

        Parameters
        ----------
        idx : int
            Index of the site in "sites_df".

        Returns
        -------
        brain_id : str
            Unique identifier for the whole-brain dataset containing the site.
        node : int
            Node ID of the site.
        """
        # Get site info
        idx = abs(idx)
        brain_id = self.merge_sites_df["brain_id"].iloc[idx]
        xyz = self.merge_sites_df["xyz"].iloc[idx]
        node = self.gt_graphs[brain_id].find_closest_node(xyz)

        # Extract rooted subgraph
        subgraph = self.gt_graphs[brain_id].get_rooted_subgraph(
            node, self.subgraph_radius
        )
        return brain_id, subgraph

    def get_indexed_positive_site(self, idx):
        """
        Gets the positive example corresponding to the given index.

        Parameters
        ----------
        idx : int
            Index of the site in "sites_df".

        Returns
        -------
        brain_id : str
            Unique identifier for the whole-brain dataset containing the site.
        node : int
            Node ID of the site.
        """
        # Check if site has a fragment
        if idx > 0 and not self.has_fragment(idx):
            return self.get_random_negative_site()

        # Get site info
        brain_id = self.merge_sites_df["brain_id"].iloc[idx]
        xyz = self.merge_sites_df["xyz"].iloc[idx]
        node = self.graphs[brain_id].find_closest_node(xyz)

        # Extract rooted subgraph
        subgraph = self.graphs[brain_id].get_rooted_subgraph(
            node, self.subgraph_radius
        )
        return brain_id, subgraph

    def get_random_negative_site(self):
        """
        Gets a random non-merge site from a fragment graph.

        Returns
        -------
        brain_id : str
            Unique identifier of the whole-brain dataset containing the site.
        node : int
            Node ID of the site.
        """
        # Sample graph
        brain_id = util.sample_once(list(self.graphs.keys()))

        # Sample node on graph
        outcome = random.random()
        while True:
            # Sample node
            if outcome <= 0.3:
                node = util.sample_once(self.graphs[brain_id].nodes)
            elif outcome > 0.3 and outcome < 0.4:
                node = util.sample_once(self.graphs[brain_id].get_leafs())
            else:
                branching_nodes = self.graphs[brain_id].get_branchings()
                if len(branching_nodes) > 0:
                    node = util.sample_once(branching_nodes)
                    if self.check_nearby_branching(brain_id, node):
                        continue
                else:
                    outcome = 0
                    continue

            # Check if node is close to merge site
            xyz = self.graphs[brain_id].node_xyz[node]
            d, _ = self.merge_site_kdtrees[brain_id].query(xyz)
            if d > 50:
                break

        # Extract rooted subgraph
        subgraph = self.graphs[brain_id].get_rooted_subgraph(
            node, self.subgraph_radius
        )
        return brain_id, subgraph

    def get_img_patch(self, brain_id, center):
        """
        Extracts and normalizes a 3D image patch from the specified whole-
        brain dataset.

        Parameters
        ----------
        brain_id : str
            Unique identifier of the whole-brain dataset to read from.
        center : numpy.ndarray
            Voxel coordinates of the patch center.

        Returns
        -------
        img_patch : numpy.ndarray
            Extracted image patch, which has been normalized and clipped to a
            maximum value of 300.
        """
        img_patch = self.img_readers[brain_id].read(center, self.patch_shape)
        img_patch = img_util.normalize(np.minimum(img_patch, 300))
        return img_patch

    def get_segment_mask(self, subgraph):
        """
        Generates a binary mask for a given subgraph within a patch.

        Parameters
        ----------
        subgraph : SkeletonGraph
            Rooted subgraph centered at the site node.

        Returns
        -------
        segment_mask : numpy.ndarray
            Binary mask for a given subgraph within a patch.
        """
        center = subgraph.get_voxel(0)
        segment_mask = np.zeros(self.patch_shape)
        for node1, node2 in subgraph.edges:
            # Get local voxel coordinates
            voxel1 = subgraph.get_local_voxel(node1, center, self.patch_shape)
            voxel2 = subgraph.get_local_voxel(node2, center, self.patch_shape)

            # Populate mask
            voxels = geometry_util.make_digital_line(voxel1, voxel2)
            img_util.annotate_voxels(segment_mask, voxels, kernel_size=3)
        return segment_mask

    # --- Helpers ---
    def __len__(self):
        """
        Returns the number of positive and negative examples of merge sites.

        Returns
        -------
        int
            Number of positive examples of merge sites.
        """
        return len(self.merge_sites_df)

    def check_nearby_branching(self, brain_id, root, max_depth=20):
        """
        Checks if there is a branching node within a specified depth from the
        given node.

        Parameters
        ----------
        brain_id : str
            Unique identifier for graph to be searched.
        root : int
            Node ID.
        max_depth : float, optional
            Maximum depth (in microns) of search. Default is 20μm.

        Returns
        -------
        bool
            Indication of whether there is a nearby branching node.
        """
        queue = [(root, 0)]
        visited = set([root])
        while queue:
            # Visit node
            i, d_i = queue.pop()
            if self.graphs[brain_id].degree[i] > 2 and d_i > 0:
                return True

            # Update queue
            for j in self.graphs[brain_id].neighbors(i):
                d_j = d_i + self.graphs[brain_id].dist(i, j)
                if j not in visited and d_j < max_depth:
                    queue.append((j, d_j))
                    visited.add(j)
        return False

    def clip_fragments_to_groundtruth(self, brain_id, graph):
        """
        Removes any node from the given fragment that is more than 100μm from
        the ground truth graph.

        Parameters
        ----------
        brain_id : str
            Unique identifier for a whole-brain dataset.
        graph : SkeletonGraph
            Fragment graph to be clipped.
        """
        assert brain_id in self.gt_graphs, "Must load GT before fragments!"
        for i, xyz in enumerate(graph.node_xyz):
            if i in graph:
                d, _ = self.gt_graphs[brain_id].kdtree.query(xyz)
                if d > 128:
                    graph.remove_node(i)
        graph.relabel_nodes()

    def count_fragments(self):
        """
        Counts the number of fragments in the dataset.

        Returns
        -------
        cnt : int
            Number of fragments in the dataset.
        """
        cnt = 0
        for graph in self.graphs.values():
            cnt += nx.number_connected_components(graph)
        return cnt

    def has_fragment(self, idx):
        """
        Checks whether a neuron fragment exists in the corresponding graph.

        Parameters
        ----------
        idx : int
            Index of the merge site in "self.merge_sites_df" to check.

        Returns
        -------
        bool
            True if the fragment exists in the graph for the corresponding
            brain, False otherwise.
        """
        brain_id = self.merge_sites_df["brain_id"].iloc[idx]
        segment_id = self.merge_sites_df["segment_id"].iloc[idx]
        swc_id = f"{segment_id}.0"
        return swc_id in self.graphs[brain_id].get_swc_ids()


class MergeSiteTrainDataset(MergeSiteDataset):
    """
    A class for storing and retrieving training examples.
    """

    def __init__(self, base_dataset=None, idxs=None):
        """
        Instantiates a MergeSiteTrainDataset object.

        Parameters
        ----------
        base_dataset : MergeSiteDataset, optional
            Dataset to be instantiated as a train dataset.
        idxs : List[int], optional
            Indices of examples to be kept in train dataset.
        """
        # Create sub-dataset
        subset_dataset = base_dataset.subset(self.__class__, idxs)
        self.__dict__.update(subset_dataset.__dict__)

        # Instance attributes
        self.transform_positive = ImageTransforms()
        self.transform_negative = ImageTransforms(use_geometric=False)

    # --- Getters ---
    def __getitem__(self, idx):
        """
        Gets the example specified by the given index.

        Parameters
        ----------
        idx : int
            Index of example.

        Returns
        -------
        patches : numpy.ndarray
            Array of stacked channels containing the image patch and label
            mask with shape (2, D, H, W).
        subgraph : SkeletonGraph
            Rooted subgraph centered at the site node.
        label : int
            1 if the example is positive and 0 otherwise.
        """
        # Call parent routine
        patches, subgraph, label = super().__getitem__(idx)

        # Apply image augmentation (if applicable)
        if label > 0:
            self.transform_positive(patches)
        else:
            self.transform_negative(patches)
        return patches, subgraph, label

    def get_site(self, idx):
        """
        Retrieves a merge or nonmerge site specified by the given index.

        Parameters
        ----------
        idx : int
            Index of site to retrieve. Positive indices correspond to merge
            sites, non-positive indices correspond to non-merge sites.

        Returns
        -------
        brain_id : str
            Unique identifier of the brain containing the site.
        node : int
            Node ID of the site.
        label : int
            1 if the example is positive and 0 otherwise.
        """
        if idx > 0:
            brain_id, subgraph = self.get_indexed_positive_site(idx)
        else:
            if np.random.random() < 0.3:
                idx = np.random.randint(0, len(self.merge_sites_df))
                brain_id, subgraph = self.get_indexed_negative_site(idx)
            else:
                brain_id, subgraph = self.get_random_negative_site()
        return brain_id, subgraph, int(idx > 0)


class MergeSiteValDataset(MergeSiteDataset):
    """
    A class for storing and retrieving validation examples.
    """

    def __init__(self, base_dataset=None, idxs=None):
        """
        Instantiates a MergeSiteValDataset object.

        Parameters
        ----------
        base_dataset : MergeSiteDataset, optional
            Dataset to be instantiated as a validation dataset.
        idxs : List[int], optional
            Indices of examples to be kept in validation dataset.
        """
        # Create sub-dataset
        subset_dataset = base_dataset.subset(self.__class__, idxs)
        self.__dict__.update(subset_dataset.__dict__)

        # Instance attributes
        self.negative_examples = self.generate_negative_examples()

    def generate_negative_examples(self):
        """
        Generates examples of non-merge sites by sampling points on fragments
        that are sufficiently far from a merge site.

        Returns
        -------
        negative_examples : pandas.DataFrame
            Dataframe containing non-merge sites that are specified by a brain
            and node ID.
        """
        negative_examples = list()
        for i in range(len(self.merge_sites_df)):
            # Get example
            if np.random.random() < 0.3:
                brain_id, subgraph = self.get_indexed_negative_site(i)
            else:
                brain_id, subgraph = self.get_random_negative_site()

            # Store info
            negative_examples.append(
                {
                    "brain_id": brain_id,
                    "subgraph": subgraph,
                    "xyz": subgraph.node_xyz[0]
                }
            )
        return negative_examples

    # --- Getters ---
    def get_site(self, idx):
        """
        Retrieves a merge or nonmerge site specified by the given index.

        Parameters
        ----------
        idx : int
            Index of site to retrieve. Positive indices correspond to merge
            sites, non-positive indices correspond to non-merge sites.

        Returns
        -------
        brain_id : str
            Unique identifier of the brain containing the site.
        node : int
            Node ID of the site.
        label : int
            1 if the example is positive and 0 otherwise.
        """
        if idx > 0:
            brain_id, subgraph = self.get_indexed_positive_site(idx)
        else:
            brain_id = self.negative_examples[abs(idx)]["brain_id"]
            subgraph = self.negative_examples[abs(idx)]["subgraph"]

        label = 1 if idx > 0 else 0
        return brain_id, subgraph, label


# --- DataLoaders ---
class MergeSiteDataLoader(DataLoader):
    """
    A custom DataLoader class that uses multithreading to read image patches
    from the cloud to form batches.
    """

    def __init__(self, dataset, batch_size=32, sampler=None):
        """
        Instantiates a MergeSiteDataLoader object.

        Parameters
        ----------
        dataset : MergeSiteDataset
            Dataset to be iterated over to train or validate.
        batch_size : int, optional
            Number of examples in each batch. Default is 32.
        """
        # Call parent class
        super().__init__(dataset, batch_size=batch_size, sampler=sampler)

        # Instance attributes
        self.patches_shape = (2,) + self.dataset.patch_shape

    # --- Core Routines ---
    def __iter__(self):
        """
        Generates batches of examples for training and validation.

        Returns
        -------
        iterator
            Generates batch of examples used during training and validation.
        """
        # Set indices
        idxs = np.arange(-len(self.dataset) // 2, len(self.dataset) // 2)
        random.shuffle(idxs)

        # Iterate over indices
        for start in range(0, len(idxs), self.batch_size):
            batch_idxs = idxs[start: start + self.batch_size]
            yield self._load_batch(batch_idxs)

    def _load_batch(self, batch_idxs):
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
        labels : torch.Tensor
            Labels corresponding to each patch.
        """
        with ThreadPoolExecutor() as executor:
            # Assign threads
            threads = list()
            for idx in batch_idxs:
                threads.append(executor.submit(self.dataset.__getitem__, idx))

            # Store results
            patches = np.zeros((self.batch_size,) + self.patches_shape)
            labels = np.zeros((self.batch_size, 1))
            for i, thread in enumerate(as_completed(threads)):
                patches[i], _, labels[i] = thread.result()
        return ml_util.to_tensor(patches), ml_util.to_tensor(labels)

    def _load_multimodal_batch(self, batch_idxs):
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
        labels : torch.Tensor
            Labels corresponding to each patch.
        """
        with ThreadPoolExecutor() as executor:
            # Assign threads
            threads = list()
            for idx in batch_idxs:
                threads.append(executor.submit(self.dataset.__getitem__, idx))

            # Store results
            labels = np.zeros((self.batch_size, 1))
            patches = np.zeros((self.batch_size,) + self.patches_shape)
            point_clouds = np.zeros((self.batch_size, 3, 3600))
            for i, thread in enumerate(as_completed(threads)):
                patches[i], subgraph, labels[i] = thread.result()
                point_clouds[i] = subgraph_to_point_cloud(subgraph)

        # Set batch dictionary
        batch = ml_util.TensorDict(
            {
                "img": ml_util.to_tensor(patches),
                "point_cloud": ml_util.to_tensor(point_clouds),
            }
        )
        return batch, ml_util.to_tensor(labels)
