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
import os
import pandas as pd
import random
import torch

from neuron_proofreader.machine_learning.augmentation import ImageTransforms
from neuron_proofreader.machine_learning.geometric_gnn_models import (
    subgraph_to_data,
)
from neuron_proofreader.machine_learning.point_cloud_models import (
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
    random_negative_example_prob = 0.8

    def __init__(
        self,
        merge_sites_df,
        anisotropy=(1.0, 1.0, 1.0),
        brightness_clip=500,
        subgraph_radius=100,
        node_spacing=5,
        patch_shape=(128, 128, 128),
        use_new_mask=False,
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
        brightness_clip : int, optional
            ...
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
        self.brightness_clip = brightness_clip
        self.node_spacing = node_spacing
        self.merge_sites_df = merge_sites_df
        self.patch_shape = patch_shape
        self.subgraph_radius = subgraph_radius
        self.use_new_mask = use_new_mask

        # Data structures
        self.graphs = dict()
        self.gt_graphs = dict()
        self.img_readers = dict()
        self.segmentation_readers = dict()
        self.merge_site_kdtrees = dict()

    # --- Load Data ---
    def load_fragment_graphs(
        self, brain_id, swc_pointer, use_anisotropy=True
    ):
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
        graph = SkeletonGraph(
            anisotropy=self.anisotropy,
            node_spacing=self.node_spacing,
            use_anisotropy=use_anisotropy
        )
        graph.load(swc_pointer)

        # Remove groundtruth skeletons
        for swc_id in graph.get_swc_ids():
            if swc_id.lower().startswith("n"):
                component_id = util.find_key(
                    graph.component_id_to_swc_id, swc_id
                )
                nodes = graph.get_nodes_with_component_id(component_id)
                graph.remove_nodes(nodes, relabel_nodes=False)

        # Remove fragments excluded from merge sites
        brain_idxs = self.merge_sites_df["brain_id"] == brain_id
        merge_sites = self.merge_sites_df[brain_idxs]
        segment_ids = set(merge_sites["segment_id"].unique())
        for nodes in map(list, list(nx.connected_components(graph))):
            node = util.sample_once(nodes)
            segment_id = graph.get_node_segment_id(node)
            if segment_id not in segment_ids:
                graph.remove_nodes(nodes, relabel_nodes=False)
        graph.relabel_nodes()

        # Build merge site kdtrees
        pts = get_brain_merge_sites(self.merge_sites_df, brain_id)
        self.merge_site_kdtrees[brain_id] = KDTree(pts)

        # Post process fragments
        self.clip_fragments_to_groundtruth(brain_id, graph)
        self.graphs[brain_id] = graph

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
        self.gt_graphs[brain_id] = SkeletonGraph(
            anisotropy=self.anisotropy,
            node_spacing=self.node_spacing,
        )
        self.gt_graphs[brain_id].load(swc_pointer)
        self.gt_graphs[brain_id].set_kdtree()

    def load_images(self, brain_id, img_path, segmentation_path):
        """
        Loads image reader for a whole-brain dataset, then stores it in the
        "img_readers" attribute.

        Parameters
        ----------
        brain_id : str
            Unique identifier for a whole-brain dataset.
        img_path : str
            Path to whole-brain image.
        segmentation_path : str
            Path to segmentation of whole-brain image.
        """
        self.img_readers[brain_id] = img_util.TensorStoreReader(img_path)
        self.segmentation_readers[brain_id] = img_util.TensorStoreReader(
            segmentation_path
        )

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
        new_dataset.remove_isolated_sites()
        return new_dataset

    def remove_isolated_sites(self):
        """
        Removes merge sites whose closest fragment is greater than a specified
        distance.
        """
        # Find non-isolated sites
        idxs = list()
        for i in range(len(self.merge_sites_df)):
            brain_id = self.merge_sites_df["brain_id"][i]
            xyz = self.merge_sites_df["xyz"][i]
            if brain_id in self.graphs:
                d, _ = self.graphs[brain_id].kdtree.query(xyz)
                if d < 10:
                    idxs.append(i)

        # Drop isolated sites
        self.merge_sites_df = self.merge_sites_df.iloc[idxs]
        self.merge_sites_df = self.merge_sites_df.reset_index(drop=True)

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
        visited = set()
        for i in [i for i in self.merge_sites_df.index if i not in idxs]:
            # Extract site info
            brain_id = self.merge_sites_df["brain_id"][i]
            segment_id = self.merge_sites_df["segment_id"][i]
            pair = (brain_id, segment_id)

            # Find fragment containing site
            if pair not in visited:
                nodes = self.graphs[brain_id].get_nodes_with_segment_id(segment_id)
                self.graphs[brain_id].remove_nodes(nodes, False)
                visited.add(pair)

        self.remove_empty_graphs()

        # Relabel nodes
        for brain_id in self.graphs:
            self.graphs[brain_id].relabel_nodes()

        # Update merge sites df
        self.merge_sites_df = self.merge_sites_df.iloc[idxs]
        self.merge_sites_df = self.merge_sites_df.reset_index(drop=True)

    def remove_empty_graphs(self):
        """
        Removes graphs without any nodes.
        """
        for brain_id in list(self.graphs.keys()):
            if len(self.graphs[brain_id].nodes) == 0:
                del self.graphs[brain_id]

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
        voxel = subgraph.get_voxel(0)

        # Extract subgraph and image patches centered at site
        img_patch = self.get_img_patch(brain_id, voxel)
        segment_mask = self.get_segment_mask(brain_id, voxel, subgraph)

        # Stack image channels
        try:
            patches = np.stack([img_patch, segment_mask], axis=0)
        except ValueError:
            img_patch = img_util.pad_to_shape(img_patch, self.patch_shape)
            patches = np.stack([img_patch, segment_mask], axis=0)
        return patches, subgraph, label

    def sample_brain_id(self):
        """
        Samples a brain ID.

        Returns
        -------
        brain_id : str
            Unique identifier of a whole-brain dataset.
        """
        while True:
            brain_id = util.sample_once(list(self.graphs.keys()))
            if len(self.graphs[brain_id].nodes) > 0:
                return brain_id

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
        label : int
            Label of example.
        """
        # Get site info
        brain_id = self.merge_sites_df["brain_id"].iloc[idx]
        xyz = self.merge_sites_df["xyz"].iloc[idx]
        node = self.gt_graphs[brain_id].find_closest_node(xyz)

        # Extract rooted subgraph
        subgraph = self.gt_graphs[brain_id].get_rooted_subgraph(
            node, self.subgraph_radius
        )
        return brain_id, subgraph, 0

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
        label : int
            Label of example.
        """
        # Get site info
        brain_id = self.merge_sites_df["brain_id"].iloc[idx]
        xyz = self.merge_sites_df["xyz"].iloc[idx]
        node = self.graphs[brain_id].find_closest_node(xyz)

        # Extract rooted subgraph
        subgraph = self.graphs[brain_id].get_rooted_subgraph(
            node, self.subgraph_radius
        )
        return brain_id, subgraph, 1

    def get_random_negative_site(self):
        """
        Gets a random non-merge site from a fragment graph.

        Returns
        -------
        brain_id : str
            Unique identifier of the whole-brain dataset containing the site.
        node : int
            Node ID of the site.
        label : int
            Label of example.
        """
        # Sample graph
        brain_id = self.sample_brain_id()

        # Sample node on graph
        outcome = random.random()
        while True:
            # Sample node
            if outcome < 0.4:
                # Any node
                node = util.sample_once(list(self.graphs[brain_id].nodes))
            #elif outcome < 0.5:
            #    # Node close to soma
            #    node = self.sample_node_nearby_soma(brain_id)
            elif outcome < 0.8:
                # Branching node
                branching_nodes = self.graphs[brain_id].get_branchings()
                if len(branching_nodes) > 0:
                    node = util.sample_once(branching_nodes)
                else:
                    outcome = 0
                    continue
            else:
                # Branching node from GT
                branching_nodes = self.gt_graphs[brain_id].get_branchings()
                node = util.sample_once(branching_nodes)
                subgraph = self.gt_graphs[brain_id].get_rooted_subgraph(
                    node, self.subgraph_radius
                )
                return brain_id, subgraph, 0

            # Extract rooted subgraph
            subgraph = self.graphs[brain_id].get_rooted_subgraph(
                node, self.subgraph_radius
            )

            # Check branching
            if self.graphs[brain_id].degree(node) > 2:
                is_high_degree = self.graphs[brain_id].degree(node) > 3
                is_too_branchy = self.check_nearby_branching(brain_id, node)
                if is_high_degree or is_too_branchy:
                    continue

            # Check if node is close to merge site
            if not self.is_nearby_merge_site(brain_id, node):
                return brain_id, subgraph, 0

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
            maximum value of "self.brightness_clip".
        """
        img_patch = self.img_readers[brain_id].read(center, self.patch_shape)
        img_patch = np.minimum(img_patch, self.brightness_clip)
        return img_util.normalize(img_patch)

    def get_segment_mask(self, brain_id, center, subgraph):
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
        # Read segmentation
        if self.use_new_mask:
            segment_mask = self.segmentation_readers[brain_id].read(
                center, self.patch_shape
            )
            segment_mask = img_util.remove_small_segments(segment_mask, 1000)
            segment_mask = 0.5 * (segment_mask > 0).astype(float)
        else:
            segment_mask = np.zeros(self.patch_shape)

        # Annotate fragment
        center = subgraph.get_voxel(0)
        offset = img_util.get_offset(center, self.patch_shape)
        for node1, node2 in subgraph.edges:
            # Get local voxel coordinates
            voxel1 = subgraph.get_local_voxel(node1, offset)
            voxel2 = subgraph.get_local_voxel(node2, offset)

            # Populate mask
            voxels = geometry_util.make_digital_line(voxel1, voxel2)
            img_util.annotate_voxels(segment_mask, voxels)
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

    def check_nearby_branching(
        self, brain_id, root, max_depth=60, use_gt=False
    ):
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
        use_gt : bool
            Indication of whether to check groundtruth graph. Default is
            False.

        Returns
        -------
        bool
            Indication of whether there is a nearby branching node.
        """
        graph = self.gt_graphs[brain_id] if use_gt else self.graphs[brain_id]
        queue = [(root, 0)]
        visited = set([root])
        while queue:
            # Visit node
            i, d_i = queue.pop()
            if graph.degree[i] > 2 and d_i > 0:
                return True

            # Update queue
            for j in graph.neighbors(i):
                d_j = d_i + graph.dist(i, j)
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
        # Compute projection distances
        assert brain_id in self.gt_graphs, "Must load GT before fragments!"
        d_gt, _ = self.gt_graphs[brain_id].kdtree.query(graph.node_xyz)
        nodes = np.where(d_gt > 100)[0]
        graph.remove_nodes(nodes)

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

    def is_nearby_merge_site(self, brain_id, node):
        """
        Checks whether to the given node is close to a merge site.

        Parameters
        ----------
        brain_id : str
            Unique identifier for graph to be searched.
        node : int
            Node ID to check if it's close to a merge site.
        """
        xyz = self.graphs[brain_id].node_xyz[node]
        dist, _ = self.merge_site_kdtrees[brain_id].query(xyz)
        return dist < 100

    def sample_node_nearby_soma(self, brain_id):
        subgraph = self.gt_graphs[brain_id].get_rooted_subgraph(0, 600)
        gt_node = util.sample_once(subgraph.nodes)
        gt_xyz = self.gt_graphs[brain_id].node_xyz[gt_node]
        d, node = self.graphs[brain_id].kdtree.query(gt_xyz)
        return node


class MergeSiteTrainDataset(MergeSiteDataset):
    """
    A class for storing and retrieving training examples.
    """

    def __init__(self, base_dataset=None, idxs=None, negative_bias=0):
        """
        Instantiates a MergeSiteTrainDataset object.

        Parameters
        ----------
        base_dataset : MergeSiteDataset, optional
            Dataset to be instantiated as a train dataset.
        idxs : List[int], optional
            Indices of examples to be kept in train dataset.
        negative_bias : float, optional
            Specifies percentage of additional negative examples to add.
        """
        # Create sub-dataset
        subset_dataset = base_dataset.subset(self.__class__, idxs)
        self.__dict__.update(subset_dataset.__dict__)

        # Instance attributes
        self.negative_bias = negative_bias
        self.transform = ImageTransforms()

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
        patches, subgraph, label = super().__getitem__(idx)
        self.transform(patches)
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
            return self.get_indexed_positive_site(idx)
        elif np.random.random() < self.random_negative_example_prob:
            return self.get_random_negative_site()
        elif abs(idx) < len(self):
            return self.get_indexed_negative_site(abs(idx))
        else:
            return self.get_random_negative_site()

    # --- Helpers ---
    def get_idxs(self):
        """
        Gets example indices to iterate over.

        Returns
        -------
        numpy.ndarray
            Example indices to iterate over.
        """
        n_negative_examples = int(len(self) * (1 + self.negative_bias))
        return np.arange(-n_negative_examples + 1, len(self))


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
        self.examples = self.generate_examples()
        self.examples_summary = self.set_examples_summary()

    def generate_examples(self):
        # Generate negative examples
        negative_examples = self.generate_negative_examples()

        # Generate positive examples
        positive_examples = list()
        for i in range(len(self.merge_sites_df)):
            brain_id, subgraph, _ = self.get_indexed_positive_site(i)
            positive_examples.append(
                {
                    "brain_id": brain_id,
                    "subgraph": subgraph,
                    "xyz": subgraph.node_xyz[0],
                    "label": 1,
                }
            )
        return positive_examples + negative_examples

    def generate_negative_examples(self):
        """
        Generates examples of non-merge sites by sampling points on fragments
        that are sufficiently far from a merge site.

        Returns
        -------
        negative_examples : List[dict]
            List of negative examples collected across all graphs.
        """
        # Subroutines
        def add_examples():
            """
            Adds the given example to the set of validation examples.
            """
            for node in random.sample(nodes, n_examples):
                # Check if close to merge site
                if not self.is_nearby_merge_site(brain_id, node):
                    subgraph = graph.get_rooted_subgraph(
                        node, self.subgraph_radius
                    )
                    negative_examples.append(
                        {
                            "brain_id": brain_id,
                            "subgraph": subgraph,
                            "xyz": subgraph.node_xyz[0],
                            "label": 0,
                        }
                    )

        # Add branching nodes
        negative_examples = list()
        for brain_id, graph in self.graphs.items():
            # Filter branching nodes near other branching nodes
            nodes = list()
            for u in graph.get_branchings():
                is_branchy = self.check_nearby_branching(brain_id, u)
                if not is_branchy and graph.degree[u] == 3:
                    nodes.append(u)

            # Add nodes to examples
            n_examples = min(len(nodes), 80)
            add_examples()

        # Add non-branching points
        for brain_id, graph in self.graphs.items():
            nodes = [u for u in graph.nodes if graph.degree[u] < 3]
            n_examples = min(len(nodes), 40)
            add_examples()
        return negative_examples

    def set_examples_summary(self):
        """
        Sets a summary of examples in the validation dataset.

        Returns
        -------
        List[dict]
            List containing example metadata stored in a dictionary.
        """
        summary = list()
        for example in self.examples:
            summary.append(
                {
                    "brain_id": example["brain_id"],
                    "xyz": example["xyz"],
                    "label": example["label"],
                }
            )
        return summary

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
        brain_id = self.examples[idx]["brain_id"]
        subgraph = self.examples[idx]["subgraph"]
        label = self.examples[idx]["label"]
        return brain_id, subgraph, label

    def get_indexed_negative_site(self, idx):
        """
        Gets the negative example corresponding to the given index.

        Parameters
        ----------
        idx : int
            Index of example.

        Returns
        -------
        brain_id : str
            Unique identifier for the whole-brain dataset containing the site.
        node : int
            Node ID of the site.
        label : int
            Label of example.
        """
        brain_id = self.negative_examples[idx]["brain_id"]
        subgraph = self.negative_examples[idx]["subgraph"]
        return brain_id, subgraph, 0

    # --- Helpers ---
    def __len__(self):
        """
        Gets the number of examples in the dataset.
        """
        return len(self.examples)

    def get_idxs(self):
        """
        Gets example indices to iterate over.

        Returns
        -------
        numpy.ndarray
            Example indices to iterate over.
        """
        return np.arange(len(self.examples))

    def save_summary(self, output_dir):
        """
        Saves the example summary as a CSV file.

        Parameters
        ----------
        output_dir : str
            Path to directory that summary file is saved to.
        """
        df = pd.DataFrame(self.examples_summary)
        df.to_csv(os.path.join(output_dir, "val_summary.csv"))


# --- DataLoaders ---
class MergeSiteDataLoader(DataLoader):
    """
    A custom DataLoader class that uses multithreading to read image patches
    from the cloud to form batches.
    """

    def __init__(
        self,
        dataset,
        batch_size=32,
        is_multimodal=False,
        modality=None,
        sampler=None,
        use_shuffle=True
    ):
        """
        Instantiates a MergeSiteDataLoader object.

        Parameters
        ----------
        dataset : MergeSiteDataset
            Dataset to be iterated over to train or validate.
        batch_size : int, optional
            Number of examples in each batch. Default is 32.
        is_multimodal : bool, optional
            Indication of whether the loaded data is multimodal. Default is
            False.
        use_shuffle : bool, optional
            Indication of whether to shuffle examples. Default is True.
        """
        # Call parent class
        super().__init__(dataset, batch_size=batch_size, sampler=sampler)
        assert modality in [None, "graph", "pointcloud"]

        # Instance attributes
        self.is_multimodal = is_multimodal
        self.modality = modality
        self.patches_shape = (2,) + self.dataset.patch_shape
        self.use_shuffle = use_shuffle

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
        idxs = self.dataset.get_idxs()
        if self.use_shuffle:
            random.shuffle(idxs)

        # Iterate over indices
        for start in range(0, len(idxs), self.batch_size):
            end = min(start + self.batch_size, len(idxs))
            if self.is_multimodal and self.modality == "graph":
                yield self._load_image_graph_batch(idxs[start: end])
            elif self.is_multimodal and self.modality == "pointcloud":
                yield self._load_image_pc_batch(idxs[start: end])
            else:
                yield self._load_image_batch(idxs[start: end])

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
