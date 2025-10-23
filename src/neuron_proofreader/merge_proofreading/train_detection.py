"""
Created on Wed July 2 11:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Dataset and dataloader utilities for processing merge site data to train
model to detect merge errors.

"""

from concurrent.futures import as_completed, ThreadPoolExecutor
from scipy.spatial import KDTree
from torch.utils.data import Dataset, DataLoader

import networkx as nx
import numpy as np
import random

from neuron_proofreader.machine_learning.augmentation import ImageTransforms
from neuron_proofreader.skeleton_graph import SkeletonGraph
from neuron_proofreader.utils import img_util, ml_util, swc_util, util


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
    context_radius : int, optional
        Radius (in microns) around merge sites used to extract rooted
        subgraph.
    gt_graphs : dict[str, SkeletonGraph]
        Dictionary that maps brain IDs to a graph containing ground truth
        tracings.
    gt_kdtrees : scipy.spatial.KDTree
        KD-Tree built from xyz coordinates stored in "gt_graphs".
    img_readers : ImageReader
        Image reader used to read raw image from cloud bucket.
    merge_graphs : dict[str, SkeletonGraph]
        Dictionary that maps brain IDs to a graph containing fragments that
        contain merge mistakes.
    merge_sites_df : pandas.DataFrame
        DataFrame containing merge sites, must contain the columns:
        "brain_id", "segmentation_id", "segment_id", and "xyz".
    multiscale : int, optional
        Level in the image pyramid that the voxel coordinate must index into.
    node_spacing : int, optional
        Spacing between nodes in the graph.
    patch_shape : tuple of int, optional
        Shape of the 3D image patches to extract.
    """

    def __init__(
        self,
        merge_sites_df,
        anisotropy=(1.0, 1.0, 1.0),
        context_radius=200,
        multiscale=0,
        node_spacing=5,
        patch_shape=(96, 96, 96),
    ):
        """
        Instantiates MergeSiteDataset object.

        Parameters
        ----------
        merge_sites_df : pandas.DataFrame
            DataFrame containing merge sites, must contain the columns:
            "brain_id", "segmentation_id", "segment_id", and "xyz".
        anisotropy : Tuple[float], optional
            Image to physical coordinates scaling factors to account for the
            anisotropy of the microscope.
        context_radius : int, optional
            Radius (in microns) around merge sites used to extract rooted
            subgraph. Default is 200um.
        multiscale : int, optional
            Level in the image pyramid that the voxel coordinate must index
            into. Default is 0.
        node_spacing : int, optional
            Spacing between nodes in the graph. Default is 5um.
        patch_shape : Tuple[int], optional
            Shape of the 3D patches to extract (depth, height, width). Default is
            (96, 96, 96).
        """
        # Instance attributes
        self.anisotropy = anisotropy
        self.context_radius = context_radius
        self.node_spacing = node_spacing
        self.merge_sites_df = merge_sites_df
        self.multiscale = multiscale
        self.patch_shape = patch_shape
        self.transform = ImageTransforms()

        # Data structures
        self.img_readers = dict()
        self.gt_graphs = dict()
        self.merge_graphs = dict()
        self.merge_site_kdtrees = dict()

    # --- Load Data ---
    def init_graph(self, swc_pointer):
        """
        Initialize a SkeletonGraph built from SWC files.

        Parameters
        ----------
        swc_pointer : any
            Pointer to SWC files to be loaded into graph.

        Returns
        -------
        SkeletonGraph
            Graph with loaded data from SWC files.
        """
        graph = SkeletonGraph(node_spacing=self.node_spacing)
        graph.load(swc_pointer)
        return graph

    def init_kdtrees(self):
        """
        Initialize KDTrees for both ground truth and merge graphs.
        """
        # Build kd-trees from node xyz coordinates
        self.gt_kdtrees = self._init_kdtree(self.gt_graphs)
        self.merge_kdtrees = self._init_kdtree(self.merge_graphs)

        # Build kd-tree from merge sites
        for brain_id in self.merge_sites_df["brain_id"].unique():
            pts = get_gt_merge_sites(self.merge_sites_df, brain_id)
            self.merge_site_kdtrees[brain_id] = KDTree(pts)

    def _init_kdtree(self, graphs):
        """
        Build KDTree for each graph contained in a dictionary.

        Parameters
        ----------
        graphs : Dict[str, SkeletonGraph]
            Dictionary mapping brain IDs to SkeletonGraph instances.

        Returns
        -------
        dict
            Dictionary mapping brain IDs to KDTrees built from the graphs.
        """
        for brain_id, graph in graphs.items():
            graph.init_kdtree()

    def load_merge_graphs(self, brain_id, swc_pointer):
        """
        Loads and processes fragments containing merge mistakes for a given
        brain sample, then loads them into "merge_graphs" attribute.

        Parameters
        ----------
        brain_id : str
            Unique identifier for the whole-brain dataset.
        swc_pointer : any
            Pointer to SWC files to be loaded into graph.
        """
        # Load graphs
        graph = self.init_graph(swc_pointer)

        # Filter non-merge components
        idxs = self.merge_sites_df["brain_id"] == brain_id
        merged_segment_ids = self.merge_sites_df["segment_id"][idxs].values
        for swc_id in graph.get_swc_ids():
            segment_id = swc_util.get_segment_id(swc_id)
            if str(segment_id) not in merged_segment_ids:
                component_id = util.find_key(graph.component_id_to_swc_id, swc_id)
                nodes = graph.get_nodes_with_component_id(component_id)
                graph.remove_nodes_from(nodes)
                graph.irreducible.remove_nodes_from(nodes)

        # Postprocess and store graph
        self.clip_fragments_to_groundtruth(brain_id, graph)
        graph.relabel_nodes()
        self.merge_graphs[brain_id] = graph

    def load_gt_graphs(self, brain_id, img_path, swc_pointer):
        """
        Loads and processes ground truth tracings and image for a given brain
        sample, then loads them into "gt_graphs" and "img_readers" attributes.

        Parameters
        ----------
        brain_id : str
            Unique identifier for the whole-brain dataset.
        swc_pointer : any
            Pointer to SWC files to be loaded into graph.
        """
        self.img_readers[brain_id] = img_util.init_reader(img_path)
        self.gt_graphs[brain_id] = self.init_graph(swc_pointer)

    # --- Get Examples ---
    def __getitem__(self, idx, use_transform):
        """
        Retrieves a training example consisting of an image patch, its
        corresponding label mask, and the associated subgraph for a given
        site.

        Parameters
        ----------
        idx : int or None
            Index of the site to retrieve. If None, a random site is sampled.
        use_transform : bool
            Indication of whether to apply data augmentation transforms to the
            image and label patches.

        Returns
        -------
        patches : numpy.ndarray
            Array of stacked channels containing the image patch and label
            mask with shape (2, D, H, W).
        subgraph : networkx.Graph
            Rooted subgraph centered at the site node.
        is_positive : int
            Binary label indicating whether the site is a positive example or
            negative.
        """
        # Get example
        brain_id, graph, node, is_positive = self.get_site(idx)
        xyz = graph.node_xyz[node]
        voxel = img_util.to_voxels(xyz, self.anisotropy, self.multiscale)

        # Extract subgraph and image patches centered at site
        subgraph = graph.get_rooted_subgraph(node, self.context_radius)
        img_patch = self.get_img_patch(brain_id, voxel)
        label_patch = self.get_label_mask(subgraph)

        # Stack image channels
        try:
            patches = np.stack([img_patch, label_patch], axis=0)
        except ValueError:
            img_patch = img_util.pad_to_shape(img_patch, self.patch_shape)
            patches = np.stack([img_patch, label_patch], axis=0)

        # Apply image augmentation (if applicable)
        if use_transform:
            _, patches = self.transform(idx, patches)
        return patches, subgraph, int(is_positive)

    def get_site(self, idx):
        if idx is None:
            return self.get_random_site()
        else:
            return self.get_indexed_site(idx)

    def get_indexed_site(self, idx):
        """
        Retrieves the site corresponding to the given index.

        Parameters
        ----------
        idx : int
            Index of the site in "merge_sites_df". Positive indices correspond
            to merge sites, while non-positive indices correspond to non-merge
            sites.

        Returns
        -------
        brain_id : str
            Unique identifier of the brain containing the site.
        graph : Graph
            Graph containing the site.
        node : int
            Node ID of the site.
        is_positive : bool
            True if the site is a merge site, False if it is a non-merge site.
        """
        # Extract graph
        brain_id = self.merge_sites_df["brain_id"].iloc[abs(idx)]
        is_positive = idx > 0
        if not is_positive:
            graph = self.gt_graphs[brain_id]
        elif self.has_fragment(idx):
            graph = self.merge_graphs[brain_id]
        else:
            return self.get_random_site()

        # Extract site info
        xyz = self.merge_sites_df["xyz"].iloc[abs(idx)]
        node = graph.query_node(xyz)
        return brain_id, graph, node, is_positive

    def get_random_site(self):
        """
        Randomly selects a non-merge site from the ground truth graphs.

        Returns
        -------
        brain_id : str
            Unique identifier of the brain containing the site.
        graph : SkeletonGraph
            Graph containing the site.
        node : int
            Node ID of the site.
        is_positive : bool
            False, since this method only samples from ground truth sites.
        """
        # Sample graph
        brain_id = util.sample_once(list(self.merge_graphs.keys()))
        graph = self.merge_graphs[brain_id]

        # Sample node on graph
        while True:
            # Sample node
            outcome = random.random()
            if outcome <= 0.3:
                node = util.sample_once(graph.nodes)
            elif outcome > 0.3 and outcome < 0.4:
                node = util.sample_once(graph.get_leafs())
            else:
                node = util.sample_once(graph.get_branchings())

            # Check if node is close to merge site
            xyz = graph.node_xyz[node]
            d, _ = self.merge_site_kdtrees[brain_id].query(xyz)
            if d > 100:
                break
        return brain_id, graph, node, False

    def get_img_patch(self, brain_id, center):
        img_patch = self.img_readers[brain_id].read(center, self.patch_shape)
        return img_util.normalize(np.minimum(img_patch, 300))

    def get_label_mask(self, subgraph):
        # Initializations
        label_mask = np.zeros(self.patch_shape)
        xyz = subgraph.node_xyz[0]
        center = img_util.to_voxels(xyz, self.anisotropy, self.multiscale)

        # Populate label mask
        for i in subgraph.nodes:
            xyz = subgraph.node_xyz[i]
            voxel = img_util.to_voxels(xyz, self.anisotropy, self.multiscale)
            voxel = shift_voxel(voxel, center, self.patch_shape)
            if img_util.is_contained(voxel, self.patch_shape, buffer=3):
                i, j, k = voxel
                label_mask[i-3:i+3, j-3:j+3, k-3:k+3] = 1
        return label_mask

    # --- Helpers ---
    def clip_fragments_to_groundtruth(self, brain_id, graph):
        """
        Removes any node from the given fragment that is more than 100μm from
        the ground truth graph.

        Parameters
        ----------
        brain_id : str
            Unique identifier of whole-brain dataset.
        graph : SkeletonGraph
            Fragment to be clipped.
        """
        assert brain_id in self.gt_graphs, "Must load GT before fragments!"
        kdtree = KDTree(self.gt_graphs[brain_id].node_xyz)
        for i, xyz in enumerate(graph.node_xyz):
            if i in graph:
                d, _ = kdtree.query(xyz)
                if d > 100:
                    graph.remove_node(i)

    def __len__(self):
        """
        Returns the number of positive and negative examples of merge sites.

        Returns
        -------
        int
            Number of positive and negative examples of merge sites.
        """
        return 2 * len(self.merge_sites_df)

    def count_fragments(self):
        """
        Counts the number of fragments in the dataset.

        Returns
        -------
        cnt : int
            Number of fragments in the dataset.
        """
        cnt = 0
        for graph in self.merge_graphs.values():
            cnt += nx.number_connected_components(graph)
        return cnt

    def has_fragment(self, idx):
        brain_id = self.merge_sites_df["brain_id"].iloc[idx]
        segment_id = self.merge_sites_df["segment_id"].iloc[idx]
        swc_id = f"{segment_id}.0"
        return swc_id in self.merge_graphs[brain_id].get_swc_ids()


class MergeSiteDataLoader(DataLoader):
    """
    DataLoader that uses multithreading to read image patches from the cloud
    to form batches.

    Attributes
    ----------
    batch_size : int
        Number of examples in each batch.
    dataset : MergeSiteDataset
        Dataset containing merge sites to be loaded during training.
    idxs : List[int]
        Example indexes to load from "dataset". Note: "dataset" contains both
        train and validation data.
    use_random_sites : bool
        Indication of whether to use random non-merge sites sampled from
        ground tracings during training.
    use_transform : bool
        Indication of whether to use data augmentation during training.
    """

    def __init__(
        self,
        dataset,
        idxs,
        batch_size=32,
        sampler=None,
        use_random_sites=False,
        use_transform=False
    ):
        """
        Instantiates MergeSiteDataloader object.

        Parameters
        ----------
        dataset : MergeSiteDataset
            Dataset containing merge sites to be loaded during training.
        idxs : List[int]
            Example indexes to load from "dataset". Note: "dataset" contains
            both train and validation data.
        batch_size : int, optional
            Number of examples in each batch. Default is 32.
        use_random_sites : bool, optional
            Indication of whether to use random non-merge sites sampled from
            ground tracings during training. Default is False.
        use_transform : bool, optional
            Indication of whether to use data augmentation during training.
            Default is False.
        """
        # Call parent class
        super().__init__(dataset, batch_size=batch_size, sampler=sampler)

        # Instance attributes
        self.idxs = self._load_idxs(idxs)
        self.use_random_sites = use_random_sites
        self.use_transform = use_transform

    def __iter__(self):
        """
        Generates batches of examples used during training and validation.

        Returns
        -------
        iterator
            Generates batch of examples used during training and validation.
        """
        random.shuffle(self.idxs)
        for i_start in range(0, len(self.idxs), self.batch_size):
            yield self._load_batch(i_start)

    def _load_batch(self, i_start):
        # Generate batch
        batch_size = min(self.batch_size, len(self.idxs) - i_start)
        with ThreadPoolExecutor() as executor:
            # Assign threads
            threads = list()
            for i_offset in range(batch_size):
                # Get idx
                idx = self.idxs[i_start + i_offset]
                if idx < 0 and self.use_random_sites:
                    idx = None if np.random.random() < 0.7 else idx

                # Submit job
                threads.append(
                    executor.submit(
                        self.dataset.__getitem__, idx, self.use_transform
                    )
                )

            # Process results
            patch_shape = (batch_size, 2,) + self.dataset.patch_shape
            patches = np.empty(patch_shape, dtype=np.float32)
            labels = np.empty((batch_size, 1), dtype=np.float32)
            for i, thread in enumerate(as_completed(threads)):
                patch, _, label = thread.result()
                patches[i] = patch
                labels[i] = label
        return ml_util.to_tensor(patches), ml_util.to_tensor(labels)

    def _load_idxs(self, idxs):
        """
        Initializes indexes that dataloader iterates over. Note: indexes
        are duplicated and one set is negated to account for negative
        examples.

        Parameters
        ----------
        idxs : List[int]
            Indexes of examples to be iterated over.

        Returns
        -------
        idxs : List[int]
            Indexes of examples to be iterated over.
        """
        negative_idxs = -np.arange(0, len(idxs))
        idxs.extend(negative_idxs.tolist())
        return idxs


# -- Helpers --
def get_brain_segmentation_pairs(merge_sites_df, idxs):
    brain_segmentation_pairs = set()
    for idx in idxs:
        brain_id = merge_sites_df["brain_id"][idx]
        segmentation_id = merge_sites_df["segmentation_id"][idx]
        brain_segmentation_pairs.add((brain_id, segmentation_id))
    return brain_segmentation_pairs


def get_gt_merge_sites(merge_sites_df, brain_id):
    """
    Gets the xyz coordinates of ground truth merge sites for a given brain.

    Parameters
    ----------
    merge_sites_df : pd.DataFrame
        DataFrame containing merge sites, must contain the columns:
        "brain_id", "segmentation_id", "segment_id", and "xyz".
    brain_id : str
        Unique identifier for a whole-brain dataset.

    Returns
    -------
    numpy.ndarray
        Ground-truth merge sites (xyz coordinates) for a given brain.
    """
    idx_mask = merge_sites_df["brain_id"] == brain_id
    return np.array(merge_sites_df.loc[idx_mask, "xyz"].tolist())


def shift_voxel(voxel, center, patch_shape):
    voxel = [v - c + s // 2 for v, c, s in zip(voxel, center, patch_shape)]
    return tuple(voxel)
