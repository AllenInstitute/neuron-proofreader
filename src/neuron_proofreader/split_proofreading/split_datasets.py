"""
Created on Fri April 11 11:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Routines for training machine learning models that classify proposals.

"""

from torch.utils.data import IterableDataset

import numpy as np
import os
import pandas as pd

from neuron_proofreader.proposal_graph import ProposalGraph
from neuron_proofreader.machine_learning.augmentation import ImageTransforms
from neuron_proofreader.machine_learning.subgraph_sampler import (
    SubgraphSampler
)
from neuron_proofreader.split_proofreading.split_feature_extraction import (
    FeaturePipeline,
    HeteroGraphData
)
from neuron_proofreader.utils import geometry_util, img_util, util


class FragmentsDataset(IterableDataset):
    """
    A dataset class for storing graphs used to train models to perform split
    correction. Graphs are stored in the "self.graphs" attribute, which is a
    dictionary containing the followin items:
        - Key: (brain_id, segmentation_id, example_id)
        - Value: graph that is an instance of ProposalGraph

    This dataset is populated using the "self.add_graph" method, which
    requires the following inputs:
        (1) key: Unique identifier of graph.
        (2) gt_pointer: Path to ground truth SWC files.
        (2) pred_pointer: Path to predicted SWC files.
        (3) img_path: Path to whole-brain image stored in cloud bucket.

    Note: This dataset supports graphs from multiple whole-brain datasets.
    """

    def __init__(
        self,
        config,
        brightness_clip=400,
        patch_shape=(128, 128, 128),
        shuffle=True,
        transform=False
    ):
        """
        Instantiates a FragmentsDataset object.

        Parameters
        ----------
        config : GraphConfig
            Config object that stores parameters used to build graphs.
        patch_shape : Tuple[int], optional
            Shape of image patch input to a vision model. Default is (128, 128,
            128).
        transform : bool, optional
            Indication of whether to apply augmentation to input images.
            Default is False.
        """
        # Instance attributes
        self.brightness_clip = brightness_clip
        self.config = config
        self.feature_extractors = dict()
        self.graphs = dict()
        self.patch_shape = patch_shape
        self.shuffle = shuffle
        self.transform = ImageTransforms() if transform else False

    # --- Load Data ---
    def add_graph(
        self,
        key,
        gt_pointer,
        pred_pointer,
        img_path,
        metadata_path=None,
        segmentation_path=None
    ):
        """
        Loads a fragments graph, generates proposals, and initializes feature
        extraction.

        Parameters
        ----------
        key : Tuple[str]
            Unique identifier used to register the graph and its feature
            pipeline.
        gt_pointer : str
            Path to ground-truth SWC files to be loaded.
        pred_pointer : str
            Path to predicted SWC files to be loaded.
        img_path : str
            Path to the raw image associated with the graph.
        metadata_path : str
            ...
        segmentation_path : str
            Path to the segmentation mask associated with the graph.
        """
        # Add graph
        gt_graph = self.load_graph(gt_pointer, is_gt=True)
        self.graphs[key] = self.load_graph(pred_pointer, metadata_path)
        self.graphs[key].generate_proposals(
            self.config.search_radius, gt_graph=gt_graph
        )

        # Generate features
        self.feature_extractors[key] = FeaturePipeline(
            self.graphs[key],
            img_path,
            self.config.search_radius,
            brightness_clip=self.brightness_clip,
            patch_shape=self.patch_shape,
            segmentation_path=segmentation_path
        )

    def load_graph(self, swc_pointer, is_gt=False, metadata_path=None):
        """
        Loads a graph by reading and processing SWC files specified by
        "swc_pointer".

        Parameters
        ----------
        swc_pointer : str
            Path to SWC files to be loaded.
        metadata_path : str
            Patch to JSON file containing metadata on block that fragments
            were extracted from.

        Returns
        -------
        graph : ProposalGraph
            Graph constructed from SWC files.
        """
        # Build graph
        graph = ProposalGraph(
            anisotropy=self.config.anisotropy,
            min_size=self.config.min_size
        )
        graph.load(swc_pointer)

        # Post process fragments
        if not is_gt:
            self.clip_fragments(graph, metadata_path)
            geometry_util.remove_doubles(graph, 200)
        return graph

    # --- Get Data ---
    def __iter__(self):
        """
        Iterates over the dataset and yields model-ready inputs and targets.

        Yields
        ------
        inputs : HeteroGraphData
            Heterogeneous graph data.
        targets : torch.Tensor
            Ground truth labels.
        """
        # Initialize subgraph samplers
        samplers = dict()
        for key, graph in self.graphs.items():
            samplers[key] = iter(SubgraphSampler(graph, max_proposals=32))

        # Iterate over dataset
        while len(samplers) > 0:
            key = self.get_next_key(samplers)
            try:
                # Feature extraction
                subgraph = next(samplers[key])
                features = self.feature_extractors[key](subgraph)

                # Get model inputs
                data = HeteroGraphData(features)
                inputs = data.get_inputs()
                targets = data.get_targets()
                yield inputs, targets
            except StopIteration:
                del samplers[key]

    def get_next_key(self, samplers):
        """
        Gets the next key to sample from a dictionary of samplers.

        Parameters
        ----------
        samplers : Dict[Tuple[str], SubgraphSampler]
            Mapping from keys to sampler objects.

        Returns
        -------
        key : Tuple[str]
            Selected key from "samplers".
        """
        if self.shuffle:
            return util.sample_once(samplers.keys())
        else:
            keys = sorted(samplers.keys())
            return keys[0]

    # --- Helpers ---
    @staticmethod
    def clip_fragments(graph, metadata_path):
        # Extract bounding box
        bucket_name, path = util.parse_cloud_path(metadata_path)
        metadata = util.read_json_from_gcs(bucket_name, path)
        origin = metadata["chunk_origin"][::-1]
        shape = metadata["chunk_shape"][::-1]

        # Clip graph
        nodes = list()
        for i in graph.nodes:
            voxel = graph.get_voxel(i)
            if not img_util.is_contained(voxel - origin, shape):
                nodes.append(i)
        graph.remove_nodes_from(nodes)
        graph.relabel_nodes()

    def n_proposals(self):
        """
        Counts the number of proposals in the dataset.

        Returns
        -------
        int
            Number of proposals.
        """
        return np.sum([graph.n_proposals() for graph in self.graphs.values()])

    def p_accepts(self):
        """
        Computes the percentage of accepted proposals in ground truth.

        Returns
        -------
        float
            Percentage of accepted proposals in ground truth.
        """
        cnts = [len(graph.gt_accepts) for graph in self.graphs.values()]
        return np.sum(cnts) / self.n_proposals()

    def save_examples_summary(self, path):
        """
        Saves a summary of examples in the dataset to the given path.

        Parameters
        ----------
        path : str
            Output path for the CSV file.
        """
        examples_summary = list()
        for key in sorted(self.graphs.keys()):
            examples_summary.extend([key] * self.graphs[key].n_proposals())
        pd.DataFrame(examples_summary).to_csv(path)


# -- Helpers --
def generate_dataset_example_ids(bucket_name, dataset_prefix):
    """
    Generates dataset example identifiers.

    Parameters
    ----------
    bucket_name : str
        Name of the Google Cloud Storage bucket.
    dataset_prefix : str
        Root prefix under which dataset contents are organized.

    Yields
    -------
    Tuple[str]
        Dataset example ID formatted as (brain_id, segmentation_id, block_id).
    """
    brain_prefixes = util.list_gcs_subdirectories(bucket_name, dataset_prefix)
    for brain_prefix in brain_prefixes:
        # Extract brain id
        brain_id = brain_prefix.split("/")[-2]

        # Iterate over segmentations
        pred_prefix = os.path.join(brain_prefix, "pred_swcs/")
        prefixes = util.list_gcs_subdirectories(bucket_name, pred_prefix)
        for brain_segmentation_prefix in prefixes:
            # Extract segmentation id
            segmentation_id = brain_segmentation_prefix.split("/")[-2]

            # Iterate over blocks
            block_prefixes = util.list_gcs_subdirectories(
                bucket_name, brain_segmentation_prefix
            )
            for block_prefix in block_prefixes:
                # Extract block id
                block_id = block_prefix.split("/")[-2]
                yield brain_id, segmentation_id, block_id
