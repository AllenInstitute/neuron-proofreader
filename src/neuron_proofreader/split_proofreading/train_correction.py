"""
Created on Sat April 12 11:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Routines for training machine learning models that classify proposals.


To do: explain how the train pipeline is organized. how is the data organized?


REMINDER: YOU NEED TO ADD PROPOSAL EDGES TO COMPUTATION GRAPH
"""

from collections import defaultdict
from concurrent.futures import as_completed, ProcessPoolExecutor
from copy import deepcopy
from torch.utils.data import IterableDataset

import numpy as np
import os
import random

from neuron_proofreader.proposal_graph import ProposalGraph
from neuron_proofreader.machine_learning.augmentation import ImageTransforms
from neuron_proofreader.split_proofreading import datasets
from neuron_proofreader.split_proofreading.feature_extraction import (
    FeaturePipeline
)
from neuron_proofreader.utils import ml_util, util


class GraphDataset(IterableDataset):
    """
    Dataset for storing graphs used to train a graph neural network to perform
    split correction. Graphs are stored in the "self.graphs" attribute, which
    is a dictionary containing the followin items:
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

    def __init__(self, config, patch_shape=(128, 128, 128), transform=False):
        # Instance attributes
        self.features = dict()
        self.graphs = dict()
        self.keys = set()
        self.patch_shape = patch_shape

        # Configs
        self.graph_config = config.graph_config
        self.ml_config = config.ml_config

        # Data augmentation (if applicable)
        self.transform = ImageTransforms() if transform else False

    # --- Dataset Properties ---
    def __len__(self):
        """
        Counts the number of graphs in the dataset.

        Returns
        -------
        int
            Number of graphs.
        """
        return len(self.graphs)

    def n_proposals(self):
        """
        Counts the number of proposals in the dataset.

        Returns
        -------
        int
            Number of proposals.
        """
        return np.sum([graph.n_proposals() for graph in self.graphs.values()])

    def n_accepts(self):
        """
        Counts the number of ground truth accepted proposals in the dataset.

        Returns
        -------
        int
            Number of ground truth accepted proposals in the dataset.
        """
        cnts = [len(graph.gt_accepts) for graph in self.graphs.values()]
        return np.sum(cnts)

    def p_accepts(self):
        """
        Computes the percentage of accepted proposals in ground truth.

        Returns
        -------
        float
            Percentage of accepted proposals in ground truth.
        """
        return self.n_accepts() / self.n_proposals()

    # --- Load Data ---
    def add_graph(
        self,
        key,
        gt_pointer,
        pred_pointer,
        img_path,
    ):
        # Add graph
        self.graphs[key] = self.load_graph(pred_pointer)
        self.graphs[key].generate_proposals(
            self.graph_config.search_radius,
            complex_bool=self.graph_config.complex_bool,
            groundtruth_graph=self.load_graph(gt_pointer),
            long_range_bool=self.graph_config.long_range_bool,
            proposals_per_leaf=self.graph_config.proposals_per_leaf,
        )
        self.keys.add(key)

        # Generate features -- add segmentation path
        feature_pipeline = FeaturePipeline(
            img_path,
            self.graph_config.search_radius,
            patch_shape=self.patch_shape
        )
        self.features[key] = feature_pipeline.run(self.graphs[key])

    def load_graph(self, swc_pointer):
        """
        Loads a graph by reading and processing SWC files specified by
        "swc_pointer".

        Parameters
        ----------
        swc_pointer : str
            Path to SWC files to be loaded.

        Returns
        -------
        graph : ProposalGraph
            Graph constructed from SWC files.
        """
        graph = ProposalGraph(
            min_size=self.graph_config.min_size,
            node_spacing=self.graph_config.node_spacing,
        )
        graph.load(swc_pointer)
        return graph

    # --- Get Data ---
    def __iter__(self):
        for key, graph in self.graphs.items():
            pass

    def __getitem__(self, key):
        features = deepcopy(self.features[key])
        if self.transform:
            with ProcessPoolExecutor() as executor:
                # Assign processes
                pending = dict()
                for proposal, patches in features.proposal_patches.items():
                    process = executor.submit(self.transform, patches)
                    pending[process] = proposal

                # Store results
                for process in as_completed(pending.keys()):
                    proposal = pending.pop(process)
                    features.proposal_patches = process.result()
        return self.graphs[key], features


# --- Custom Dataloader ---
class GraphDataLoader:

    def __init__(self, graph_dataset, batch_size=32, shuffle=True):
        # Instance attributes
        self.batch_size = batch_size
        self.graph_dataset = graph_dataset
        self.shuffle = shuffle

    def __iter__(self):
        """
        Generates a list of batches for training a graph neural network. Each
        batch is a tuple that contains the following:
            - key (str): Unique identifier of a graph in self.graphs.
            - graph (networkx.Graph): GNN computation graph.
            - proposals (List[frozenset[int]]): List of proposals in graph.

        Parameters
        ----------
        batch_size : int
            Maximum number of proposals in each batch.

        Returns
        -------
        ...
        """
        # Initializations
        if self.shuffle:
            keys = list(self.graph_dataset.keys)
            random.shuffle(keys)

        # Main
        for key in keys:
            graph, features = self.graph_dataset[key]
            proposals = set(graph.list_proposals())
            while len(proposals) > 0:
                # Get batch
                batch = ml_util.get_batch(graph, proposals, self.batch_size)
                proposals -= batch["proposals"]

                # Extract features
                accepts = graph.gt_accepts
                batch_features = self.get_batch_features(batch, features)
                data = datasets.init(batch_features, batch["graph"], accepts)
                yield data

    def get_batch_features(self, batch, features):
        # Node features
        batch_features = defaultdict(lambda: defaultdict(dict))
        for i in batch["graph"].nodes:
            batch_features["nodes"][i] = features["nodes"][i]

        # Edge features
        for e in map(frozenset, batch["graph"].edges):
            if e in batch["proposals"]:
                batch_features["proposals"][e] = features["proposals"][e]
            else:
                batch_features["branches"][e] = features["branches"][e]

        # Image patches
        if "patches" in features:
            for p in batch["proposals"]:
                batch_features["patches"][p] = features["patches"][p]
        return batch_features


# -- Helpers --
def generate_dataset_example_ids(bucket_name, dataset_prefix):
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
            block_prefixes = util.list_gcs_subdirectories(bucket_name, brain_segmentation_prefix)
            for block_prefix in block_prefixes:
                # Extract block id
                block_id = block_prefix.split("/")[-2]
                yield brain_id, segmentation_id, block_id


def truncate(hat_y, y):
    """
    Truncates "hat_y" so that this tensor has the same shape as "y". Note this
    operation removes the predictions corresponding to branches so that loss
    is computed over proposals.

    Parameters
    ----------
    hat_y : torch.Tensor
        Tensor to be truncated.
    y : torch.Tensor
        Tensor used as a reference.

    Returns
    -------
    torch.Tensor
        Truncated "hat_y".
    """
    return hat_y[: y.size(0), 0]
