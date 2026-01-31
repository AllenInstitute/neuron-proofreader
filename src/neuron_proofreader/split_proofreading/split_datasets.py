"""
Created on Fri April 11 11:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Routines for training machine learning models that classify proposals.

"""

from torch.utils.data import IterableDataset

import os
import pandas as pd

from neuron_proofreader.proposal_graph import ProposalGraph
from neuron_proofreader.machine_learning.augmentation import ImageTransforms
from neuron_proofreader.machine_learning.subgraph_sampler import (
    SeededSubgraphSampler,
    SubgraphSampler
)
from neuron_proofreader.split_proofreading.split_feature_extraction import (
    FeaturePipeline,
    HeteroGraphData
)
from neuron_proofreader.utils import geometry_util, img_util, util


# --- Single Brain Dataset ---
class FragmentsDataset(IterableDataset):

    def __init__(
        self,
        fragments_path,
        img_path,
        config,
        gt_path=None,
        metadata_path=None,
        segmentation_path=None,
        soma_centroids=None
    ):
        """
        Instantiates a FragmentsDataset object.

        Parameters
        ----------
        fragments_path : str
            Path to predicted SWC files to be loaded.
        img_path : str
            Path to the raw image associated with the fragments.
        config : Config
            ...
        gt_pointer : str, optional
            Path to ground-truth SWC files to be loaded. Default is None.
        metadata_path : str
            ...
        segmentation_path : str
            Path to the segmentation that fragments were generated from.
            Default is None.
        """
        # Instance attributes
        self.config = config
        self.transform = ImageTransforms() if config.ml.transform else False
        self.graph = self._load_graph(
            fragments_path, metadata_path, segmentation_path, soma_centroids
        )

        # Feature extractor
        self.feature_extractor = FeaturePipeline(
            self.graph,
            img_path,
            brightness_clip=self.config.ml.brightness_clip,
            patch_shape=self.config.ml.patch_shape,
            segmentation_path=segmentation_path
        )

    def _load_graph(
        self, fragments_path, metadata_path, segmentation_path, soma_centroids
    ):
        """
        Loads a graph by reading and processing SWC files specified by the
        given path.

        Parameters
        ----------
        fragments_path : str
            Path to SWC files to be loaded.
        metadata_path : str
            Patch to JSON file containing metadata on block that fragments
            were extracted from.
        soma_centroids : List[Tuple[float]]
            List of physical coordinates that represent soma centers.

        Returns
        -------
        graph : ProposalGraph
            Graph constructed from SWC files.
        """
        # Build graph
        graph = ProposalGraph(
            anisotropy=self.config.graph.anisotropy,
            min_size=self.config.graph.min_size,
            min_size_with_proposals=self.config.graph.min_size_with_proposals,
            node_spacing=self.config.graph.node_spacing,
            prune_depth=self.config.graph.prune_depth,
            remove_high_risk_merges=self.config.graph.remove_high_risk_merges,
            segmentation_path=segmentation_path,
            soma_centroids=soma_centroids
        )
        graph.load(fragments_path)

        # Post process fragments
        if metadata_path:
            self.clip_fragments(graph, metadata_path)

        if self.config.graph.remove_doubles:
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
        for subgraph in self.get_sampler():
            yield self.get_inputs(subgraph)

    def get_inputs(self, subgraph):
        features = self.feature_extractor(subgraph)
        data = HeteroGraphData(features)
        if self.graph.gt_path:
            return data.get_inputs(), data.get_targets()
        else:
            return data.get_inputs()

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

    def get_sampler(self):
        batch_size = self.config.ml.batch_size
        if len(self.graph.soma_ids) > 0:
            sampler = SeededSubgraphSampler(
                self.graph, max_proposals=batch_size
            )
        else:
            sampler = SubgraphSampler(self.graph, max_proposals=batch_size)
        return iter(sampler)


# --- Multi-Brain Dataset ---
class MultiBrainFragmentsDataset:
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
    def __init__(self, shuffle=True):
        # Instance attributes
        self.datasets = dict()
        self.shuffle = shuffle

    def add_dataset(self, key, dataset):
        self.datasets[key] = dataset

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
        samplers = self.init_samplers()
        while len(samplers) > 0:
            key = self.get_next_key(samplers)
            try:
                subgraph = next(samplers[key])
                yield self.datasets.get_inputs(subgraph)
            except StopIteration:
                del samplers[key]

    # --- Helpers ---
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

    def init_samplers(self):
        samplers = dict()
        for key, dataset in self.datasets.items():
            batch_size = dataset.config.ml.batch_size
            samplers[key] = iter(
                SubgraphSampler(dataset.graph, max_proposals=batch_size)
            )
        return samplers

    def n_proposals(self):
        """
        Counts the number of proposals in the dataset.

        Returns
        -------
        int
            Number of proposals.
        """
        cnt = 0
        for dataset in self.datasets.values():
            cnt += dataset.graph.n_proposals()
        return cnt

    def p_accepts(self):
        """
        Computes the percentage of accepted proposals in ground truth.

        Returns
        -------
        float
            Percentage of accepted proposals in ground truth.
        """
        accepts_cnt = 0
        for dataset in self.datasets.values():
            accepts_cnt += len(dataset.graph.gt_accepts)
        return accepts_cnt / self.n_proposals()

    def save_examples_summary(self, path):
        """
        Saves a summary of examples in the dataset to the given path.

        Parameters
        ----------
        path : str
            Output path for the CSV file.
        """
        examples_summary = list()
        for key in sorted(self.datasets.keys()):
            n_proposals = self.datasets[key].graph.n_proposals()
            examples_summary.extend([key] * n_proposals)
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
