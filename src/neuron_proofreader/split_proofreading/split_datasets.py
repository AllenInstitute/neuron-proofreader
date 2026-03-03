"""
Created on Fri April 11 11:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Implementation of dataset objects that contain graphs and facilitate feature
generation for training and inference in split-correction tasks.

"""

from torch.utils.data import IterableDataset

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
from neuron_proofreader.utils import geometry_util, util


# --- Single Brain Dataset ---
class FragmentsDataset(IterableDataset):
    """
    A dataset object that contains a graph built from fragments corresponding
    to a single brain. Note that this dataset supports fragments extracted
    from either a block or whole-brain.
    """

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
            Configuration object containing parameters and settings.
        gt_path : str, optional
            Path to ground-truth SWC files to be loaded. Default is None.
        metadata_path : str, optional
            Patch to JSON file containing metadata on block that fragments
            were extracted from. Default is None.
        segmentation_path : str, optional
            Path to the segmentation that fragments were generated from.
            Default is None.
        soma_centroids : List[Tuple[int]], optional
            Phyiscal coordinates of soma centroids. Default is None.
        """
        # Instance attributes
        self.config = config
        self.gt_path = gt_path
        self.transform = ImageTransforms() if config.ml.transform else False

        # Build graph
        self.graph = self._load_graph(
            fragments_path,
            metadata_path=metadata_path,
            segmentation_path=segmentation_path,
            soma_centroids=soma_centroids
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
        self,
        fragments_path,
        metadata_path=None,
        segmentation_path=None,
        soma_centroids=None
    ):
        """
        Loads a graph by reading and processing SWC files specified by the
        given path.

        Parameters
        ----------
        fragments_path : str
            Path to SWC files to be loaded.
        metadata_path : str, optional
            Patch to JSON file containing metadata on block that fragments
            were extracted from. Default is None
        segmentation_path : str, optional
            Path to the segmentation that fragments were generated from.
            Default is None.
        soma_centroids : List[Tuple[float]], optional
            List of physical coordinates that represent soma centers. Default
            is None.

        Returns
        -------
        graph : ProposalGraph
            Graph constructed from SWC files.
        """
        # Build graph
        graph = ProposalGraph(
            anisotropy=self.config.graph.anisotropy,
            gt_path=self.gt_path,
            min_size=self.config.graph.min_size,
            min_size_with_proposals=self.config.graph.min_size_with_proposals,
            node_spacing=self.config.graph.node_spacing,
            prune_depth=self.config.graph.prune_depth,
            remove_high_risk_merges=self.config.graph.remove_high_risk_merges,
            segmentation_path=segmentation_path,
            soma_centroids=soma_centroids,
            verbose=self.config.graph.verbose
        )
        graph.load(fragments_path)

        # Post process fragments
        if metadata_path:
            graph.clip_to_bbox(metadata_path)

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
            Input data.
        targets : torch.Tensor
            Ground truth labels.
        """
        for subgraph in self.get_sampler():
            features = self.feature_extractor(subgraph)
            yield HeteroGraphData(features)

    # --- Helpers ---
    def get_sampler(self):
        """
        Gets a subgraph sampler that is used to iterate over dataset.

        Returns
        -------
        sampler : SubgraphSampler
            Subgraph sampler that is used to iterate over dataset.
        """
        batch_size = self.config.ml.batch_size
        return iter(SubgraphSampler(self.graph, max_proposals=batch_size))


# --- Multi-Brain Dataset ---
class FragmentsDatasetCollection(IterableDataset):
    """
    A dataset class for storing a set of FragmentDataset objects corresponding
    to different whole-brain images.
    """

    def __init__(self, shuffle=True):
        """
        Instantiates a FragmentsDatasetCollection object.

        Parameters
        ----------
        shuffle : bool, optional
            Indication of whether to shuffle examples. Default is True.
        """
        # Instance attributes
        self.datasets = dict()
        self.shuffle = shuffle

    def add_dataset(
        self,
        key,
        fragments_path,
        img_path,
        config,
        gt_path=None,
        metadata_path=None,
        segmentation_path=None,
        soma_centroids=None
    ):
        """
        Adds a dataset to the collection of datasets.

        Parameters
        ----------
        key : hashable
            Unique identifier of the dataset to be added.
        fragments_path : str
            Path to predicted SWC files to be loaded.
        img_path : str
            Path to the raw image associated with the fragments.
        config : Config
            Configuration object containing parameters and settings.
        gt_path : str, optional
            Path to ground-truth SWC files to be loaded. Default is None.
        metadata_path : str, optional
            Patch to JSON file containing metadata on block that fragments
            were extracted from. Default is None.
        segmentation_path : str, optional
            Path to the segmentation that fragments were generated from.
            Default is None.
        soma_centroids : List[Tuple[int]], optional
            Phyiscal coordinates of soma centroids. Default is None.
        """
        assert key not in self.datasets, "Key has been used!"
        self.datasets[key] = FragmentsDataset(
            fragments_path,
            img_path,
            config,
            gt_path=gt_path,
            metadata_path=metadata_path,
            segmentation_path=segmentation_path,
            soma_centroids=soma_centroids
        )

    def __iter__(self):
        """
        Iterates over the datasets and yields model-ready inputs and targets.

        Yields
        ------
        inputs : TensorDict
            Heterogeneous graph data.
        targets : torch.Tensor
            Ground truth labels.
        """
        samplers = self.init_samplers()
        while len(samplers) > 0:
            key = self.get_next_key(samplers)
            try:
                # Extract features
                subgraph = next(samplers[key])
                features = self.datasets[key].feature_extractor(subgraph)
                data = HeteroGraphData(features)

                # Get training inputs
                inputs = data.get_inputs()
                targets = data.get_targets()
                yield inputs, targets
            except StopIteration:
                del samplers[key]

    def generate_proposals(self, search_radius):
        """
        Generates proposals for each dataset.

        Parameters
        ----------
        search_radius : float
            Search radius used to generate proposals.
        """
        # Proposal generation
        for key in self.datasets:
            self.datasets[key].graph.generate_proposals(search_radius)

        # Report results
        print("# Proposals:", self.n_proposals())
        print("% Accepts:", self.p_accepts())

    # --- Helpers ---
    def __len__(self):
        """
        Returns the number of datasets in self.

        Returns
        -------
        float
            Number of datasets in self.
        """
        return len(self.datasets)

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
        """
        Initializes subgraph samplers for each dataset.

        Returns
        -------
        samplers : Dict[hashable, SubgraphSampler]
            Subgraph samplers used to iterate over the datasets.
        """
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
        return accepts_cnt / (self.n_proposals() + 1e-5)

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
