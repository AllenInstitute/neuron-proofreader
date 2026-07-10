"""
Created on Fri April 11 11:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Implementation of dataset objects that contain graphs and facilitate feature
generation for training and inference in split-correction tasks.

"""

from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from torch.utils.data import IterableDataset
from tqdm import tqdm

import os
import pandas as pd

from neuron_proofreader.machine_learning.subgraph_sampler import (
    SubgraphSampler,
)
from neuron_proofreader.split_proofreading.split_feature_extraction import (
    FeaturePipeline,
    HeteroGraphData,
)
from neuron_proofreader.utils import util


# --- Datasets ---
class FragmentsDataset(IterableDataset):
    """
    A dataset object that contains a graph built from fragments corresponding
    to a single brain.
    """

    def __init__(self, fragments_graph, img_config, proposals_per_batch=64):
        """
        Instantiates a FragmentsDataset object.

        Parameters
        ----------
        fragments_graph : str
            ...
        img_config : ImageConfig
            Configuration object containing image parameters.
        proposals_per_batch : int, optional
            Maximum number of proposals in subgraphs yielded per batch.
            Default is 64.
        """
        self.graph = fragments_graph
        self.feature_extractor = FeaturePipeline(
            self.graph,
            img_config.img_path,
            brightness_clip=img_config.brightness_clip,
            patch_shape=img_config.patch_shape,
        )
        self.proposals_per_batch = proposals_per_batch

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
    def __getattr__(self, name):
        return getattr(self.graph, name)

    def get_sampler(self):
        """
        Gets a subgraph sampler used to iterate over dataset.

        Returns
        -------
        sampler : SubgraphSampler
            Subgraph sampler that is used to iterate over dataset.
        """
        sampler = SubgraphSampler(
            self.graph, max_proposals=self.proposals_per_batch
        )
        return iter(sampler)


class FragmentsDatasetCollection(IterableDataset):
    """
    A dataset class for storing a set of FragmentDataset objects corresponding
    to different whole-brain images.
    """

    def __init__(self, datasets, shuffle=True):
        """
        Instantiates a FragmentsDatasetCollection object.

        Parameters
        ----------
        shuffle : bool, optional
            True if examples should be shuffled each epoch. Default is True.
        """
        # Instance attributes
        self.datasets = datasets
        self.shuffle = shuffle

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
        # Create prefetching data structures
        samplers = self.init_samplers()
        queue = Queue(maxsize=self.prefetch * len(samplers))
        active_keys = set(samplers.keys())

        # Launch one prefetch thread per dataset
        with ThreadPoolExecutor(max_workers=len(samplers)) as executor:
            for key, sampler in samplers.items():
                executor.submit(self._worker, key, sampler, queue)

            # Consume from queue until all datasets exhausted
            while active_keys:
                key, inputs, targets = queue.get()
                if inputs is StopIteration:
                    active_keys.discard(key)
                    continue
                if isinstance(inputs, Exception):
                    raise inputs
                yield inputs, targets

    def _worker(self, key, sampler, queue):
        """
        Runs in a background thread, prefetches extracted features into queue.
        """
        try:
            for subgraph in sampler:
                features = self.datasets[key].feature_extractor(subgraph)
                data = HeteroGraphData(features)
                queue.put((key, data.get_inputs(), data.get_targets()))
        except Exception as e:
            queue.put((key, e, None))
        finally:
            queue.put((key, StopIteration, None))

    def generate_proposals(self, proposal_config):
        """
        Generates proposals for each dataset.

        Parameters
        ----------
        proposal_config : ProposalConfig
            Configuration object containing parameters for generating
            proposals.
        """
        for key in tqdm(self.datasets, desc="Generate Proposals"):
            self.datasets[key].graph.generate_proposals(
                proposal_config.search_radius,
                **proposal_config,
            )

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
        return 100 * accepts_cnt / (self.n_proposals() + 1e-5)

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
def generate_example_ids(ds_prefix):
    """
    Generates dataset example identifiers.

    Parameters
    ----------
    ds_prefix : str
        Root prefix under which datasets are organized.

    Yields
    -------
    Tuple[str]
        Dataset ID formatted as (brain_id, segmentation_id, block_id).
    """
    bucket_name, _ = util.parse_cloud_path(ds_prefix)
    for brain_prefix in util.list_gcs_subprefixes(ds_prefix):
        # Extract brain id
        brain_id = brain_prefix.split("/")[-2]

        # Iterate over segmentations
        pred_prefix = os.path.join(bucket_name, brain_prefix, "pred_swcs")
        for brain_seg_prefix in util.list_gcs_subprefixes(pred_prefix):
            # Extract segmentation id
            segmentation_id = brain_seg_prefix.split("/")[-2]

            # Iterate over blocks
            ex_prefix = os.path.join(bucket_name, brain_seg_prefix)
            for block_prefix in util.list_gcs_subprefixes(ex_prefix):
                # Extract block id
                block_id = block_prefix.split("/")[-2]
                yield brain_id, segmentation_id, block_id
