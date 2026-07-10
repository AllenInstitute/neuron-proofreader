"""
Created on Fri April 11 11:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Implementation of dataset objects that contain graphs and facilitate feature
generation for training and inference in split-correction tasks.

"""

from collections import deque
from concurrent.futures import ThreadPoolExecutor, FIRST_COMPLETED, wait
from queue import Queue
from threading import Thread
from torch.utils.data import Dataset, IterableDataset

import numpy as np
import os
import random

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

    def __init__(self, fragments_graph, img_config):
        """
        Instantiates a FragmentsDataset object.

        Parameters
        ----------
        fragments_graph : str
            ...
        img_config : ImageConfig
            Configuration object containing image parameters.
        """
        self.graph = fragments_graph
        self.feature_extractor = FeaturePipeline(
            self.graph,
            img_config.img_path,
            brightness_clip=img_config.brightness_clip,
            patch_shape=img_config.patch_shape,
        )

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
        for subgraph in self.sampler():
            yield HeteroGraphData(self.feature_extractor(subgraph))

    # --- Helpers ---
    def __getattr__(self, name):
        return getattr(self.graph, name)

    def sampler(self, batch_size):
        """
        Gets a subgraph sampler used to iterate over dataset.

        Returns
        -------
        sampler : SubgraphSampler
            Subgraph sampler that is used to iterate over dataset.
        """
        return iter(SubgraphSampler(self.graph, max_proposals=batch_size))


class FragmentsCollection(Dataset):
    """
    Stores FragmentDataset objects for a set of whole-brain images and
    manages proposal generation and dataset-level statistics.
    """

    def __init__(self):
        """
        Parameters
        ----------
        datasets : Dict[Hashable, FragmentDataset]
            Mapping from dataset key to FragmentDataset.
        """
        self.datasets = dict()

    def add(self, key, dataset):
        assert key not in self.datasets, "Dataset already exists!"
        self.datasets[key] = dataset

    def __getitem__(self, key):
        return self.datasets[key]

    # --- Helpers ---
    def __getattr__(self, name):
        return getattr(self.datasets, name)

    def __len__(self):
        return len(self.datasets)

    def __repr__(self):
        n_p = np.sum([ds.n_proposals() for ds in self.values()])
        n_a = np.sum([len(ds.gt_accepts) for ds in self.values()])
        return (
            f"FragmentsCollection("
            f"num_datasets={len(self)}, "
            f"num_proposals={n_p}, "
            f"percent_accepts={100 * n_a / (n_p + 1e-5):.2f})"
        )


class FragmentsLoader:
    """
    Prefetching loader that samples subgraphs from a FragmentsCollection.
    A bounded pool of worker threads pulls batches from datasets in
    round-robin order, overlapping I/O across up to `max_workers` datasets
    at once, and feeds a bounded queue consumed by the training loop.
    """

    def __init__(self, collection, batch_size=32, prefetch=4, max_workers=4, shuffle=True):
        """
        Parameters
        ----------
        collection : FragmentsCollection
        batch_size : int, optional
            Max proposals per subgraph sampled from each dataset. Default is 32.
        prefetch : int, optional
            Number of batches to buffer ahead of consumption. Default is 4.
        max_workers : int, optional
            Number of datasets that may be fetched from concurrently. Bounded
            regardless of collection size. Default is 4.
        shuffle : bool, optional
            If True, datasets are visited in random round-robin order each
            epoch. Default is True.
        """
        self.batch_size = batch_size
        self.collection = collection
        self.prefetch = prefetch
        self.max_workers = max_workers
        self.shuffle = shuffle

    def __iter__(self):
        queue = Queue(maxsize=self.prefetch)
        thread = Thread(target=self._produce, args=(queue,), daemon=True)
        thread.start()

        while True:
            item = queue.get()
            if item is StopIteration:
                return
            if isinstance(item, Exception):
                raise item
            yield item

    def _produce(self, queue):
        try:
            samplers = self._init_samplers()
            rotation = deque(samplers.keys())
            if self.shuffle:
                random.shuffle(rotation)

            n_workers = min(self.max_workers, len(rotation)) or 1
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                pending = dict()  # future -> key
                for _ in range(n_workers):
                    self._submit_next(executor, rotation, samplers, pending)

                while pending:
                    done, _ = wait(pending.keys(), return_when=FIRST_COMPLETED)
                    for future in done:
                        key = pending.pop(future)
                        batch = future.result()
                        if batch is not None:
                            queue.put(batch)
                            rotation.append(key)  # dataset has more, requeue
                        self._submit_next(executor, rotation, samplers, pending)
        except Exception as e:
            queue.put(e)
            return
        queue.put(StopIteration)

    def _submit_next(self, executor, rotation, samplers, pending):
        if not rotation:
            return
        key = rotation.popleft()
        future = executor.submit(self._fetch_batch, key, samplers[key])
        pending[future] = key

    def _fetch_batch(self, key, sampler):
        try:
            subgraph = next(sampler)
        except StopIteration:
            return None
        dataset = self.collection[key]
        features = dataset.feature_extractor(subgraph)
        data = HeteroGraphData(features)
        return data.get_inputs(), data.get_targets()

    def _init_samplers(self):
        samplers = dict()
        for key, dataset in self.collection.datasets.items():
            samplers[key] = iter(
                SubgraphSampler(dataset.graph, max_proposals=self.batch_size)
            )
        return samplers

    def __len__(self):
        batch_cnt = 0
        for ds in self.collection.datasets.values():
            batch_cnt += np.ceil(ds.n_proposals() / self.batch_size)
        return int(batch_cnt)


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
