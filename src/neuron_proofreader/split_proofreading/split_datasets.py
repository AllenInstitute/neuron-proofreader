"""
Created on Fri April 11 11:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Dataset that iterates over the proposals of a fragments graph in batches and
extracts the model inputs for each batch on a background thread.

"""

from queue import Queue
from threading import Thread
from torch.utils.data import IterableDataset

from neuron_proofreader.machine_learning.subgraph_sampler import (
    SubgraphSampler,
)
from neuron_proofreader.split_proofreading.split_feature_extraction import (
    HeteroGraphData,
    ProposalPatchExtractor,
    ProposalTreeFeatureExtractor,
)


class ArboristSplitDataset(IterableDataset):
    """
    Yields one HeteroGraphData per batch of proposals, with an image patch
    and an Arborist TreeSample per proposal. Attribute lookups that are not
    found on the dataset fall through to the underlying graph.
    """

    def __init__(
        self,
        fragments_graph,
        img_config,
        batch_size=32,
        prefetch=2,
        **tree_kwargs,
    ):
        """
        Parameters
        ----------
        fragments_graph : ProposalGraph
            Graph with proposals generated.
        img_config : ImageConfig
            Image configuration.
        batch_size : int, optional
            Maximum number of proposals per batch. Default is 32.
        prefetch : int, optional
            Number of batches to extract features for ahead of the caller.
            Note: each queued batch holds its image patches in memory, which
            is roughly 7MB per proposal. Default is 2.
        **tree_kwargs
            Forwarded to ProposalTreeFeatureExtractor.
        """
        self.batch_size = batch_size
        self.graph = fragments_graph
        self.prefetch = prefetch
        self.patch_extractor = ProposalPatchExtractor(fragments_graph, img_config)
        self.tree_extractor = ProposalTreeFeatureExtractor(**tree_kwargs)
        self._proposal_subset = None

    def __getattr__(self, name):
        return getattr(self.graph, name)

    def __iter__(self):
        """
        Features are extracted on a background thread so that reading image
        patches overlaps the caller's forward pass. At most "prefetch"
        batches are held in memory at once.
        """
        queue = Queue(maxsize=self.prefetch)
        sentinel = object()

        def producer():
            try:
                for data in self.extract_batches():
                    queue.put(data)
            except Exception as e:
                queue.put(e)
            finally:
                queue.put(sentinel)

        Thread(target=producer, daemon=True).start()
        while (item := queue.get()) is not sentinel:
            if isinstance(item, Exception):
                raise item
            yield item

    def extract_batches(self):
        """
        The image reads of the next batch are issued before the current batch
        is processed, so the cloud latency of a batch is hidden behind the
        CPU work of the previous one.
        """
        sampler = SubgraphSampler(
            self.graph,
            max_proposals=self.batch_size,
            proposals=self._proposal_subset,
        )
        sampler = iter(sampler)
        subgraph = next(sampler, None)
        reads = self.issue_reads(subgraph)
        while subgraph is not None:
            next_subgraph = next(sampler, None)
            next_reads = self.issue_reads(next_subgraph)
            patches = self.patch_extractor(subgraph, reads)
            trees = {p: self.tree_extractor(self.graph, p) for p in subgraph.proposals}
            yield HeteroGraphData(subgraph, patches, trees)
            subgraph, reads = next_subgraph, next_reads

    def issue_reads(self, subgraph):
        if subgraph is None:
            return None
        return self.patch_extractor.issue_reads(subgraph)

    def iter_proposals(self, proposals):
        """
        Iterates over only the given proposals, otherwise exactly as
        "__iter__" does.
        """
        self._proposal_subset = list(proposals)
        try:
            yield from self
        finally:
            self._proposal_subset = None
