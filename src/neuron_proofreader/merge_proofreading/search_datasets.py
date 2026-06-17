"""
Created on Wed August 4 16:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code for detecting merge mistakes on skeletons generated from an automated
image segmentation.

"""

from abc import ABC, abstractmethod
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from queue import Queue
from threading import Thread
from torch.utils.data import IterableDataset

import networkx as nx
import numpy as np
import torch

from neuron_proofreader.machine_learning.image_dataloader import (
    DetectionBatchLoader,
    DetectionPatchLoader,
)
from neuron_proofreader.utils import img_util, ml_util, util


# --- Datasets ---
class SearchDataset(IterableDataset, ABC):

    def __init__(
        self,
        graph,
        img_config,
        is_multimodal=False,
        min_search_size=0,
        prefetch=32,
        subgraph_radius=100,
    ):
        # Call parent class
        super().__init__()

        # Instance attributes
        self.distance_traversed = 0
        self.graph = graph
        self.is_multimodal = is_multimodal
        self.min_size = min_search_size
        self.patch_shape = img_config.patch_shape
        self.prefetch = prefetch
        self.subgraph_radius = subgraph_radius

        # Input getter
        if is_multimodal:
            self.get_input = self.get_patch_and_pointcloud
        else:
            self.get_input = self.get_patch

    # --- Core routines ---
    def __iter__(self):
        patch_queue = Queue(maxsize=self.prefetch)
        sentinel = object()

        def producer():
            sites = self._all_sites()
            with ThreadPoolExecutor(max_workers=self.prefetch) as executor:
                futures = {}

                def fill():
                    while len(futures) < self.prefetch:
                        try:
                            site = next(sites)
                            futures[
                                executor.submit(self.patch_loader, site)
                            ] = site
                        except StopIteration:
                            break

                fill()
                while futures:
                    done, _ = wait(futures, return_when=FIRST_COMPLETED)
                    for f in done:
                        patch_queue.put((futures.pop(f), f.result()))
                    fill()

            patch_queue.put(sentinel)

        Thread(target=producer, daemon=True).start()

        while True:
            item = patch_queue.get()
            if item is sentinel:
                break
            yield from self.get_input(*item)

    def _all_sites(self):
        visited_ids = set()
        valid_ids = self.find_fragments_to_search()
        for u in self.graph.leaf_nodes():
            component_id = self.node_component_id[u]
            if component_id not in visited_ids and component_id in valid_ids:
                visited_ids.add(component_id)
                yield from self.generate_component_sites(u)

    @abstractmethod
    def generate_component_sites(self, root):
        """
        Abstract method to be implemented by subclasses.
        """
        pass

    # --- Helpers ---
    def __getattr__(self, name):
        return getattr(self.graph, name)

    @abstractmethod
    def estimate_iterations(self):
        pass

    def find_fragments_to_search(self):
        component_ids = set()
        for nodes in nx.connected_components(self.graph):
            # Compute path length
            node = util.sample_once(list(nodes))
            length = self.graph.cable_length(
                max_depth=self.min_size, root=node
            )

            # Check if path length satisfies threshold
            if length > self.min_size:
                component_ids.add(self.node_component_id[node])
        return component_ids

    def is_contained(self, node):
        voxel = self.node_voxel(node)
        shape = self.patch_loader.img.shape()[2::]
        buffer = np.max(self.patch_shape) + 1
        return img_util.is_contained(voxel, shape, buffer=buffer)

    def is_near_leaf(self, node, threshold=32):
        # Check if node is branching
        if self.degree[node] > 2:
            return False

        # Search neighborhood
        queue = [(node, 0)]
        visited = {node}
        while len(queue) > 0:
            # Visit node
            i, dist_i = queue.pop()
            if self.degree[i] == 1:
                return True

            # Update queue
            for j in self.neighbors(i):
                dist_j = dist_i + self.dist(i, j)
                if j not in visited and dist_j < threshold:
                    queue.append((j, dist_j))
                    visited.add(j)
        return False

    def is_node_valid(self, node):
        is_contained = self.is_contained(node)
        is_nonleaf = not self.is_near_leaf(node)
        return is_contained and is_nonleaf


class DenseSearchDataset(SearchDataset):

    max_batch_span = 512

    def __init__(
        self,
        graph,
        img_config,
        is_multimodal=False,
        min_search_size=0,
        prefetch=64,
        step_size=40,
        subgraph_radius=100,
    ):
        # Call parent class
        super().__init__(
            graph,
            img_config,
            is_multimodal=is_multimodal,
            min_search_size=min_search_size,
            prefetch=prefetch,
            subgraph_radius=subgraph_radius,
        )

        # Instance attributes
        self.patch_loader = DetectionBatchLoader(self.graph, img_config)
        self.search_mode = "dense"
        self.step_size = step_size

    def generate_component_sites(self, root):
        """
        Generates batches of nodes from the connected component that contains
        the given root node.

        Returns
        -------
        Iterator[numpy.ndarray]
            Generator that yields batches of nodes from the connected
            component containing the given root node.
        """
        nodes = list()
        for i, j in nx.dfs_edges(self.graph, source=root):
            # Check if starting new batch
            self.distance_traversed += self.dist(i, j)
            if len(nodes) == 0:
                if self.is_node_valid(i):
                    root = i
                    last_node = i
                    nodes.append(i)
                else:
                    continue

            # Check whether to yield batch
            if self.dist(root, j) > self.max_batch_span:
                yield np.array(nodes, dtype=int)
                nodes = list()

            # Visit j
            is_next = self.dist(last_node, j) >= self.step_size - 2
            is_branching = self.degree[j] >= 3
            if (is_next or is_branching) and self.is_node_valid(j):
                last_node = j
                nodes.append(j)
                if len(nodes) == 1:
                    root = j

        # Yield any remaining nodes after the loop
        if nodes:
            yield np.array(nodes, dtype=int)

    def get_patch(self, nodes, img, offset):
        img = torch.from_numpy(img).float()
        voxels = np.array([self.node_voxel(i) for i in nodes], dtype=int)
        for node, center in zip(nodes, voxels - offset):
            s = img_util.get_slices(center, self.patch_shape)
            yield node, img[(slice(0, 2), *s)]

    def generate_patch_and_pc(self, nodes, img, offset):
        pass

    # --- Helpers ---
    def estimate_iterations(self):
        """
        Estimates the number of iterations required to search graph.

        Returns
        -------
        int
            Estimated number of iterations required to search graph.
        """
        # Search graph
        total_cable_length = 0
        n_fragments = 0
        for nodes in map(list, nx.connected_components(self.graph)):
            cable_length = self.cable_length(root=nodes[0])
            if cable_length > self.min_size:
                total_cable_length += cable_length
                n_fragments += 1

        # Report results
        print("# Fragments:", n_fragments)
        print(f"Total Cable Length: {total_cable_length / 10**5:.2f}cm")
        return int(total_cable_length / self.step_size)


class SparseSearchDataset(SearchDataset):
    pass


class BranchingSearchDataset(SearchDataset):

    def __init__(
        self,
        graph,
        img_config,
        is_multimodal=False,
        min_search_size=0,
        prefetch=64,
        step_size=10,
        subgraph_radius=100,
    ):
        # Call parent class
        super().__init__(
            graph,
            img_config,
            is_multimodal=is_multimodal,
            min_search_size=min_search_size,
            prefetch=prefetch,
            subgraph_radius=subgraph_radius,
        )

        # Instance attributes
        self.patch_loader = DetectionPatchLoader(self.graph, img_config)
        self.search_mode = "branching_nodes"

    def estimate_iterations(self):
        return len(self.branching_nodes())

    def generate_component_sites(self, root):
        visited = set()
        for i, j in nx.dfs_edges(self.graph, source=root):
            if self.degree[i] >= 3 and i not in visited:
                visited.add(i)
                yield i

    def get_patch(self, node, img):
        yield node, torch.from_numpy(img).float()
