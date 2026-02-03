"""
Created on Wed August 4 16:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code for detecting merge mistakes on skeletons generated from an automated
image segmentation.

"""

from abc import ABC, abstractmethod
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from torch.nn.functional import sigmoid
from torch.utils.data import IterableDataset
from time import time
from tqdm import tqdm

import networkx as nx
import numpy as np
import os
import torch

from neuron_proofreader.machine_learning.point_cloud_models import (
    subgraph_to_point_cloud,
)
from neuron_proofreader.utils import (
    geometry_util,
    img_util,
    ml_util,
    swc_util,
    util,
)


class MergeDetector:

    def __init__(
        self,
        dataset,
        model,
        model_path,
        device="cuda",
        remove_detected_sites=False,
        threshold=0.4,
    ):
        # Instance attributes
        self.dataset = dataset
        self.device = device
        self.node_preds = np.ones((len(dataset.graph.node_xyz))) * 1e-2
        self.patch_shape = dataset.patch_shape
        self.remove_detected_sites = remove_detected_sites
        self.threshold = threshold

        # Load model
        self.model = model
        ml_util.load_model(model, model_path, device=self.device)

    # --- Core routines
    def search_graph(self):
        # Initialize progress bar
        pbar = tqdm(total=self.dataset.estimate_iterations())
        t0 = time()

        # Iterate over dataset
        likelihoods = list()
        merge_sites = list()
        for nodes, x_nodes in self.dataset:
            y_nodes = self.predict(x_nodes)
            idxs = np.where(y_nodes > self.threshold)[0]
            if len(idxs) > 0:
                merge_sites.extend(nodes[idxs].tolist())
                likelihoods.extend(y_nodes[idxs].tolist())

            self.node_preds[np.array(nodes)] = y_nodes
            pbar.update(len(nodes))

        # Non-maximum suppression of detected sites
        merge_sites = self.filter_with_nms(merge_sites, likelihoods)
        rate = self.dataset.distance_traversed / (time() - t0)

        # Report results
        print("\n# Detected Merge Sites:", len(merge_sites))
        print(f"Distance Traversed: {self.dataset.distance_traversed:.2f}μm")
        print(f"Merge Proofreading Rate: {rate:.2f}μm/s")

        # Remove merge mistakes (optional)
        if self.remove_detected_sites:
            pass
        return merge_sites

    def predict(self, x_nodes):
        """
        Predicts merge site likelihoods for the given node features.

        Parameters
        ----------
        x_nodes : torch.Tensor
            Node features.

        Returns
        -------
        numpy.ndarray
            Predicted merge site likelihoods.
        """
        with torch.inference_mode():
            x_nodes = x_nodes.to(self.device)
            y_nodes = sigmoid(self.model(x_nodes))
            return np.squeeze(ml_util.to_cpu(y_nodes, to_numpy=True), axis=1)

    def filter_with_nms(self, merge_sites, likelihoods):
        # Sort by confidence
        idxs = np.argsort(likelihoods)
        merge_sites = [merge_sites[i] for i in idxs]

        # NMS
        merge_sites_set = set(merge_sites)
        filtered_merge_sites = set()
        while merge_sites:
            # Local max
            root = merge_sites.pop()
            xyz_root = self.dataset.graph.node_xyz[root]
            if root in merge_sites_set:
                filtered_merge_sites.add(root)
                merge_sites_set.remove(root)
            else:
                continue

            # Suppress neighborhood
            queue = [(root, 0)]
            visited = set([root])
            while queue:
                # Visit node
                i, dist_i = queue.pop()
                if i in merge_sites_set:
                    xyz_i = self.dataset.graph.node_xyz[i]
                    iou = img_util.compute_iou3d(
                        xyz_i, xyz_root, self.patch_shape, self.patch_shape
                    )
                    if iou > 0.35:
                        merge_sites_set.remove(i)
                        self.node_preds[i] = 1e-2

                # Populate queue
                for j in self.dataset.graph.neighbors(i):
                    dist_j = dist_i + self.dataset.graph.dist(i, j)
                    if j not in visited and dist_j < self.patch_shape[0]:
                        queue.append((j, dist_j))
                        visited.add(j)
        return filtered_merge_sites

    def remove_merge_sites(self, detected_merge_sites):
        pass

    # --- Helpers ---
    def get_detected_sites(self, threshold):
        nodes = np.where(self.node_preds >= threshold)[0]
        return [self.dataset.graph.node_xyz[i] for i in nodes]

    def save_results(self, output_dir, output_prefix_s3=None, save_fragments=True):
        # Get predicted merge sites
        nodes = np.where(self.node_preds >= self.threshold)[0]
        detected_sites = [self.dataset.graph.node_xyz[i] for i in nodes]

        # Save predicted merge sites
        zip_path = os.path.join(output_dir, "detected_sites.zip")
        swc_util.write_points(
            zip_path,
            detected_sites,
            color="1.0 0.0 0.0",
            prefix="merge-site",
            radius=10,
        )

        # Save fragments
        if save_fragments:
            fragments_path = os.path.join(output_dir, "fragments.zip")
            self.dataset.graph.to_zipped_swcs(fragments_path)

        # Upload results to S3 (if applicable)
        if output_prefix_s3:
            bucket_name, prefix = util.parse_cloud_path(output_prefix_s3)
            util.upload_dir_to_s3(output_dir, bucket_name, prefix)

    def save_parameters(self, output_dir):
        json_path = os.path.join(output_dir, "detection_parameters.json")
        parameters = {
            "accept_threshold": self.threshold,
            "is_multimodal": self.dataset.is_multimodal,
            "min_search_size": self.dataset.min_size,
            "patch_shape": self.patch_shape,
            "remove_detected_sites": self.remove_detected_sites,
            "search_mode": self.dataset.search_mode,
            "subgraph_radius": self.dataset.subgraph_radius,
        }
        util.write_json(json_path, parameters)


# --- Data Handling ---
class GraphDataset(IterableDataset, ABC):

    def __init__(
        self,
        graph,
        img_path,
        patch_shape,
        batch_size=16,
        brightness_clip=300,
        is_multimodal=False,
        min_search_size=0,
        prefetch=64,
        segmentation_path=None,
        subgraph_radius=100,
        use_new_mask=False
    ):
        # Call parent class
        super().__init__()

        # Instance attributes
        self.batch_size = batch_size
        self.brightness_clip = brightness_clip
        self.distance_traversed = 0
        self.graph = graph
        self.is_multimodal = is_multimodal
        self.min_size = min_search_size
        self.patch_shape = patch_shape
        self.prefetch = prefetch
        self.segmentation_path = segmentation_path
        self.subgraph_radius = subgraph_radius
        self.use_new_mask = use_new_mask

        # Batch getter
        if is_multimodal:
            self.get_batch = self._get_multimodal_batch
        else:
            self.get_batch = self._get_batch

        # Image reader
        self.img_reader = img_util.TensorStoreReader(img_path)
        if self.segmentation_path:
            self.segmentation_reader = img_util.TensorStoreReader(
                segmentation_path
            )

    # --- Core routines ---
    def __iter__(self):
        # Find fragment IDs to check
        valid_ids = self.find_fragments_to_search()

        # Search graph
        visited_ids = set()
        for u in self.graph.get_leafs():
            component_id = self.graph.node_component_id[u]
            if component_id not in visited_ids and component_id in valid_ids:
                visited_ids.add(component_id)
                yield from self._generate_batches_from_component(u)

    @abstractmethod
    def _generate_batches_from_component(self, root):
        """
        Abstract method to be implemented by subclasses.
        """
        pass

    @abstractmethod
    def _generate_batch_nodes(self, root):
        """
        Abstract method to be implemented by subclasses.
        """

    # --- Helpers ---
    @abstractmethod
    def estimate_iterations(self):
        pass

    def find_fragments_to_search(self):
        component_ids = set()
        for nodes in nx.connected_components(self.graph):
            # Compute path length
            node = util.sample_once(list(nodes))
            length = self.graph.path_length(root=node, max_depth=self.min_size)

            # Check if path length satisfies threshold
            if length > self.min_size:
                component_ids.add(self.graph.node_component_id[node])
        return component_ids

    def get_patch_centers(self, nodes):
        patch_centers = [self.graph.get_voxel(i) for i in nodes]
        return np.array(patch_centers, dtype=int)

    def get_label_mask(self, nodes, img_shape, offset):
        # Read segmentation
        if self.use_new_mask:
            center = [o + s // 2 for o, s in zip(offset, img_shape)]
            segment_mask = self.segmentation_reader.read(center, img_shape)
            segment_mask = img_util.remove_small_segments(segment_mask, 1000)
            segment_mask = 0.5 * (segment_mask > 0).astype(int)
        else:
            segment_mask = np.zeros(img_shape)

        # Annotate mask
        subgraph = self.get_contained_subgraph(nodes, img_shape, offset)
        for i, j in subgraph.edges:
            voxel_i = self.graph.get_voxel(i) - offset
            voxel_j = self.graph.get_voxel(j) - offset
            voxels = geometry_util.make_digital_line(voxel_i, voxel_j)
            img_util.annotate_voxels(segment_mask, voxels)
        return segment_mask

    def get_contained_subgraph(self, nodes, img_shape, offset):
        queue = list(nodes)
        visited = set(nodes)
        subgraph = nx.Graph()
        while queue:
            # Visit node
            i = queue.pop()
            voxel_i = self.graph.get_voxel(i) - offset
            if not img_util.is_contained(voxel_i, img_shape, buffer=1):
                continue

            # Update queue
            for j in self.graph.neighbors(i):
                voxel_j = self.graph.get_voxel(j) - offset
                if img_util.is_contained(voxel_j, img_shape):
                    subgraph.add_edge(i, j)
                    if j not in visited:
                        queue.append(j)
                        visited.add(j)
        return subgraph

    def is_contained(self, node):
        voxel = self.graph.get_voxel(node)
        shape = self.img_reader.shape()[2::]
        buffer = np.max(self.patch_shape) + 1
        return img_util.is_contained(voxel, shape, buffer=buffer)

    def read_superchunk(self, nodes):
        # Compute bounding box
        patch_centers = self.get_patch_centers(nodes)
        buffer = 1 + np.array(self.patch_shape) // 2
        start = patch_centers.min(axis=0) - buffer
        end = patch_centers.max(axis=0) + buffer

        # Read image
        shape = (end - start).astype(int)
        center = (start + shape // 2).astype(int)
        superchunk = self.img_reader.read(center, shape)
        superchunk = np.minimum(superchunk, self.brightness_clip)
        return superchunk, start.astype(int)

    def is_near_leaf(self, node, threshold=20):
        # Check if node is branching
        if self.graph.degree[node] > 2:
            return False

        # Search neighborhood
        queue = [(node, 0)]
        visited = {node}
        while len(queue) > 0:
            # Visit node
            i, dist_i = queue.pop()
            if self.graph.degree[i] == 1:
                return True

            # Update queue
            for j in self.graph.neighbors(i):
                dist_j = dist_i + self.graph.dist(i, j)
                if j not in visited and dist_j < threshold:
                    queue.append((j, dist_j))
                    visited.add(j)
        return False

    def is_node_valid(self, node):
        is_contained = self.is_contained(node)
        is_nonleaf = not self.is_near_leaf(node)
        return is_contained and is_nonleaf


class DenseGraphDataset(GraphDataset):

    def __init__(
        self,
        graph,
        img_path,
        patch_shape,
        batch_size=16,
        brightness_clip=300,
        is_multimodal=False,
        min_search_size=0,
        prefetch=128,
        segmentation_path=None,
        step_size=10,
        subgraph_radius=100,
        use_new_mask=False
    ):
        # Call parent class
        super().__init__(
            graph,
            img_path,
            patch_shape,
            batch_size=batch_size,
            brightness_clip=brightness_clip,
            is_multimodal=is_multimodal,
            min_search_size=min_search_size,
            prefetch=prefetch,
            segmentation_path=segmentation_path,
            subgraph_radius=subgraph_radius,
            use_new_mask=use_new_mask
        )

        # Instance attributes
        self.search_mode = "dense"
        self.step_size = step_size

    def _generate_batches_from_component(self, root):
        # Subroutines
        def submit_thread():
            try:
                nodes = next(batch_nodes_generator)
                thread = executor.submit(self.read_superchunk, nodes)
                pending[thread] = nodes
            except StopIteration:
                pass

        # Main
        batch_nodes_generator = self._generate_batch_nodes(root)
        with ThreadPoolExecutor(max_workers=128) as executor:
            try:
                # Prefetch batches
                pending = dict()
                for _ in range(self.prefetch):
                    submit_thread()

                # Yield batches
                while pending:
                    done, _ = wait(pending.keys(), return_when=FIRST_COMPLETED)
                    for thread in done:
                        # Process completed thread
                        nodes = pending.pop(thread)
                        img, offset = thread.result()
                        yield self.get_batch(nodes, img, offset)

                        # Continue submitting threads
                        submit_thread()
            finally:
                pass

    def _generate_batch_nodes(self, root):
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
            self.distance_traversed += self.graph.dist(i, j)
            if len(nodes) == 0:
                if self.is_node_valid(i):
                    root = i
                    last_node = i
                    nodes.append(i)
                else:
                    continue

            # Check whether to yield batch
            is_node_far = self.graph.dist(root, j) > 512
            is_batch_full = len(nodes) == self.batch_size
            if is_node_far or is_batch_full:
                # Yield nodes in batch
                yield np.array(nodes, dtype=int)

                # Reset batch metadata
                nodes = list()

            # Visit j
            is_next = self.graph.dist(last_node, j) >= self.step_size - 2
            is_branching = self.graph.degree[j] >= 3
            if (is_next or is_branching) and self.is_node_valid(j):
                last_node = j
                nodes.append(j)
                if len(nodes) == 1:
                    root = j

        # Yield any remaining nodes after the loop
        if nodes:
            yield np.array(nodes, dtype=int)

    def _get_batch(self, nodes, img, offset):
        # Initializations
        label_mask = self.get_label_mask(nodes, img.shape, offset)
        patch_centers = self.get_patch_centers(nodes) - offset

        # Populate batch array
        batch = np.empty((len(nodes), 2,) + self.patch_shape)
        for i, center in enumerate(patch_centers):
            s = img_util.get_slices(center, self.patch_shape)
            batch[i, 0, ...] = img_util.normalize(img[s])
            batch[i, 1, ...] = label_mask[s]
        return nodes, torch.tensor(batch, dtype=torch.float)

    def _get_multimodal_batch(self, nodes, img, offset):
        # Initializations
        label_mask = self.get_label_mask(nodes, img.shape, offset)
        patch_centers = self.get_patch_centers(nodes) - offset

        # Populate batch array
        patches = np.empty((len(nodes), 2,) + self.patch_shape)
        point_clouds = np.empty((len(nodes), 3, 3600), dtype=np.float32)
        for i, (node, center) in enumerate(zip(nodes, patch_centers)):
            s = img_util.get_slices(center, self.patch_shape)
            patches[i, 0, ...] = img_util.normalize(img[s])
            patches[i, 1, ...] = label_mask[s]

            subgraph = self.graph.get_rooted_subgraph(
                node, self.subgraph_radius
            )
            point_clouds[i] = subgraph_to_point_cloud(subgraph)

        # Build batch dictionary
        batch = ml_util.TensorDict(
            {
                "img": ml_util.to_tensor(patches),
                "point_cloud": ml_util.to_tensor(point_clouds),
            }
        )
        return nodes, batch

    # --- Helpers ---
    def estimate_iterations(self):
        """
        Estimates the number of iterations required to search graph.

        Returns
        -------
        int
            Estimated number of iterations required to search graph.
        """
        # Set min size
        length = 0
        n_componenets = 0
        for nodes in map(list, nx.connected_components(self.graph)):
            node = util.sample_once(nodes)
            length_component = self.graph.path_length(root=node)
            if length_component > self.min_size:
                length += length_component
                n_componenets += 1
        print("# Fragments to Search:", n_componenets)
        return int(length / self.step_size)


class SparseGraphDataset(GraphDataset):

    def __init__(
        self,
        graph,
        img_path,
        patch_shape,
        batch_size=16,
        is_multimodal=False,
        min_search_size=0,
        prefetch=128,
        segmentation_path=None,
        subgraph_radius=100,
        use_new_mask=False
    ):
        # Call parent class
        super().__init__(
            graph,
            img_path,
            patch_shape,
            batch_size=batch_size,
            is_multimodal=is_multimodal,
            min_search_size=min_search_size,
            prefetch=prefetch,
            segmentation_path=segmentation_path,
            subgraph_radius=subgraph_radius,
            use_new_mask=use_new_mask
        )

        # Instance attributes
        self.search_mode = "branching_points"

    def _generate_batches_from_component(self):
        pass

    def _generate_batch_nodes(self, root):
        nodes = list()
        patch_centers = list()
        for i, j in nx.dfs_edges(self.graph, source=root):
            # Check if starting new batch
            self.distance_traversed += self.graph.dist(i, j)
            if len(patch_centers) == 0 and self.graph.degree[i] > 2:
                root = i
                nodes.append(i)
                patch_centers.append(self.graph.get_voxel(i))

            # Check whether to yield batch
            is_node_far = self.graph.dist(root, j) > 256
            is_batch_full = len(patch_centers) == self.batch_size
            if is_node_far or is_batch_full:
                # Yield batch metadata
                patch_centers = np.array(patch_centers, dtype=int)
                nodes = np.array(nodes, dtype=int)
                yield nodes, patch_centers

                # Reset batch metadata
                nodes = list()
                patch_centers = list()

            # Visit j
            if self.graph.degree[j] > 2:
                nodes.append(j)
                patch_centers.append(self.graph.get_voxel(j))
                if len(patch_centers) == 1:
                    root = j

    # --- Helpers ---
    def estimate_iterations(self):
        """
        Estimates the number of iterations required to search graph.

        Returns
        -------
        int
            Estimated number of iterations required to search graph.
        """
        return len(self.graph.get_branchings())
