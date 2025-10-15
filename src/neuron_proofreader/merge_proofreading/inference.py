"""
Created on Wed August 4 16:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code for detecting merge mistakes on skeletons generated from an automated
image segmentation.

"""

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from torch.nn.functional import sigmoid
from torch.utils.data import IterableDataset
from time import time
from tqdm import tqdm

import networkx as nx
import numpy as np
import torch

from neuron_proofreader.utils import img_util, ml_util, util


class MergeDetector:

    def __init__(
        self,
        graph,
        img_path,
        model,
        model_path,
        patch_shape,
        anisotropy=(1.0, 1.0, 1.0),
        batch_size=32,
        device="cuda",
        min_size=0,
        prefetch=128,
        remove_detected_sites=False,
        threshold=0.4,
        step_size=10,
    ):
        # Instance attributes
        self.batch_size = batch_size
        self.device = device
        self.graph = graph
        self.patch_shape = patch_shape
        self.remove_detected_sites = remove_detected_sites
        self.step_size = step_size
        self.threshold = threshold

        # Load model
        self.model = model
        ml_util.load_model(model, model_path, device=self.device)

        # Initialize dataset
        self.dataset = IterableGraphDataset(
            graph,
            img_path,
            patch_shape,
            anisotropy=anisotropy,
            batch_size=batch_size,
            min_size=min_size,
            prefetch=prefetch,
            step_size=step_size,
        )

    # --- Core routines
    def search_graph(self):
        # Find fragment IDs to search
        approx_length = self.graph.number_of_edges() * self.graph.node_spacing
        pbar = tqdm(total=int(approx_length / self.step_size))
        self.graph.node_radius[:] = 1  # temp

        # Iterate over dataset
        t0 = time()
        merge_sites = list()
        confidences = list()
        for nodes, x_nodes in self.dataset:
            y_nodes = self.predict(x_nodes)
            idxs = np.where(y_nodes > self.threshold)[0]
            if len(idxs) > 0:
                merge_sites.extend(nodes[idxs].tolist())
                confidences.extend(y_nodes[idxs].tolist())

            # temp
            self.graph.node_radius[np.array(nodes)] = 10 * y_nodes
            pbar.update(self.batch_size)

        # Non-maximum suppression of detected sites
        merge_sites = self.filter_with_nms(merge_sites, confidences)
        rate = self.dataset.distance_traversed / (time() - t0)
        print("\n# Detected Merge Sites:", len(merge_sites))
        print(f"Distance Traversed: {self.dataset.distance_traversed:.2f}μm")
        print(f"Merge Proofreading Rate: {rate:.2f}μm/s")

        # Optionally, remove merge mistakes from graphs
        if self.remove_detected_sites:
            pass
        return merge_sites

    def predict(self, x_nodes):
        with torch.no_grad():
            x_nodes = x_nodes.to(self.device)
            y_nodes = sigmoid(self.model(x_nodes))
            return np.squeeze(ml_util.to_cpu(y_nodes, to_numpy=True), axis=1)

    def filter_with_nms(self, merge_sites, confidences):
        # Sort by confidence
        idxs = np.argsort(confidences)
        merge_sites = [merge_sites[i] for i in idxs]

        # NMS
        merge_sites_set = set(merge_sites)
        filtered_merge_sites = set()
        while merge_sites:
            # Local max
            root = merge_sites.pop()
            xyz_root = self.graph.node_xyz[root]
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
                    xyz_i = self.graph.node_xyz[i]
                    iou = img_util.compute_iou3d(
                        xyz_i, xyz_root, self.patch_shape, self.patch_shape
                    )
                    if iou > 0.3:
                        merge_sites_set.remove(i)
                        self.graph.node_radius[i] = 1

                # Populate queue
                for j in self.graph.neighbors(i):
                    dist_j = dist_i + self.graph.dist(i, j)
                    if j not in visited and dist_j < self.patch_shape[0]:
                        queue.append((j, dist_j))
                        visited.add(j)
        return filtered_merge_sites

    def remove_merge_sites(self, detected_merge_sites):
        pass


# --- Data Handling ---
class IterableGraphDataset(IterableDataset):

    def __init__(
        self,
        graph,
        img_path,
        patch_shape,
        anisotropy=(1.0, 1.0, 1.0),
        batch_size=16,
        min_size=0,
        prefetch=128,
        step_size=10,
    ):
        # Call parent class
        super().__init__()

        # Instance attributes
        self.anisotropy = anisotropy
        self.batch_size = batch_size
        self.distance_traversed = 0
        self.graph = graph
        self.min_size = min_size
        self.patch_shape = patch_shape
        self.prefetch = prefetch
        self.step_size = step_size

        # Image reader
        self.img_reader = img_util.init_reader(img_path)

    # --- Core routines ---
    def __iter__(self):
        # Subroutines
        def submit_thread():
            try:
                nodes, patch_centers = next(batch_metadata_iter)
                thread = executor.submit(self.read_superchunk, patch_centers)
                pending[thread] = (nodes, patch_centers)
            except StopIteration:
                pass

        # Main
        batch_metadata_iter = self.generate_batch_metadata()
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
                        nodes, patch_centers = pending.pop(thread)
                        img, offset = thread.result()
                        yield self.get_batch(img, offset, patch_centers, nodes)

                        # Continue submitting threads
                        submit_thread()
            finally:
                pass

    def generate_batch_metadata(self):
        """
        Generates metadata (nodes, patch_centers) used to generate batches.

        Parameters
        ----------
        None

        Returns
        -------
        iterator
            Generator that yields node IDs and patch centers used to generate
            batches across the whole graph.
        """
        # Find fragment IDs to check
        valid_ids = self.find_fragments_to_search()

        # Search graph
        visited_ids = set()
        for i in self.graph.get_leafs():
            component_id = self.graph.node_component_id[i]
            if component_id not in visited_ids and component_id in valid_ids:
                visited_ids.add(component_id)
                yield from self._generate_batch_metadata_for_component(i)

    def _generate_batch_metadata_for_component(self, root):
        """
        Generates metadata (nodes, patch_centers) used to generate batches
        for the connected component containing the given root node.

        Returns
        -------
        iterator
            Generator that yields node IDs and patch centers used to generate
            batches for the connected component containing the given root
            node.
        """
        nodes = list()
        patch_centers = list()
        visited = set()
        for i, j in nx.dfs_edges(self.graph, source=root):
            # Check if starting new batch
            self.distance_traversed += self.graph.dist(i, j)
            if len(patch_centers) == 0:
                root = i
                last_node = i
                nodes.append(i)
                patch_centers.append(self.get_voxel(i))
                visited.add(i)

            # Check whether to yield batch
            is_node_far = self.graph.dist(root, j) > 512
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
            if j not in visited:
                visited.add(j)
                is_next = self.graph.dist(last_node, j) >= self.step_size - 1
                is_branching = self.graph.degree[j] >= 3
                if is_next or is_branching:
                    last_node = j
                    nodes.append(j)
                    patch_centers.append(self.get_voxel(j))
                    if len(patch_centers) == 1:
                        root = j

        # Yield any remaining nodes after the loop
        if patch_centers:
            patch_centers = np.array(patch_centers, dtype=int)
            nodes = np.array(nodes, dtype=int)
            yield nodes, patch_centers

    def get_batch(self, img, offset, patch_centers, nodes):
        # Initializations
        label_mask = self.get_label_mask(nodes, img.shape, offset)
        patch_centers -= offset

        # Populate batch array
        batch = np.empty((len(patch_centers), 2,) + self.patch_shape)
        for i, center in enumerate(patch_centers):
            s = img_util.get_slices(center, self.patch_shape)
            batch[i, 0, ...] = img[s]
            batch[i, 1, ...] = label_mask[s]

        # Normalize image
        mn, mx = np.percentile(batch[0, 0, ...], [1, 99.9])
        batch[:, 0, ...] = np.clip((batch[:, 0, ...] - mn) / (mx - mn), 0, 1)
        return nodes, torch.tensor(batch, dtype=torch.float)

    # --- Helpers ---
    def find_fragments_to_search(self):
        component_ids = set()
        for nodes in nx.connected_components(self.graph):
            # Compute path length
            node = util.sample_once(list(nodes))
            l = self.graph.path_length(root=node, max_depth=self.min_size)

            # Check if path length satisfies threshold
            if l > self.min_size:
                component_ids.add(self.graph.node_component_id[node])
        return component_ids

    def read_superchunk(self, patch_centers):
        # Compute bounding box
        buffer = np.array(self.patch_shape) / 2
        start = patch_centers.min(axis=0) - buffer
        end = patch_centers.max(axis=0) + buffer + 1

        # Read image
        shape = (end - start).astype(int)
        center = (start + shape // 2).astype(int)
        superchunk = self.img_reader.read(center, shape)
        return superchunk, start.astype(int)

    def get_label_mask(self, nodes, img_shape, offset):
        label_mask = np.zeros(img_shape)
        queue = list(nodes)
        visited = set(nodes)
        while queue:
            # Visit node
            node = queue.pop()
            voxel = self.get_voxel(node) - offset
            is_contained = img_util.is_contained(voxel, img_shape, buffer=3)
            if is_contained:
                label_mask[
                    voxel[0] - 3: voxel[0] + 3,
                    voxel[1] - 3: voxel[1] + 3,
                    voxel[2] - 3: voxel[2] + 3
                ] = 1
            # Update queue
            if is_contained:
                for nb in self.graph.neighbors(node):
                    if nb not in visited:
                        queue.append(nb)
                        visited.add(nb)
        return label_mask

    def get_voxel(self, node):
        """
        Gets the voxel coordinate of the given node.

        Parameters
        ----------
        node : int
            Node ID.

        Returns
        -------
        Tuple[int]
            Voxel coordinate of the given node.
        """
        xyz = self.graph.node_xyz[node]
        return img_util.to_voxels(xyz, self.anisotropy)
