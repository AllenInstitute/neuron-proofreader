"""
Created on Wed June 15 16:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code for detecting merge mistakes on skeletons generated from an automated
image segmentation.

"""

from collections import deque
from copy import deepcopy
from scipy.spatial import KDTree
from torch.nn.functional import sigmoid
from torch.utils.data import DataLoader
from time import time
from tqdm import tqdm

import networkx as nx
import numpy as np
import os
import pandas as pd
import torch

from neuron_proofreader.utils import img_util, ml_util, swc_util, util


class MergeDetector:

    def __init__(
        self,
        dataset,
        model,
        batch_size=16,
        device="cuda",
        threshold=0.5,
    ):
        # Instance attributes
        self.batch_size = batch_size
        self.dataset = dataset
        self.device = device
        self.node_preds = np.zeros((len(dataset.node_xyz)))
        self.patch_shape = dataset.patch_shape
        self.visited_sites = list()
        self.threshold = threshold

        # Load model
        self.model = model

    # --- Core routines ---
    def search_graph(self):
        # Iterate over dataset
        t0 = time()
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size)
        pbar = tqdm(total=self.dataset.estimate_iterations())
        for nodes, x_nodes in dataloader:
            self.node_preds[np.array(nodes)] = self.predict(x_nodes)
            self.visited_sites.extend(nodes.tolist())
            pbar.update(len(nodes))
        pbar.close()

        # Non-maximum suppression of detected sites
        merge_sites = np.where(self.node_preds > self.threshold)[0]
        likelihoods = self.node_preds[merge_sites]
        merge_sites = self.apply_graph_nms(merge_sites, likelihoods)

        # Iteratively average nearby sites
        while True:
            before = len(merge_sites)
            merge_sites = self.avg_nearby_sites(merge_sites)
            if before == len(merge_sites):
                break

        # Report results
        rate = len(self.visited_sites) / (time() - t0)
        print("\n# Detected Merges:", len(merge_sites))
        print(f"Proofreading Rate: {rate:.2f} site/s")
        return merge_sites

    def predict(self, x):
        """
        Predicts merge site likelihoods for the given node features.

        Parameters
        ----------
        x : torch.Tensor
            Node features with shape Nx2xMxMxM, where N is the number of nodes
            and MxMxM is the patch shape.

        Returns
        -------
        y : numpy.ndarray
            Predicted merge site likelihoods.
        """
        self.model.eval()
        with torch.inference_mode():
            x = x.to(self.device)
            y = sigmoid(self.model(x))
            y = y.detach().cpu().numpy()
            return np.squeeze(y, axis=1)

    def apply_graph_nms(self, merge_sites, likelihoods):
        # Sort by confidence
        merge_sites = [merge_sites[i] for i in np.argsort(likelihoods)[::-1]]
        merge_sites = deque(merge_sites)

        # NMS
        merge_sites_set = set(merge_sites)
        filtered_merge_sites = set()
        while merge_sites:
            # Local max
            root = merge_sites.popleft()
            xyz_root = self.dataset.node_xyz[root]
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
                    xyz_i = self.dataset.node_xyz[i]
                    iou = img_util.compute_iou3d(
                        xyz_i, xyz_root, self.patch_shape, self.patch_shape
                    )
                    if iou > 0.3 and self.dataset.degree[i] == 2:
                        merge_sites_set.remove(i)
                        self.node_preds[i] = 0

                # Populate queue
                for j in self.dataset.neighbors(i):
                    dist_j = dist_i + self.dataset.dist(i, j)
                    if j not in visited and dist_j < self.patch_shape[0]:
                        queue.append((j, dist_j))
                        visited.add(j)
        return filtered_merge_sites

    def avg_nearby_sites(self, merge_sites, max_dist=24):
        # Sort sites by likelihood
        merge_sites = list(merge_sites)
        likelihoods = [self.node_preds[i] for i in merge_sites]
        merge_sites = [merge_sites[i] for i in np.argsort(likelihoods)[::-1]]

        # Search for spatially nearby sites
        visited = set()
        new_merge_sites = list()
        sites_kdtree = KDTree([self.dataset.node_xyz[i] for i in merge_sites])
        for root in merge_sites:
            # Check whether to skip
            if root in visited:
                continue
            else:
                visited.add(root)

            # Get nearby sites
            xyz_query = self.dataset.node_xyz[root]
            idxs = sites_kdtree.query_ball_point(xyz_query, max_dist)
            nodes = [merge_sites[i] for i in idxs]

            # Check whether to combine sites
            if len(nodes) > 1:
                hits = list()
                for node in nodes:
                    try:
                        path = nx.shortest_path(
                            self.dataset.graph, source=root, target=node
                        )
                        if self.dataset.path_length(path) < max_dist + 4:
                            hits.append(node)
                            visited.add(node)
                    except nx.exception.NetworkXNoPath:
                        pass
    
                # Add site to list
                xyz_arr = np.array([self.dataset.node_xyz[i] for i in hits])
                xyz_avg = xyz_arr.mean(axis=0)
                best_node = min(
                    hits,
                    key=lambda n: np.linalg.norm(self.dataset.node_xyz[n] - xyz_avg)
                )
                new_merge_sites.append(best_node)

                # Update node predictions
                likelihood = self.node_preds[root]
                for node in hits:
                    if node != best_node:
                        self.node_preds[node] = 0
                self.node_preds[best_node] = likelihood
            else:
                new_merge_sites.append(root)
        return new_merge_sites

    def remove_merge_sites(self, merge_site_nodes, max_depth=10):
        rm_nodes = set()
        for root in tqdm(merge_site_nodes, desc="Remove Merge Sites"):
            # Extract neighborhood
            root = self.dataset.find_nearby_branching_node(root)
            nbhd = self.dataset.nodes_within_distance(root, max_depth)

            # Check for branching node in neighborhood
            for i in list(nbhd):
                if i != root and self.dataset.degree[i] >= 3:
                    nbhd_i = self.dataset.nodes_within_distance(root, 8)
                    nbhd.extend(nbhd_i)

            # Add nodes to removal list
            rm_nodes.update(set(nbhd))

        # Update graph
        self.dataset.remove_nodes(rm_nodes)
        print("# Nodes Deleted:", len(rm_nodes))

    # --- Save Results ---
    def save(self, output_dir, inplace=True):
        self.save_fragment_predictions(output_dir, inplace=inplace)
        self.save_parameters(output_dir)
        self.save_predictions(output_dir)
        self.save_sites(output_dir)

    def save_fragment_predictions(self, output_dir, inplace=True):
        fragments_path = os.path.join(output_dir, "fragment_preds.zip")
        if inplace:
            self.dataset.node_radius = 10 * np.maximum(self.node_preds, 0.1)
            self.dataset.to_zipped_swcs(fragments_path, use_radius=True)
        else:
            graph = deepcopy(self.dataset.graph)
            graph.node_radius = 10 * np.maximum(self.node_preds, 0.1)
            graph.to_zipped_swcs(fragments_path, use_radius=True)

    def save_parameters(self, output_dir):
        json_path = os.path.join(output_dir, "detection_parameters.json")
        parameters = {
            "accept_threshold": self.threshold,
            "is_multimodal": self.dataset.is_multimodal,
            "min_search_size": self.dataset.min_size,
            "patch_shape": self.patch_shape,
            "search_mode": self.dataset.search_mode,
            "subgraph_radius": self.dataset.subgraph_radius,
        }
        util.write_json(json_path, parameters)

    def save_predictions(self, output_dir):
        nodes = np.array(self.visited_sites, dtype=int)
        df = pd.DataFrame(
            columns=["xyz", "Segment_ID", "Prediction", "Degree"]
        )
        df["xyz"] = list(map(tuple, self.dataset.node_xyz[nodes]))
        df["Prediction"] = self.node_preds[nodes]
        df["Segment_ID"] = [self.dataset.node_segment_id(i) for i in nodes]
        df["Degree"] = [self.dataset.degree[i] for i in nodes]
        df.to_csv(os.path.join(output_dir, "model_predictions.csv"))

    def save_sites(self, output_dir):
        # Get predicted merge sites
        nodes = np.where(self.node_preds >= self.threshold)[0]
        detected_sites = [self.dataset.node_xyz[i] for i in nodes]
        print("# Sites Saved:", len(nodes))

        # Save predicted merge sites
        zip_path = os.path.join(output_dir, "detected_sites.zip")
        swc_util.write_points(
            zip_path,
            detected_sites,
            color="1.0 0.0 0.0",
            prefix="merge-site",
            radius=10,
        )

    def save_train_dataset(self, output_dir):
        # Extract fragments to save
        roots = list()
        visited_ids = set()
        for i in np.where(self.node_preds >= self.threshold)[0]:
            cc_id = self.dataset.node_component_id[i]
            if cc_id not in visited_ids:
                roots.append([i])
                visited_ids.add(cc_id)

        # Save fragments
        zip_path = os.path.join(output_dir, "fragments.zip")
        self.dataset._batch_to_zipped_swcs(roots, zip_path, False)
        self.save_sites(output_dir)
        print("# Fragments Saved:", len(roots))

    # --- Helpers ---
    def get_detected_sites(self, threshold):
        nodes = np.where(self.node_preds >= threshold)[0]
        return [self.dataset.node_xyz[i] for i in nodes]
