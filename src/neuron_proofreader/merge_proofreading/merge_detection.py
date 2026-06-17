"""
Created on Wed June 15 16:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code for detecting merge mistakes on skeletons generated from an automated
image segmentation.

"""

from torch.nn.functional import sigmoid
from torch.utils.data import DataLoader
from time import time
from tqdm import tqdm

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
        model_path,
        batch_size=16,
        device="cuda",
        remove_detected_sites=False,
        threshold=0.5,
    ):
        # Instance attributes
        self.batch_size = batch_size
        self.dataset = dataset
        self.device = device
        self.node_preds = np.zeros((len(dataset.node_xyz)))
        self.patch_shape = dataset.patch_shape
        self.remove_detected_sites = remove_detected_sites
        self.threshold = threshold

        # Load model
        self.model = model
        ml_util.load_model(model, model_path, device=self.device)

    # --- Core routines ---
    def search_graph(self):
        # Iterate over dataset
        t0 = time()
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size)
        pbar = tqdm(total=self.dataset.estimate_iterations())
        for nodes, x_nodes in dataloader:
            self.node_preds[np.array(nodes)] = self.predict(x_nodes)
            pbar.update(len(nodes))

        # Non-maximum suppression of detected sites
        merge_sites = np.where(self.node_preds > self.threshold)[0]
        likelihoods = self.node_preds[merge_sites]
        merge_sites = self.filter_with_nms(merge_sites, likelihoods)

        # Report results
        rate = self.dataset.distance_traversed / (time() - t0)
        print("\n# Detected Merge Sites:", len(merge_sites))
        print(f"Distance Traversed: {self.dataset.distance_traversed:.2f}μm")
        print(f"Merge Proofreading Rate: {rate:.2f}μm/s")

        # Remove merge mistakes (optional)
        if self.remove_detected_sites:
            pass
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
        numpy.ndarray
            Predicted merge site likelihoods.
        """
        with torch.inference_mode():
            x = x.to(self.device)
            y = sigmoid(self.model(x))
            return np.squeeze(ml_util.to_cpu(y, to_numpy=True), axis=1)

    def filter_with_nms(self, merge_sites, likelihoods):
        # Sort by confidence
        merge_sites = [merge_sites[i] for i in np.argsort(likelihoods)]

        # NMS
        merge_sites_set = set(merge_sites)
        filtered_merge_sites = set()
        while merge_sites:
            # Local max
            root = merge_sites.pop()
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
                    if iou > 0.35:
                        merge_sites_set.remove(i)
                        self.node_preds[i] = 0

                # Populate queue
                for j in self.dataset.neighbors(i):
                    dist_j = dist_i + self.dataset.dist(i, j)
                    if j not in visited and dist_j < self.patch_shape[0]:
                        queue.append((j, dist_j))
                        visited.add(j)
        return filtered_merge_sites

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

    # --- Helpers ---
    def get_detected_sites(self, threshold):
        nodes = np.where(self.node_preds >= threshold)[0]
        return [self.dataset.node_xyz[i] for i in nodes]

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

    def save_results(
        self, output_dir, output_prefix_s3=None, save_fragments=True
    ):
        self.save_sites(output_dir)
        if save_fragments:
            self.dataset.graph.node_radius = 10 * np.maximum(
                self.node_preds, 0.1
            )
            fragments_path = os.path.join(output_dir, "fragments.zip")
            self.dataset.to_zipped_swcs(fragments_path, use_radius=True)

        # Upload results to S3 (if applicable)
        if output_prefix_s3:
            bucket_name, prefix = util.parse_cloud_path(output_prefix_s3)
            util.upload_dir_to_s3(output_dir, bucket_name, prefix)

    def save_sites(self, output_dir):
        # Save model predictions
        df = pd.DataFrame(
            columns=["World", "Segment_ID", "Prediction", "Degree"]
        )
        df["World"] = list(map(tuple, self.dataset.node_xyz))
        df["Prediction"] = self.node_preds
        df["Segment_ID"] = [
            self.dataset.node_segment_id(i) for i in self.dataset.nodes
        ]
        df["Degree"] = [self.dataset.degree[i] for i in self.dataset.nodes]
        df.to_csv(os.path.join(output_dir, "model_predictions.csv"))

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
