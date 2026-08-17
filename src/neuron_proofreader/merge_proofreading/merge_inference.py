"""
Created on Sun Aug 3 16:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Merge proofreading classes. MergeProofreader is an abstract base that defines
a search / remove template: each subclass implements search() to identify merge
site nodes, and the base __call__ derives xyz coordinates and removes them.

    Subclasses
    ----------
    MLMergeProofreader
        CNN-based detection; overrides __call__ to add saving and optional
        removal.
    HighRiskMergeProofreader
        Heuristic; detects closely-packed branching nodes.
    SomaMergeProofreader
        Heuristic; detects merge errors on soma-connecting paths.

"""

from abc import ABC, abstractmethod
from collections import defaultdict, deque
from copy import deepcopy
from scipy.spatial import KDTree
from time import time
from torch.nn.functional import sigmoid
from torch.utils.data import DataLoader
from tqdm import tqdm

import networkx as nx
import numpy as np
import os
import pandas as pd
import torch

from arborist.utils.swc_loading import to_zipped_points
from neuron_proofreader.merge_proofreading.search_datasets import (
    DenseSearchDataset,
    SparseSearchDataset,
    multimodal_collate,
)
from neuron_proofreader.utils import img_util, util


class MergeProofreader(ABC):
    """
    Abstract base class for merge proofreading. Subclasses implement search()
    to identify merge site representative nodes; this class removes them and
    derives xyz coordinates.
    """

    def __init__(self, graph, output_dir, log_handle=None):
        self.graph = graph
        self.output_dir = output_dir
        log_path = os.path.join(output_dir, "summary.txt")
        self.log_handle = log_handle or open(log_path, "a")

    @abstractmethod
    def search(self):
        """
        Identifies merge site nodes.

        Returns
        -------
        List[int]
            Node IDs of detected merge site representatives.
        """
        pass

    def __call__(self):
        """
        Runs search, removes detected merge sites, and returns their physical
        coordinates.

        Returns
        -------
        list
            Physical coordinates (xyz) of detected merge sites.
        """
        merge_nodes = self.search()
        self.graph.remove_merge_sites(merge_nodes)
        return merge_nodes

    def save_sites(self, merge_xyz_list):
        """
        Saves detected merge site coordinates to a zip file.

        Parameters
        ----------
        merge_xyz_list : list
            Physical coordinates of detected merge sites.
        """
        zip_path = os.path.join(self.output_dir, "detected_sites.zip")
        to_zipped_points(
            zip_path,
            merge_xyz_list,
            color="1.0 0.0 0.0",
            prefix="merge-site",
            radius=10,
        )
        print("# Sites Saved:", len(merge_xyz_list))

    def save_parameters(self):
        pass

    def log(self, txt):
        print(txt)
        self.log_handle.write(txt + "\n")


class MLMergeProofreader(MergeProofreader):
    """
    CNN-based merge proofreader. Scores skeleton nodes with a trained model,
    applies graph-aware NMS and spatial averaging to consolidate detections,
    and optionally removes detected sites.
    """

    step_name = "learned_merge_correction"

    def __init__(
        self,
        graph,
        model,
        img_config,
        output_dir,
        mode="dense",
        batch_size=16,
        device="cuda",
        modality="image",
        save_result=True,
        min_search_size=0,
        prefetch=64,
        threshold=0.5,
        log_handle=None,
    ):
        """
        Initializes an MLMergeProofreader.

        Parameters
        ----------
        graph : FragmentsGraph
            Skeleton graph to search for merge errors.
        model : torch.nn.Module
            Trained model used to score candidate merge sites.
        img_config : ImageConfig
            Config object that contains parameters for processing images.
        output_dir : str
            Directory where results of the inference will be saved.
        mode : str, optional
            Search strategy. "dense" scores every node along each fragment;
            "sparse" restricts scoring to branching nodes. Default is "dense".
        batch_size : int, optional
            Number of patches per forward pass. Default is 16.
        device : str, optional
            Device on which to run inference. Default is "cuda".
        save_result : bool, optional
            If True, saves detection results to output_dir. Default is True.
        min_search_size : float, optional
            Minimum fragment cable length (in microns) to include in the
            search. Default is 0.
        prefetch : int, optional
            Number of patches to prefetch. Default is 64.
        threshold : float, optional
            Confidence threshold above which a site is flagged as a merge.
            Default is 0.5.
        log_handle : file-like, optional
            Open file handle to write log messages to. If None, a new
            summary.txt is opened in output_dir. Default is None.
        """
        # Call parent class
        super().__init__(graph, output_dir, log_handle)

        # Create search dataset
        DatasetClass = (
            DenseSearchDataset if mode == "dense" else SparseSearchDataset
        )
        self.dataset = DatasetClass(
            graph,
            img_config,
            modality=modality,
            min_search_size=min_search_size,
            prefetch=prefetch,
        )

        # Instance attributes
        self.batch_size = batch_size
        self.device = device
        self.modality = modality
        self.model = model
        self.mode = mode
        self.save_result = save_result
        self.node_preds = np.zeros((len(graph.node_xyz)))
        self.patch_shape = self.dataset.patch_shape
        self.threshold = threshold
        self.visited_sites = list()
        self.merge_sites_xyz = list()

    # --- Core ---
    def __call__(self):
        # Search graph
        t0 = time()
        self.log("Search Graph...")
        merge_sites = self.search()

        # Cache XYZ coords and save predictions before graph is modified
        self.merge_sites_xyz = [
            self.graph.node_xyz[i].tolist() for i in merge_sites
        ]
        if self.save_result:
            self.save_predictions()

        self.graph.remove_merge_sites(merge_sites)

        # Report results
        t, unit = util.time_writer(time() - t0)
        self.log(f"# Detected Merges: {len(merge_sites)}")
        self.log(f"Module Runtime: {t:.2f} {unit}\n")
        if self.save_result:
            self.save_fragment_predictions(inplace=False)
            self.save_parameters()

        return merge_sites

    def search(self):
        # Detect merge errors with classification
        t0 = time()
        self.model.eval()
        collate_fn = multimodal_collate if self.modality != "image" else None
        dataloader = DataLoader(
            self.dataset, batch_size=self.batch_size, collate_fn=collate_fn
        )
        pbar = tqdm(total=self.dataset.estimate_iterations())
        for nodes, x_nodes in dataloader:
            self.node_preds[np.array(nodes)] = self.predict(x_nodes)
            self.visited_sites.extend(nodes.tolist())
            pbar.update(len(nodes))
        pbar.close()

        # Filter merge errors with NMS
        merge_sites = np.where(self.node_preds >= self.threshold)[0]
        likelihoods = self.node_preds[merge_sites]
        merge_sites = self.apply_graph_nms(merge_sites, likelihoods)

        # Filter merge errors via spatial averaging
        while True:
            before = len(merge_sites)
            merge_sites = self.avg_nearby_sites(merge_sites)
            if before == len(merge_sites):
                break

        # Report results
        rate = len(self.visited_sites) / (time() - t0)
        print("\n# Detected Merges:", len(merge_sites))
        print(f"Proofreading Rate: {rate:.2f} sites/s")
        return merge_sites

    def predict(self, x):
        """
        Predicts merge site likelihoods for the given node features.

        Parameters
        ----------
        x : torch.Tensor or dict
            For unimodal models: tensor of shape (N, 2, M, M, M).
            For multimodal models: dict with keys "img" (tensor) and
            "tree_sample" (List[TreeSample]).

        Returns
        -------
        numpy.ndarray
            Predicted merge site likelihoods.
        """
        with torch.inference_mode():
            if isinstance(x, dict):
                x = {
                    k: v.to(self.device) if torch.is_tensor(v) else v
                    for k, v in x.items()
                }
            else:
                x = x.to(self.device)
            y = sigmoid(self.model(x))
            y = y.detach().cpu().numpy()
            return np.squeeze(y, axis=1)

    def apply_graph_nms(self, merge_sites, likelihoods):
        # Create merge site data structures
        merge_sites = [merge_sites[i] for i in np.argsort(likelihoods)]
        merge_sites = deque(merge_sites)
        merge_sites_set = set(merge_sites)

        # Filter merge sites with graph-based NMS
        filtered_merge_sites = set()
        while merge_sites:
            root = merge_sites.pop()
            xyz_root = self.graph.node_xyz[root]
            if root in merge_sites_set:
                filtered_merge_sites.add(root)
                merge_sites_set.remove(root)
            else:
                continue

            # Search nbhd of merge site
            queue = [(root, 0)]
            visited = {root}
            while queue:
                # Visit node
                i, dist_i = queue.pop()
                if i in merge_sites_set:
                    xyz_i = self.graph.node_xyz[i]
                    iou = img_util.compute_iou3d(
                        xyz_i, xyz_root, self.patch_shape, self.patch_shape
                    )
                    if iou > 0.3 and self.graph.degree[i] == 2:
                        merge_sites_set.remove(i)
                        self.node_preds[i] = 0

                # Populate queue
                for j in self.graph.neighbors(i):
                    dist_j = dist_i + self.graph.dist(i, j)
                    if j not in visited and dist_j < self.patch_shape[0]:
                        queue.append((j, dist_j))
                        visited.add(j)
        return filtered_merge_sites

    def avg_nearby_sites(self, merge_sites, max_dist=24):
        # Create merge site data structures
        merge_sites = list(merge_sites)
        likelihoods = [self.node_preds[i] for i in merge_sites]
        merge_sites = [merge_sites[i] for i in np.argsort(likelihoods)[::-1]]

        # Search for nearby merge sites
        visited = set()
        new_merge_sites = list()
        sites_kdtree = KDTree([self.graph.node_xyz[i] for i in merge_sites])
        for root in merge_sites:
            # Check if node is already visited
            if root in visited:
                continue
            visited.add(root)

            # Find nearby sites
            xyz_query = self.graph.node_xyz[root]
            idxs = sites_kdtree.query_ball_point(xyz_query, max_dist)
            nodes = [merge_sites[i] for i in idxs]

            # Check whether to average sites in a single one
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

                xyz_arr = np.array([self.graph.node_xyz[i] for i in hits])
                xyz_avg = xyz_arr.mean(axis=0)
                best_node = min(
                    hits,
                    key=lambda n: np.linalg.norm(
                        self.graph.node_xyz[n] - xyz_avg
                    ),
                )
                new_merge_sites.append(best_node)

                likelihood = self.node_preds[root]
                for node in hits:
                    if node != best_node:
                        self.node_preds[node] = 0
                self.node_preds[best_node] = likelihood
            else:
                new_merge_sites.append(root)
        return new_merge_sites

    # --- Save ---
    def save(self, inplace=True):
        self.save_fragment_predictions(inplace=inplace)
        self.save_parameters()
        self.save_predictions()
        nodes = np.where(self.node_preds >= self.threshold)[0]
        self.save_sites([self.graph.node_xyz[i] for i in nodes])

    def save_fragment_predictions(self, inplace=True):
        fragments_path = os.path.join(
            self.output_dir, "fragment_merge_preds.zip"
        )
        if inplace:
            self.graph.node_radius = 10 * np.maximum(self.node_preds, 0.1)
            self.dataset.to_zipped_swcs(fragments_path, use_radius=True)
        else:
            graph = deepcopy(self.graph)
            graph.node_radius = 10 * np.maximum(self.node_preds, 0.1)
            graph.to_zipped_swcs(fragments_path, use_radius=True)

    def save_parameters(self):
        json_path = os.path.join(self.output_dir, "detection_parameters.json")
        parameters = {
            "accept_threshold": self.threshold,
            "modality": self.dataset.modality,
            "min_search_size": self.dataset.min_size,
            "patch_shape": self.patch_shape,
            "search_mode": self.dataset.search_mode,
            "subgraph_radius": self.dataset.subgraph_radius,
        }
        util.write_json(json_path, parameters)

    def save_predictions(self):
        nodes = np.array(self.visited_sites, dtype=int)
        df = pd.DataFrame(
            columns=["xyz", "Segment_ID", "Prediction", "Degree"]
        )
        df["xyz"] = list(map(tuple, self.graph.node_xyz[nodes]))
        df["Prediction"] = self.node_preds[nodes]
        df["Segment_ID"] = [self.dataset.node_segment_id(i) for i in nodes]
        df["Degree"] = [self.graph.degree[i] for i in nodes]
        df.to_csv(os.path.join(self.output_dir, "model_predictions.csv"))

    def save_train_dataset(self):
        roots = list()
        visited_ids = set()
        for i in np.where(self.node_preds >= self.threshold)[0]:
            cc_id = self.graph.node_component_id[i]
            if cc_id not in visited_ids:
                roots.append([i])
                visited_ids.add(cc_id)

        zip_path = os.path.join(self.output_dir, "fragments.zip")
        self.dataset._batch_to_zipped_swcs(roots, zip_path, False)
        nodes = np.where(self.node_preds >= self.threshold)[0]
        self.save_sites([self.graph.node_xyz[i] for i in nodes])
        print("# Fragments Saved:", len(roots))

    # --- Helpers ---
    def get_detected_sites(self, threshold):
        nodes = np.where(self.node_preds >= threshold)[0]
        return [self.graph.node_xyz[i] for i in nodes]


class HighRiskMergeProofreader(MergeProofreader):
    """
    Heuristic merge proofreader that detects closely-packed branching nodes
    and branching nodes with degree >= 4, both indicators of merge errors.
    """

    step_name = "heuristic_merge_correction"

    def __init__(self, graph, output_dir, max_dist=7, log_handle=None):
        """
        Initializes a HighRiskMergeProofreader.

        Parameters
        ----------
        graph : FragmentsGraph
            Skeleton graph to search for high-risk merge sites.
        output_dir : str
            Directory where results will be saved.
        max_dist : float, optional
            Maximum distance (in microns) between branching nodes that
            qualifies as high-risk. Default is 7.
        log_handle : file-like, optional
            Open file handle to write log messages to. Default is None.
        """
        super().__init__(graph, output_dir, log_handle)
        self.max_dist = max_dist

    def search(self):
        # Initializations
        branching_nodes = set(self.graph.branching_nodes())
        soma_nodes = np.array(self.graph.soma_nodes(), dtype=int)
        if len(soma_nodes) > 0:
            somas_kdtree = KDTree(self.graph.node_xyz[soma_nodes])

        # Search branching nodes
        merge_nodes = list()
        while branching_nodes:
            # Check if too close to soma
            root = branching_nodes.pop()
            if len(soma_nodes):
                dist, _ = somas_kdtree.query(self.graph.node_xyz[root])
                if dist < 300:
                    continue

            # Traverse nbhd
            hit_branching_nodes = set()
            queue = [(root, 0)]
            visited = {root}
            while queue:
                i, dist_i = queue.pop()
                if self.graph.degree[i] > 2 and i != root:
                    hit_branching_nodes.add(i)
                for j in self.graph.neighbors(i):
                    dist_j = dist_i + self.graph.dist(i, j)
                    if j not in visited and dist_j < self.max_dist:
                        queue.append((j, dist_j))
                        visited.add(j)

            # Determine if likely merge and remove nbhd
            if hit_branching_nodes or self.graph.degree(root) > 3:
                merge_nodes.append(root)
                branching_nodes -= hit_branching_nodes

        return merge_nodes


class SomaMergeProofreader(MergeProofreader):
    """
    Heuristic merge proofreader that detects merge errors on paths connecting
    multiple somas within the same connected component.
    """

    step_name = "somas_merge_correction"

    def __init__(self, graph, output_dir, log_handle=None):
        """
        Initializes a SomaMergeProofreader.

        Parameters
        ----------
        graph : FragmentsGraph
            Skeleton graph to search for soma merge errors.
        output_dir : str
            Directory where results will be saved.
        log_handle : file-like, optional
            Open file handle to write log messages to. Default is None.
        """
        super().__init__(graph, output_dir, log_handle)

    def search(self):
        if len(self.graph.soma_centroids) == 0:
            return list()

        component_id_to_soma_nodes = defaultdict(set)
        somas_kdtree = KDTree(self.graph.soma_centroids)
        for i in self.graph.soma_nodes():
            component_id_to_soma_nodes[self.graph.node_component_id[i]].add(i)

        merge_nodes = list()
        for soma_nodes in component_id_to_soma_nodes.values():
            if 1 < len(soma_nodes) < 20:
                for i in self.graph.find_connecting_path(list(soma_nodes)):
                    dist, _ = somas_kdtree.query(self.graph.node_xyz[i])
                    if self.graph.degree[i] > 2 and dist > 25:
                        merge_nodes.append(i)

        return merge_nodes

    def __call__(self):
        merge_nodes = super().__call__()
        self.graph.remove_small_components()
        return merge_nodes
