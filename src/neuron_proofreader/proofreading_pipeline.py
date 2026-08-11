"""
Created on Fri June 13 16:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code for running full neuron proofreading pipeline, including both split and
merge detection and correction.

"""

from copy import copy
from time import time

import numpy as np
import os

from neuron_proofreader.merge_proofreading.merge_inference import (
    HighRiskMergeProofreader,
    MLMergeProofreader,
    SomaMergeProofreader,
)
from neuron_proofreader.proposal_graph import ProposalGraph
from neuron_proofreader.split_proofreading.split_inference import (
    SplitProofreader,
)
from neuron_proofreader.utils import geometry_util, util


class ProofreadPipeline:

    def __init__(
        self,
        swcs_path,
        graph_config,
        img_config,
        output_dir,
        device="cuda",
        log_preamble="",
        soma_centroids=list(),
    ):
        """
        Initializes an object that executes the full split proofreading
        pipeline.

        Parameters
        ----------
        swcs_path : str
            Path to SWC files to be loaded into graph.
        graph_config : GraphConfig
            Config object that contains parameters for building graph.
        img_config : ImageConfig
            Config object that contains parameters for processing images.
        output_dir : str
            Directory where the results of the inference will be saved.
        log_preamble : str, optional
            String to be added to the beginning of log. Default is an empty
            string.
        soma_centroids : List[Tuple[float]], optional
            Physical coordinates of soma centroids. Default is an empty list.
        """
        # Instance attributes
        self.device = device
        self.img_config = img_config
        self.output_dir = output_dir
        self.step_cnt = 0

        # Logger
        util.mkdir(self.output_dir)
        log_path = os.path.join(self.output_dir, "summary.txt")
        self.log_handle = open(log_path, "a")
        self.log(log_preamble)

        # Load data
        self.load_graph(graph_config, swcs_path, soma_centroids)

    def load_graph(self, config, swcs_path, soma_centroids):
        """
        Loads a graph from the given fragments.

        Parameters
        ----------
        swcs_path : str
            Path to SWC files to be loaded into graph.
        config : GraphConfig
            Configuration object that contains parameters for building graph.
        """
        # Load data
        t0 = time()
        self.log("Build Graph")
        self.graph = ProposalGraph(
            anisotropy=config.anisotropy,
            min_cable_length=config.min_cable_length,
            node_spacing=config.node_spacing,
            verbose=config.verbose,
        )
        self.graph.load(swcs_path)
        self.graph.load_somas(soma_centroids)

        # Remove doubled fragments
        if config.remove_doubles:
            geometry_util.remove_doubles(self.graph, 200)

        # Save original graph state
        self.save_graph("original_swcs")
        self.log("\nInitial Graph...")
        self.log(self.graph.__repr__())

        # Report runtime
        elapsed, unit = util.time_writer(time() - t0)
        self.log(f"Module Runtime: {elapsed:.2f} {unit}\n")

    # --- Split Proofreading ---
    def split_proofreading(
        self,
        model,
        proposals_config,
        batch_size=32,
        dt=0.05,
        min_threshold=0.8,
        removal_threshold=0.3,
        patch_shape=None,
        save_detections=True,
    ):
        # Run inference
        self.step_cnt += 1
        self.log(f"\nStep {self.step_cnt}: Split Proofreading")
        img_config = self._img_config(patch_shape)
        step_output = self._step_dir(SplitProofreader.step_name)
        proofreader = SplitProofreader(
            self.graph,
            model,
            img_config,
            step_output,
            batch_size=batch_size,
            device=self.device,
            log_handle=self.log_handle,
        )
        proofreader(
            proposals_config,
            dt=dt,
            min_threshold=min_threshold,
            removal_threshold=removal_threshold,
        )

        # Save final graph
        if save_detections:
            self.log("Final Graph...")
            self.log(self.graph.__repr__())
            self.reconfigure_node_radius()
            self.save_graph("final_swcs")

    def connect_soma_fragments(self, max_dist=25):
        self.log(f"\nConnect Soma Fragments with dist={max_dist}")
        summary = self.graph.connect_soma_fragments(max_dist=max_dist)
        self.log(summary)

    # --- Merge Proofreading ---
    def merge_proofreading(self, mode, save_detections=True, save_fragments=False):
        """
        Runs rule-based merge proofreading.

        Parameters
        ----------
        mode : str
            Detection strategy. Options are "heuristic" and "connected_somas".
        save_detections : bool, optional
            Indication of whether to save detected sites to disk. Default is
            True.
        save_fragments : bool, optional
            If True, saves the corrected graph SWCs into the step directory.
            Default is False.
        """
        self.step_cnt += 1
        self.log(f"\nStep {self.step_cnt}: Merge Proofreading ({mode})")

        if mode == "heuristic":
            ProofreaderClass = HighRiskMergeProofreader
        elif mode == "connected_somas":
            ProofreaderClass = SomaMergeProofreader
        else:
            raise ValueError(f"Unknown merge proofreading mode: {mode!r}")

        step_output = self._step_dir(ProofreaderClass.step_name)
        proofreader = ProofreaderClass(
            self.graph, step_output, log_handle=self.log_handle
        )
        merge_nodes = proofreader()
        self.log(f"# Merges Detected: {len(merge_nodes)}")

        if save_detections:
            merge_sites = [self.graph.node_xyz[i] for i in merge_nodes]
            proofreader.save_sites(merge_sites)
        if save_fragments:
            self._save_graph_to(step_output)
            proofreader.save_parameters()

    def learned_merge_detection(
        self,
        mode,
        model,
        batch_size=16,
        threshold=0.5,
        min_search_size=0,
        patch_shape=None,
        prefetch=64,
        save_detections=True,
        save_fragments=True,
    ):
        """
        Runs learned merge detection using a CNN.

        Parameters
        ----------
        mode : str
            Search strategy. "dense" scores every node along each fragment;
            "sparse" restricts scoring to branching nodes.
        model : torch.nn.Module
            Trained model used to score candidate merge sites.
        batch_size : int, optional
            Number of patches per forward pass. Default is 16.
        threshold : float, optional
            Confidence threshold above which a site is flagged as a merge.
            Default is 0.5.
        min_search_size : float, optional
            Minimum fragment cable length (in microns) to include in the
            search. Default is 0.
        patch_shape : Tuple[int], optional
            Patch shape to use for image sampling, overriding img_config.
            Default is None (uses img_config.patch_shape).
        prefetch : int, optional
            Number of patches to prefetch. Default is 64.
        save_detections : bool, optional
            If True, saves detection results to output_dir. Default is True.
        """
        self.step_cnt += 1
        self.log(f"\nStep {self.step_cnt}: Learned Merge Detection ({mode})")
        img_config = self._img_config(patch_shape)
        step_output = self._step_dir(MLMergeProofreader.step_name)
        proofreader = MLMergeProofreader(
            self.graph,
            model,
            img_config,
            step_output,
            mode=mode,
            batch_size=batch_size,
            device=self.device,
            min_search_size=min_search_size,
            prefetch=prefetch,
            threshold=threshold,
            log_handle=self.log_handle,
        )
        merge_nodes = proofreader()
        self.log(f"# Merges Detected: {len(merge_nodes)}")

        if save_detections:
            proofreader.save_sites(proofreader.merge_sites_xyz)
        if save_fragments:
            self._save_graph_to(step_output)
            proofreader.save_parameters()

    # --- Helpers ---
    def log(self, txt):
        """
        Logs and prints the given text.

        Parameters
        ----------
        txt : str
            Text to be logged and printed.
        """
        print(txt)
        self.log_handle.write(txt)
        self.log_handle.write("\n")

    def reconfigure_node_radius(self):
        n_nodes = len(self.graph.node_radius)
        self.graph.node_radius = np.ones((n_nodes), dtype=np.float16)
        for i, j in self.graph.accepts:
            self.graph.node_radius[i] = 6
            self.graph.node_radius[j] = 6

    def save_fragment_ids(self):
        path = f"{self.output_dir}/segment_ids.txt"
        segment_ids = list(self.graph.component_id_to_swc_id.values())
        util.write_list(path, segment_ids)

    def _img_config(self, patch_shape):
        if patch_shape is None:
            return self.img_config
        cfg = copy(self.img_config)
        cfg.patch_shape = patch_shape
        return cfg

    def _step_dir(self, name):
        path = os.path.join(self.output_dir, f"step{self.step_cnt}_{name}")
        util.mkdir(path)
        return path

    def save_graph(self, dirname):
        self._save_graph_to(os.path.join(self.output_dir, dirname))

    def _save_graph_to(self, dirpath):
        util.mkdir(dirpath)
        temp_dir = os.path.join(dirpath, "temp")
        self.graph.to_zipped_swcs_multithreaded(temp_dir)
        zip_paths = util.list_paths(temp_dir, extension=".zip")
        util.combine_zips(zip_paths, os.path.join(dirpath, "swcs.zip"))
        util.rmdir(temp_dir)
