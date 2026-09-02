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
from neuron_proofreader.merge_proofreading.search_datasets import (
    DenseSearchDataset,
    SparseSearchDataset,
)
from neuron_proofreader.proposal_graph import ProposalGraph
from neuron_proofreader.split_proofreading.split_inference import (
    LearnedSplitProofreader,
    SomaSplitProofreader,
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
    def soma_split_proofreading(self, max_dist=25, save_fragments=True):
        """
        Runs soma-based split proofreading by connecting fragments near somas.

        Parameters
        ----------
        max_dist : float, optional
            Maximum distance (in microns) to search for fragments near each
            soma. Default is 25.
        save_fragments : bool, optional
            If True, saves the corrected graph SWCs into the step directory.
            Default is True.
        """
        # Step initializations
        self.step_cnt += 1
        self.log(f"\nStep {self.step_cnt}: Soma Split Proofreading")
        step_output = self._step_dir(SomaSplitProofreader.step_name)

        # Run proofreading
        proofreader = SomaSplitProofreader(
            self.graph,
            step_output,
            max_dist=max_dist,
            log_handle=self.log_handle,
        )
        proofreader()

        # Save results
        if save_fragments:
            self.log("Graph State...")
            self.log(self.graph.__repr__())
            self.save_graph(os.path.basename(step_output))

    def learned_split_detection(
        self,
        model,
        proposals_config,
        split_config,
        save_fragments=True,
    ):
        """
        Runs learned split detection using a GNN.

        Parameters
        ----------
        model : torch.nn.Module
            Trained model used to classify proposals.
        proposals_config : ProposalsConfig
            Config object with settings for proposal generation.
        split_config : SplitInferenceConfig
            Config object with settings for split inference.
        save_fragments : bool, optional
            If True, saves the corrected graph SWCs into the step directory.
            Default is True.
        """
        # Step initializations
        assert split_config.batch_size, "split_config.batch_size must be set!"
        self.step_cnt += 1
        self.log(f"\nStep {self.step_cnt}: Learned Split Detection")
        img_config = self._img_config(split_config.patch_shape)
        step_output = self._step_dir(LearnedSplitProofreader.step_name)

        # Run proofreading
        proofreader = LearnedSplitProofreader(
            self.graph,
            model,
            img_config,
            step_output,
            batch_size=split_config.batch_size,
            device=self.device,
            log_handle=self.log_handle,
        )
        proofreader(
            proposals_config,
            dt=split_config.dt,
            min_threshold=split_config.min_threshold,
            removal_threshold=split_config.removal_threshold,
        )

        # Save result
        if save_fragments:
            self.log("Graph State...")
            self.log(self.graph.__repr__())
            self.reconfigure_node_radius()
            self.save_graph(os.path.basename(step_output))

    # --- Merge Proofreading ---
    def merge_proofreading(
        self, mode, save_detections=True, save_fragments=True
    ):
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
            Default is True.
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
            self.save_graph(os.path.basename(step_output))
            proofreader.save_parameters()

    def learned_merge_detection(
        self,
        model,
        merge_config,
        save_detections=True,
        save_fragments=True,
    ):
        """
        Runs learned merge detection using a CNN.

        Parameters
        ----------
        model : torch.nn.Module
            Trained model used to score candidate merge sites.
        merge_config : MergeInferenceConfig
            Config object with settings for merge inference.
        save_detections : bool, optional
            If True, saves detection results to output_dir. Default is True.
        save_fragments : bool, optional
            If True, saves the corrected graph SWCs into the step directory.
            Default is True.
        """
        # Check that batch size is set
        if merge_config.batch_size is None:
            raise ValueError("merge_config.batch_size must be set!")

        # Step initializations
        self.step_cnt += 1
        self.log(f"\nStep {self.step_cnt}: Learned Merge Detection ({merge_config.search_mode})")
        img_config = self._img_config(merge_config.patch_shape)
        step_output = self._step_dir(MLMergeProofreader.step_name)

        # Create dataset
        DatasetClass = DenseSearchDataset if merge_config.search_mode == "dense" else SparseSearchDataset
        dataset = DatasetClass(
            self.graph,
            img_config,
            min_search_size=merge_config.min_search_size,
            prefetch=merge_config.prefetch,
        )

        # Run proofreading
        proofreader = MLMergeProofreader(
            dataset,
            model,
            step_output,
            batch_size=merge_config.batch_size,
            device=self.device,
            threshold=merge_config.threshold,
            log_handle=self.log_handle,
        )
        merge_nodes = proofreader()
        self.log(f"# Merges Detected: {len(merge_nodes)}")

        # Save results
        if save_detections:
            proofreader.save_sites(proofreader.merge_sites_xyz)
        if save_fragments:
            self.save_graph(os.path.basename(step_output))

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
        self.log_handle.write(txt + "\n")

    def reconfigure_node_radius(self):
        n_nodes = self.graph.num_nodes()
        radius = np.ones((n_nodes), dtype=np.float16)
        for i, j in self.graph.accepts:
            radius[i] = 6
            radius[j] = 6
        self.graph.node_feats["radius"] = radius

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

    def save_final_result(self):
        self.log("\nFinal Graph...")
        self.log(self.graph.__repr__())
        self.reconfigure_node_radius()
        self.save_graph("final_swcs")

    def save_graph(self, dirname):
        dirpath = os.path.join(self.output_dir, dirname)
        util.mkdir(dirpath)
        temp_dir = os.path.join(dirpath, "temp")
        self.graph.to_zipped_swcs_multithreaded(temp_dir)
        zip_paths = util.list_paths(temp_dir, extension=".zip")
        util.combine_zips(zip_paths, os.path.join(dirpath, "swcs.zip"))
        util.rmdir(temp_dir)
