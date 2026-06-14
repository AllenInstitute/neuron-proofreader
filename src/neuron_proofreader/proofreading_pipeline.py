"""
Created on Fri June 13 16:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code for running full neuron proofreading pipeline, including both split and
merge detection and correction.

"""

from time import time

import numpy as np
import os

from neuron_proofreader.proposal_graph import ProposalGraph
from neuron_proofreader.split_proofreading.split_inference import (
    SplitProofreader,
)
from neuron_proofreader.utils import geometry_util, swc_util, util


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
        self.step_cnt += 1
        self.log(f"Step {self.step_cnt}: Build Graph")
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
        save_result=True,
    ):
        # Create proofreader
        proofreader = SplitProofreader(
            self.graph,
            model,
            self.img_config,
            self.output_dir,
            batch_size=batch_size,
            device=self.device,
            log_handle=self.log_handle,
        )

        # Run inference
        self.step_cnt += 1
        self.log(f"\nStep {self.step_cnt}: Split Proofreading")
        proofreader(
            proposals_config,
            dt=dt,
            min_threshold=min_threshold,
            removal_threshold=removal_threshold,
        )

        # Save final graph
        if save_result:
            self.log("Final Graph...")
            self.log(self.graph.__repr__())
            self.reconfigure_node_radius()
            self.save_graph("corrected_swcs")

    def connect_soma_fragments(self, max_dist=25):
        self.step_cnt += 1
        self.log(f"\nStep {self.step_cnt}: Connect Soma Fragments with dist={max_dist}")
        summary = self.graph.connect_soma_fragments(max_dist=max_dist)
        self.log(summary)

    # --- Merge Proofreading ---
    def merge_proofreading(self, mode):
        # Report step
        self.step_cnt += 1
        self.log(
            f"\nStep {self.step_cnt}: Merge Proofreading with mode={mode}"
        )

        # Detect merges
        if mode == "heuristic":
            merge_sites, summary = self.graph.remove_high_risk_merges()
        elif mode == "connected_somas":
            merge_sites, summary = self.graph.remove_soma_merges()

        # Report results
        self.log(summary)

        # Save sites
        color = "# COLOR 1.0 0.0 0.0"
        zip_path = os.path.join(self.output_dir, f"{mode}_merge_sites.zip")
        swc_util.write_points(
            zip_path, merge_sites, color=color, prefix="merge_site", radius=10
        )
        
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

    def save_graph(self, dirname):
        # Save graph across set of ZIPs
        temp_dir = os.path.join(self.output_dir, "temp")
        self.graph.to_zipped_swcs_multithreaded(temp_dir)

        # Combine ZIPs into single ZIP
        zip_paths = util.list_paths(temp_dir, extension=".zip")
        final_zip_path = os.path.join(self.output_dir, dirname, "swcs.zip")
        util.mkdir(os.path.join(self.output_dir, dirname))
        util.combine_zips(zip_paths, final_zip_path)
        util.rmdir(temp_dir)
