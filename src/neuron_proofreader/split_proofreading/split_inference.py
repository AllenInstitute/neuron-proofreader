"""
Created on Fri November 03 15:30:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code that executes the full split correction pipeline.

    Inference Pipeline:
        1. Graph Construction
            Build graph from neuron fragments.

        2. Proposal Generation
            Generate proposals for potential connections between fragments.

        3. Proposal Classification
            a. Feature Generation
                Extract features from proposals and graph for a machine
                learning model.
            b. Predict with Graph Neural Network (GNN)
                Run a GNN to classify proposals as accept/reject
                based on the learned features.
            c. Merge Accepted Proposals
                Add accepted proposals to the fragments graph as edges.

Note: Steps 2 and 3 of the inference pipeline can be iterated in a loop that
      repeats multiple times by calling the pipeline in a loop

"""

from time import time
from tqdm import tqdm

import networkx as nx
import numpy as np
import pandas as pd
import os
import torch

from neuron_proofreader.split_proofreading.split_datasets import (
    FragmentsDataset,
)
from neuron_proofreader.utils import ml_util, util


class InferencePipeline:
    """
    Class that executes the full split proofreader inference pipeline by
    performing the following steps:
        (1) Graph Construction
        (2) Proposal Generation
        (3) Proposal Classification.
    """

    def __init__(
        self,
        fragments_path,
        img_path,
        output_dir,
        model,
        config,
        log_preamble="",
        segmentation_path=None,
        soma_centroids=list(),
    ):
        """
        Initializes an object that executes the full split correction
        pipeline.

        Parameters
        ----------
        fragments_path : str
            Path to SWC files to be loaded into graph.
        img_path : str
            Path to whole-brain image corresponding to the given fragments.
        output_dir : str
            Directory where the results of the inference will be saved.
        config : Config
            Configuration object containing parameters and settings required
            for the inference pipeline.
        log_preamble : str, optional
            String to be added to the beginning of log. Default is an empty
            string.
        segmentation_path : str, optional
            Path to segmentation corresponding to the given fragments. Default
            is None.
        soma_centroids : List[Tuple[float]], optional
            Physcial coordinates of soma centroids. Default is an empty list.
        """
        # Instance attributes
        self.accepted_proposals = list()
        self.config = config
        self.img_path = img_path
        self.model = model.to(config.ml.device)
        self.output_dir = output_dir
        self.soma_centroids = soma_centroids

        # Logger
        util.mkdir(self.output_dir)
        log_path = os.path.join(self.output_dir, "runtimes.txt")
        self.log_handle = open(log_path, "a")
        self.log(log_preamble)

        # Load data
        self._load_data(fragments_path, img_path, segmentation_path)

    def _load_data(self, fragments_path, img_path, segmentation_path):
        """
        Builds a graph from the given fragments.

        Parameters
        ----------
        fragments_path : str
            Path to SWC files to be loaded into graph.
        img_path : str
            Path to whole-brain image corresponding to the given fragments.
        segmentation_path : str
            Path to segmentation corresponding to the given fragments.
        """
        # Load data
        t0 = time()
        self.log("Step 1: Build Graph")
        self.dataset = FragmentsDataset(
            fragments_path,
            img_path,
            self.config,
            segmentation_path=segmentation_path,
            soma_centroids=self.soma_centroids,
        )
        self.save_fragment_ids()
        self.save_graph("original_swcs")

        # Postprocess fragments with somas
        self.dataset.graph.remove_soma_merges()
        self.dataset.graph.connect_soma_fragments()
        self.save_graph("precorrected_swcs")

        # Log results
        elapsed, unit = util.time_writer(time() - t0)
        self.log(self.dataset.graph.summary(prefix="\nInitial"))
        self.log(f"Module Runtime: {elapsed:.2f} {unit}\n")

    # --- Pipelines ---
    def __call__(self, search_radius):
        """
        Executes the full inference pipeline.

        Parameters
        ----------
        search_radius : float
            Search radius (in microns) used to generate proposals.
        """
        # Generate proposal
        t0 = time()
        self.generate_proposals(search_radius)
        preds = self.predict_proposals()

        # Update graph
        self.merge_with_threshold_schedule(preds, self.config.ml.threshold)

        # Report results
        t, unit = util.time_writer(time() - t0)
        self.log(self.dataset.graph.summary(prefix="\nFinal"))
        self.log(f"Total Runtime: {t:.2f} {unit}\n")
        self.save_results()

    def multistep(
        self, search_radius, low_threshold=0.3, high_threshold=0.9
    ):
        # Generate proposals
        t0 = time()
        self.generate_proposals(search_radius)
        preds = self.predict_proposals(suffix="_round1")

        # Round 1: Update graph
        self.merge_with_threshold_schedule(
            preds, high_threshold, only_leaf2leaf=True
        )
        self.filter_proposals(preds, low_threshold)

        # Round 2: Update graph
        preds = self.predict_proposals()
        self.merge_with_threshold_schedule(
            preds, self.config.ml.threshold, only_leaf2leaf=False
        )

        # Report results
        t, unit = util.time_writer(time() - t0)
        self.log(self.dataset.graph.summary(prefix="\nFinal"))
        self.log(f"Total Runtime: {t:.2f} {unit}\n")
        self.save_results()

    # --- Core Routines ---
    def filter_proposals(self, preds, threshold):
        cnt = 0
        for proposal, pred in preds.items():
            is_valid = self.dataset.graph.is_mergeable(*proposal)
            if pred < threshold or not is_valid:
                self.dataset.graph.remove_proposal(proposal)
                cnt += 1

        print("# Proposals Removed:", cnt)
        print("# Proposals Remaining:", self.dataset.graph.n_proposals())

    def generate_proposals(self, search_radius):
        """
        Generates proposals for the fragments graph based on the specified
        configuration.

        Parameters
        ----------
        search_radius : float
            Search radius (in microns) used to generate proposals.
        """
        # Main
        t0 = time()
        self.log("\nStep 2: Generate Proposals")
        self.log(f"Search Radius: {search_radius}")
        self.dataset.graph.generate_proposals(
            search_radius,
            allow_nonleaf_proposals=self.config.graph.allow_nonleaf_proposals,
        )

        n_proposals = format(self.dataset.graph.n_proposals(), ",")
        n_proposals_blocked = self.dataset.graph.n_proposals_blocked

        # Report results
        t, unit = util.time_writer(time() - t0)
        self.log(f"# Proposals: {n_proposals}")
        self.log(f"# Proposals Blocked: {n_proposals_blocked}")
        self.log(f"Module Runtime: {t:.2f} {unit}\n")

    def merge_with_threshold_schedule(
        self, preds, min_threshold, dt=0.05, only_leaf2leaf=False
    ):
        """
        Classifies and iteratively merges proposals using a decreasing
        confidence threshold.

        Parameters
        ----------
        preds : Dict[Frozenset[int], float]
            Dictionary that maps proposals to model predictions.
        min_threshold : float
            Minimum threshold for accepting proposals.
        dt : float, optional
            Step size for decreasing the threshold at each iteration. Default
            is 0.05.
        only_leaf2leaf : bool, optional
            Indication of whether to only merge leaf2leaf proposals. Default
            is False.
        """
        # Initializations
        t0 = time()
        self.log("Step 3: Run Inference")
        n_proposals = self.dataset.graph.n_proposals()

        # Progressive merging
        new_threshold = 0.99
        while True:
            # Update graph
            cur_threshold = new_threshold
            self.merge_proposals(
                preds, cur_threshold, only_leaf2leaf=only_leaf2leaf
            )

            # Update threshold
            new_threshold = max(cur_threshold - dt, min_threshold)
            if cur_threshold == new_threshold:
                break
        n_accepts = len(self.dataset.graph.accepts)

        # Report results
        t, unit = util.time_writer(time() - t0)
        self.log(f"# Merges Blocked: {self.dataset.graph.n_merges_blocked}")
        self.log(f"# Accepted: {format(n_accepts, ',')}")
        self.log(f"% Accepted: {100 * n_accepts / n_proposals:.2f}")
        self.log(f"Module Runtime: {t:.2f} {unit}\n")
        return preds

    def predict_proposals(self, suffix=""):
        """
        Performs inference over all proposals and saves model predictions.

        Returns
        -------
        preds : Dict[Frozenset[int], float]
            Dictionary that maps proposals to the model prediction.
        """
        # Main
        preds = dict()
        pbar = tqdm(total=self.dataset.graph.n_proposals(), desc="Inference")
        for data in self.dataset:
            preds.update(self.predict(data))
            pbar.update(data.n_proposals())

        # Save results
        self.save_proposal_results(preds, suffix=suffix)
        return preds

    def merge_proposals(self, preds, threshold, only_leaf2leaf=False):
        """
        Merges nodes corresponding to for proposals that satify the threshold
        and no loop creation requirements.

        Parameters
        ----------
        preds : Dict[Frozenset[int], float]
            Dictionary that maps proposals to the model prediction.
        threshold : float
            Threshold used to determine which proposals to accept based on
            model prediction.
        only_leaf2leaf : bool, optional
            Indication of whether to only merge leaf2leaf proposals. Default
            is False.
        """
        proposals = self.dataset.graph.sorted_proposals()
        for proposal in [p for p in proposals if p in preds]:
            # Check for leaf2leaf condition
            is_leaf2leaf = self.dataset.graph.is_leaf2leaf(proposal)
            if only_leaf2leaf and not is_leaf2leaf:
                continue

            # Check if proposal satifies threshold
            i, j = proposal
            if preds[proposal] < threshold:
                continue

            # Check if proposal creates a loop
            if not nx.has_path(self.dataset.graph, i, j):
                self.dataset.graph.merge_proposal(proposal)
            del preds[proposal]

    def save_results(self):
        """
        Saves the processed results from running the inference pipeline,
        namely the corrected SWC files and a list of the merged SWC ids.
        """
        self.reconfigure_node_radius()
        self.save_graph("corrected_swcs")
        self.save_connections()
        self.config.save(self.output_dir)
        self.log_handle.close()

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

    def predict(self, data):
        """
        ...

        Parameters
        ----------
        data : HeteroGraphData
            ...

        Returns
        -------
        Dict[Frozenset[int], float]
            Dictionary that maps proposal IDs to model predictions.
        """
        # Generate predictions
        with torch.inference_mode():
            device = self.config.ml.device
            x = data.get_inputs().to(device)
            with torch.cuda.amp.autocast(enabled=True):
                hat_y = torch.sigmoid(self.model(x))

        # Reformat predictions
        idx_to_id = data.idxs_proposals.idx_to_id
        hat_y = ml_util.tensor_to_list(hat_y)
        return {idx_to_id[i]: y_i for i, y_i in enumerate(hat_y)}

    def save_graph(self, dirname):
        # Set paths
        temp_dir = os.path.join(self.output_dir, "temp")
        output_zip_path = os.path.join(self.output_dir, dirname, "swcs.zip")
        util.mkdir(temp_dir)
        util.mkdir(os.path.join(self.output_dir, dirname))

        # Save swcs
        self.dataset.graph.to_zipped_swcs_multithreaded(temp_dir)
        zip_paths = util.list_paths(temp_dir, extension=".zip")
        util.combine_zips(zip_paths, output_zip_path)
        util.rmdir(temp_dir)

    def save_proposal_results(self, preds_dict, suffix=""):
        summary = list()
        for proposal, pred in preds_dict.items():
            # Extract info
            i, j = proposal
            segment_i = self.dataset.graph.node_swc_id(i)
            segment_j = self.dataset.graph.node_swc_id(j)

            # Add info
            summary.append(
                {
                    "Proposal": (segment_i, segment_j),
                    "Leaf2Leaf": self.dataset.graph.is_leaf2leaf(proposal),
                    "Length": self.dataset.graph.proposal_length(proposal),
                    "Prediction": pred,
                    "Segment1": segment_i,
                    "Segment2": segment_j,
                    "Voxel1": self.dataset.graph.node_voxel(i),
                    "Voxel2": self.dataset.graph.node_voxel(j),
                    "World1": self.dataset.graph.node_xyz[i],
                    "World2": self.dataset.graph.node_xyz[j],
                }
            )

        # Save results
        path = os.path.join(self.output_dir, f"proposal_summary{suffix}.csv")
        pd.DataFrame(summary).set_index("Proposal").to_csv(path)

    def reconfigure_node_radius(self):
        n_nodes = len(self.dataset.graph.node_radius)
        self.dataset.graph.node_radius = np.ones((n_nodes), dtype=np.float16)
        for i, j in self.dataset.graph.accepts:
            self.dataset.graph.node_radius[i] = 6
            self.dataset.graph.node_radius[j] = 6

    def save_connections(self):
        """
        Writes the accepted proposals to a text file. Each line contains the
        two SWC IDs as comma separated values.
        """
        path = os.path.join(self.output_dir, "connections.txt")
        with open(path, "w") as f:
            for id1, id2 in self.dataset.graph.merged_ids:
                f.write(f"{id1}, {id2}" + "\n")

    def save_fragment_ids(self):
        path = f"{self.output_dir}/segment_ids.txt"
        segment_ids = list(self.dataset.graph.component_id_to_swc_id.values())
        util.write_list(path, segment_ids)
