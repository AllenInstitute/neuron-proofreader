"""
Created on Fri November 03 15:30:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code that executes the full split correction pipeline.

    Inference Pipeline:
        1. Proposal Generation
            Generate proposals for potential connections between fragments.

        2. Proposal Classification
            a. Feature Generation
                Extract features from proposals and graph for a machine
                learning model.
            b. Predict with Graph Neural Network (GNN)
                Run a GNN to classify proposals as accept/reject
                based on the learned features.

        3. Merge Accepted Proposals
            Add accepted proposals as edges to the graph.

"""

from time import time
from tqdm import tqdm

import pandas as pd
import rustworkx as rx
import os
import torch

from neuron_proofreader.split_proofreading.split_datasets import (
    FragmentsDataset,
)
from neuron_proofreader.utils import ml_util, util


class SomaSplitProofreader:
    """
    Heuristic split proofreader that reconnects fragments close to soma
    locations by adding edges between soma nodes and nearby fragment nodes.
    """

    step_name = "soma_split_correction"

    def __init__(self, graph, output_dir, max_dist=25, log_handle=None):
        """
        Initializes a SomaSplitProofreader.

        Parameters
        ----------
        graph : FragmentsGraph
            Skeleton graph to search for split errors near somas.
        output_dir : str
            Directory where results will be saved.
        max_dist : float, optional
            Maximum distance (in microns) to search for fragments near each
            soma. Default is 25.
        log_handle : file-like, optional
            Open file handle to write log messages to. If None, a new
            summary.txt is opened in output_dir. Default is None.
        """
        self.graph = graph
        self.max_dist = max_dist
        self.output_dir = output_dir
        log_path = os.path.join(output_dir, "summary.txt")
        self.log_handle = log_handle or open(log_path, "a")

    def __call__(self):
        t0 = time()
        self.log("Connect Soma Fragments...")
        summary = self.graph.connect_soma_fragments(max_dist=self.max_dist)
        self.log(summary)
        t, unit = util.time_writer(time() - t0)
        self.log(f"Module Runtime: {t:.2f} {unit}\n")

    def log(self, txt):
        print(txt)
        self.log_handle.write(txt + "\n")


class LearnedSplitProofreader:
    """
    Class that executes the full learned split proofreader inference pipeline.
    """

    step_name = "learned_split_correction"

    def __init__(
        self,
        graph,
        model,
        img_config,
        output_dir,
        batch_size=32,
        device="cuda",
        log_handle=None,
    ):
        """
        Initializes an object that executes the full split correction
        pipeline.

        Parameters
        ----------
        ...
        """
        # Instance attributes
        self.dataset = FragmentsDataset(
            graph,
            img_config,
            batch_size=batch_size,
        )
        self.device = device
        self.model = self.prepare_model(model)
        self.output_dir = output_dir

        # Logger
        log_path = os.path.join(self.output_dir, "summary.txt")
        self.log_handle = log_handle or open(log_path, "a")

    @staticmethod
    def prepare_model(model):
        """
        Puts the model in inference mode with the 3D CNN weights in
        channels-last layout, which makes its fp16 convolutions about 1.3x
        faster on tensor cores when the input patches use the same layout
        (see "predict"). cuDNN autotuning is deliberately left off: it gave
        no further speedup here and costs ~10s for every distinct batch size,
        and partially filled batches make batch sizes vary.
        """
        model.eval()
        if hasattr(model, "patch_embedding"):
            model.patch_embedding.to(memory_format=torch.channels_last_3d)
        return model

    def __call__(
        self,
        proposals_config,
        dt=0.1,
        min_threshold=0.8,
        removal_threshold=0.3,
    ):
        """
        Executes the full inference pipeline.

        Parameters
        ----------
        proposals_config : ProposalsConfig
            Config object with settings for proposal generation.
        dt : float, optional
            Increment that acceptance threshold is lowered by. Default is 0.1.
        min_threshold : float, optional
            Minimum threshold for accepting proposals. Default is 0.8.
        removal_threshold : float, optional
            Proposals with model predictions less than this value are removed.
            Default is 0.3.
        """
        # Generate proposals
        self.generate_proposals(proposals_config)
        total_proposals = self.dataset.n_proposals()

        # Run inference
        cnt = 0
        t0 = time()
        for only_leaf2leaf in [True, False]:
            # Reset threshold
            name = "_leaf2leaf" if only_leaf2leaf else ""
            new_threshold = 0.99
            if not only_leaf2leaf:
                min_threshold += dt

            # Generate predictions
            while self.dataset.proposals:
                # Generate proposal predictons
                cnt += 1
                self.log(
                    f"\n--- Threshold={new_threshold} w/ only_leaf2leaf={only_leaf2leaf} ---"
                )
                preds = self.predict_proposals(
                    suffix=f"{name}_round={cnt}_threshold={new_threshold}"
                )

                # Merge accepted proposals
                cur_threshold = new_threshold
                self.merge_with_threshold_schedule(
                    preds, cur_threshold, dt=dt, only_leaf2leaf=only_leaf2leaf
                )

                # Remove rejected proposals
                self.remove_proposals(preds, removal_threshold)

                # Update acceptance threshold
                if cur_threshold <= min_threshold:
                    break
                new_threshold = max(cur_threshold - dt, min_threshold)

        # Report results
        t, unit = util.time_writer(time() - t0)
        p_accepts = 100 * len(self.dataset.accepts) / total_proposals
        self.log(f"Overall Accepted: {p_accepts:.2f}%")
        self.log(f"Total Runtime: {t:.2f} {unit}\n")
        self.save_connections()

        # Drop unresolved proposals: nothing downstream uses them, and every
        # later relabel_nodes (e.g. after merge removal) would re-add them.
        self.dataset.reset_proposals()

    # --- Core Routines ---
    def generate_proposals(self, proposals_config):
        """
        Generates proposals for the fragments graph based on the specified
        configuration.

        Parameters
        ----------
        proposals_config : ProposalsConfig
            Config object with settings for proposal generation.
        """
        # Main
        t0 = time()
        self.log("Generate Proposals...")
        self.dataset.generate_proposals(
            proposals_config.search_radius,
            allow_nonleaf_proposals=proposals_config.allow_nonleaf_proposals,
            max_proposals_per_leaf=proposals_config.max_proposals_per_leaf,
            min_size_with_proposals=proposals_config.min_size_with_proposals,
        )

        # Report results
        n_proposals = format(self.dataset.n_proposals(), ",")
        n_proposals_blocked = self.dataset.n_proposals_blocked
        t, unit = util.time_writer(time() - t0)

        self.log(f"Search Radius: {proposals_config.search_radius}")
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
        n_proposals = self.dataset.n_proposals()
        n_accepts = 0

        # Progressive merging
        new_threshold = 0.99
        while True:
            # Update graph
            cur_threshold = new_threshold
            n_accepts += self.merge_proposals(
                preds, cur_threshold, only_leaf2leaf=only_leaf2leaf
            )

            # Update threshold
            if cur_threshold <= min_threshold:
                break
            new_threshold = max(cur_threshold - dt, min_threshold)

        # Report results
        t, unit = util.time_writer(time() - t0)
        self.log("Inference...")
        self.log(f"# Merges Blocked: {self.dataset.n_merges_blocked}")
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
        pbar = tqdm(total=self.dataset.n_proposals(), desc="Inference")
        for data in self.dataset:
            preds.update(self.predict(data))
            pbar.update(data.n_proposals())

        # Save results
        self.save_model_predictions(preds, suffix=suffix)
        return preds

    def merge_proposals(self, preds, threshold, only_leaf2leaf=False):
        """
        Merges proposals with model prediction above threshold and does
        not create a loop.

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
        n_accepts = 0
        proposals = self.dataset.sorted_proposals()
        for proposal in [p for p in proposals if p in preds]:
            # Check for leaf2leaf condition
            is_leaf2leaf = self.dataset.is_leaf2leaf(proposal)
            if only_leaf2leaf and not is_leaf2leaf:
                continue

            # Check if proposal satifies threshold
            if preds[proposal] < threshold:
                continue

            # Check if proposal creates a loop
            i, j = proposal
            if not rx.graph_has_path(self.dataset.graph, i, j):
                self.dataset.merge_proposal(proposal)
                n_accepts += 1
            del preds[proposal]
        return n_accepts

    def remove_proposals(self, preds, threshold):
        # Remove based on model predictions and mergeability
        cnt = 0
        for proposal, pred in preds.items():
            is_valid = self.dataset.is_mergeable(*proposal)
            if pred < threshold or not is_valid:
                self.dataset.remove_proposal(proposal)
                cnt += 1

        # Sanity check
        for proposal in self.dataset.list_proposals():
            i, j = proposal
            if self.dataset.degree(i) > 2 or self.dataset.degree(j) > 2:
                self.dataset.remove_proposal(proposal)
                cnt += 1

        self.log("Remove Proposals...")
        self.log(f"# Proposals Removed: {cnt}")
        self.log(f"# Proposals Remaining: {self.dataset.n_proposals()}\n")

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
        self.log_handle.flush()

    def predict(self, data):
        """
        Runs the model on one batch of proposals.

        Parameters
        ----------
        data : HeteroGraphData
            Batch of proposals with image patches and graph features.

        Returns
        -------
        Dict[Frozenset[int], float]
            Dictionary that maps proposal IDs to model predictions.
        """
        # Generate predictions
        with torch.inference_mode():
            x = data.get_inputs().to(self.device)
            x["img"] = x["img"].contiguous(
                memory_format=torch.channels_last_3d
            )
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                hat_y = torch.sigmoid(self.model(x))

        # Reformat predictions
        idx_to_id = data.idxs_proposals.idx_to_id
        hat_y = ml_util.tensor_to_list(hat_y)
        return {idx_to_id[i]: y_i for i, y_i in enumerate(hat_y)}

    def save_connections(self):
        """
        Writes accepted proposals to a text file. Each line contains the two
        SWC IDs as comma separated values.
        """
        path = os.path.join(self.output_dir, "connections.txt")
        with open(path, "w") as f:
            for id1, id2 in self.dataset.merged_ids:
                f.write(f"{id1}, {id2}" + "\n")

    def save_model_predictions(self, preds_dict, suffix=""):
        summary = list()
        for proposal, pred in preds_dict.items():
            # Extract info
            i, j = proposal
            segment_i = self.dataset.node_swc_id(i)
            segment_j = self.dataset.node_swc_id(j)

            # Add info
            summary.append(
                {
                    "Proposal": (segment_i, segment_j),
                    "Leaf2Leaf": self.dataset.is_leaf2leaf(proposal),
                    "Length": self.dataset.proposal_length(proposal),
                    "Prediction": pred,
                    "Segment1": segment_i,
                    "Segment2": segment_j,
                    "Voxel1": self.dataset.node_voxel(i),
                    "Voxel2": self.dataset.node_voxel(j),
                    "World1": self.dataset.node_xyz[i],
                    "World2": self.dataset.node_xyz[j],
                }
            )

        # Save results
        path = os.path.join(self.output_dir, f"proposal_summary{suffix}.csv")
        pd.DataFrame(summary).set_index("Proposal").to_csv(path)
