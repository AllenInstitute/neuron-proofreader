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

from scipy.spatial import KDTree
from time import time
from tqdm import tqdm

import numpy as np
import pandas as pd
import os
import torch

from neuron_proofreader.split_proofreading.new_split_feature_extraction import (
    ArboristSplitDataset,
)
from neuron_proofreader.split_proofreading.split_datasets import (
    FragmentsDataset,
)
from neuron_proofreader.utils import ml_util, util


def write_connections(merged_ids, output_dir):
    """
    Writes merged SWC ID pairs to connections.txt in output_dir.
    """
    path = os.path.join(output_dir, "connections.txt")
    with open(path, "w") as f:
        for id1, id2 in merged_ids:
            f.write(f"{id1}, {id2}\n")


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
        write_connections(self.graph.merged_ids, self.output_dir)

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
        use_tree_features=False,
    ):
        """
        Initializes an object that executes the full split correction
        pipeline.

        Parameters
        ----------
        ...
        """
        # Instance attributes
        DatasetClass = (
            ArboristSplitDataset if use_tree_features else FragmentsDataset
        )
        self.dataset = DatasetClass(
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
        # Length features are normalized by this radius, which must not
        # change between stages.
        self.dataset.graph.search_radius = proposals_config.max_search_radius

        # Staged search: proposals are generated at increasing radii and the
        # splits found at each radius are corrected before searching farther,
        # so a leaf resolved by a short proposal never spawns long ones.
        # Scores of unresolved proposals carry over between stages.
        t0 = time()
        preds = dict()
        total_proposals = 0
        for radius, max_angle in proposals_config.search_schedule():
            old_to_new = self.generate_proposals(
                proposals_config, radius, max_angle=max_angle
            )
            preds = self.remap_preds(preds, old_to_new)
            n_new = len(set(self.dataset.list_proposals()) - preds.keys())
            total_proposals += n_new
            self.log(f"# New Proposals: {format(n_new, ',')}\n")
            preds = self.correct_splits(
                preds,
                dt,
                min_threshold,
                removal_threshold,
                suffix=f"_radius={radius:g}",
            )

        # Report results
        t, unit = util.time_writer(time() - t0)
        p_accepts = 100 * len(self.dataset.accepts) / max(total_proposals, 1)
        self.log(f"Overall Accepted: {p_accepts:.2f}%")
        self.log(f"Total Runtime: {t:.2f} {unit}\n")
        self.save_connections()

        # Drop unresolved proposals: nothing downstream uses them, and every
        # later relabel_nodes (e.g. after merge removal) would re-add them.
        self.dataset.reset_proposals()

    def correct_splits(
        self, preds, dt, min_threshold, removal_threshold, suffix=""
    ):
        """
        Scores and merges the graph's current proposals with a decreasing
        threshold schedule, first leaf-to-leaf proposals only, then all.

        Parameters
        ----------
        preds : Dict[Frozenset[int], float]
            Scores of proposals already scored in an earlier stage.
        dt : float
            Increment that acceptance threshold is lowered by.
        min_threshold : float
            Minimum threshold for accepting proposals.
        removal_threshold : float
            Proposals with model predictions less than this are removed.
        suffix : str, optional
            Suffix for saved prediction files. Default is "".

        Returns
        -------
        preds : Dict[Frozenset[int], float]
            Scores of the proposals left unresolved.
        """
        # Scores are kept across rounds; a proposal is only re-scored once an
        # accepted merge lands within reach of its features (image patch or
        # tree subgraph), since nothing else changes them.
        cnt = 0
        stale = set(self.dataset.list_proposals()) - preds.keys()
        reach = self.feature_reach()
        for only_leaf2leaf in [True, False]:
            # Reset threshold
            name = "_leaf2leaf" if only_leaf2leaf else ""
            new_threshold = 0.99
            pass_min_threshold = min_threshold
            if not only_leaf2leaf:
                pass_min_threshold += dt

            # Generate predictions
            while self.dataset.proposals:
                # Score proposals whose features changed since last scored
                cnt += 1
                self.log(
                    f"\n--- Threshold={new_threshold} w/ only_leaf2leaf={only_leaf2leaf} ---"
                )
                preds.update(self.predict_proposals(stale))
                self.log(
                    f"# Proposals Scored: {len(stale)} of {self.dataset.n_proposals()}"
                )
                self.save_model_predictions(
                    preds,
                    suffix=f"{suffix}{name}_round={cnt}_threshold={new_threshold}",
                )

                # Merge accepted proposals
                cur_threshold = new_threshold
                accepts_before = set(self.dataset.accepts)
                self.merge_with_threshold_schedule(
                    preds, cur_threshold, dt=dt, only_leaf2leaf=only_leaf2leaf
                )

                # Remove rejected proposals
                self.remove_proposals(preds, removal_threshold)

                # Invalidate scores near this round's merges
                accepted = self.dataset.accepts - accepts_before
                remaining = set(self.dataset.list_proposals())
                preds = {p: v for p, v in preds.items() if p in remaining}
                stale = remaining - preds.keys()
                stale |= self.proposals_within(accepted, remaining, reach)

                # Update acceptance threshold
                if cur_threshold <= pass_min_threshold:
                    break
                new_threshold = max(cur_threshold - dt, pass_min_threshold)
        return preds

    # --- Core Routines ---
    def generate_proposals(self, proposals_config, radius, max_angle=None):
        """
        Generates proposals for the fragments graph at the given radius,
        keeping existing proposals and skipping rejected pairs.

        Parameters
        ----------
        proposals_config : ProposalsConfig
            Config object with settings for proposal generation.
        radius : float
            Search radius (in microns).
        max_angle : float, optional
            Tangent cone angle (in degrees) for long-range search. Default is
            None.

        Returns
        -------
        old_to_new : dict or None
            Node ID remapping if trimming relabeled nodes, otherwise None.
        """
        # Main
        t0 = time()
        self.log(f"\nGenerate Proposals (radius={radius:g})...")
        old_to_new = self.dataset.generate_proposals(
            radius,
            allow_nonleaf_proposals=proposals_config.allow_nonleaf_proposals,
            max_angle=max_angle,
            max_attempts=1,
            max_proposals_per_leaf=proposals_config.max_proposals_per_leaf,
            min_size_with_proposals=proposals_config.min_size_with_proposals,
        )

        # Report results
        n_proposals = format(self.dataset.n_proposals(), ",")
        n_proposals_blocked = self.dataset.n_proposals_blocked
        t, unit = util.time_writer(time() - t0)

        self.log(f"Search Radius: {radius:g}")
        if max_angle is not None:
            self.log(f"Tangent Cone Angle: {max_angle:g}")
        self.log(f"# Proposals: {n_proposals}")
        self.log(f"# Proposals Blocked: {n_proposals_blocked}")
        self.log(f"Module Runtime: {t:.2f} {unit}")
        return old_to_new

    @staticmethod
    def remap_preds(preds, old_to_new):
        """
        Applies a node ID remapping to scored proposals, dropping any whose
        endpoint was removed.
        """
        if old_to_new is None:
            return preds
        remapped = dict()
        for proposal, pred in preds.items():
            i, j = proposal
            if i in old_to_new and j in old_to_new:
                key = frozenset({int(old_to_new[i]), int(old_to_new[j])})
                remapped[key] = pred
        return remapped

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

    def predict_proposals(self, proposals):
        """
        Performs inference over the given proposals.

        Parameters
        ----------
        proposals : Iterable[Frozenset[int]]
            Proposals to score.

        Returns
        -------
        preds : Dict[Frozenset[int], float]
            Dictionary that maps proposals to the model prediction.
        """
        preds = dict()
        proposals = list(proposals)
        if not proposals:
            return preds

        pbar = tqdm(total=len(proposals), desc="Inference")
        for data in self.dataset.iter_proposals(proposals):
            preds.update(self.predict(data))
            pbar.update(data.n_proposals())
        return preds

    def feature_reach(self):
        """
        Returns the distance (in microns) from a proposal endpoint beyond
        which a graph change cannot alter the proposal's features. The tree
        subgraph extends "max_depth" from the proposal midpoint and the image
        patch is sized to the proposal length plus padding, so both are
        bounded by the longest proposal plus the larger of the two.
        """
        dataset = self.dataset
        padding = dataset.feature_extractor.image_extractor.patch_loader.padding
        padding *= np.max(dataset.anisotropy)
        depth = getattr(dataset, "tree_extractor", None)
        depth = depth.max_depth if depth else 0
        max_length = max(
            map(dataset.proposal_length, dataset.list_proposals()), default=0
        )
        return max(depth, padding) + max_length

    def proposals_within(self, accepted, proposals, radius):
        """
        Returns the proposals with an endpoint within "radius" of the
        midpoint of any accepted proposal.
        """
        if not accepted or not proposals:
            return set()
        kdtree = KDTree([self.dataset.proposal_midpoint(p) for p in accepted])
        proposals = list(proposals)
        xyz = np.array([self.dataset.node_xyz[i] for p in proposals for i in p])
        hits = kdtree.query_ball_point(xyz, radius, return_length=True) > 0
        hits = hits.reshape(len(proposals), 2).any(axis=1)
        return {p for p, hit in zip(proposals, hits) if hit}

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
        proposals = [p for p in self.dataset.sorted_proposals() if p in preds]
        for proposal in tqdm(proposals, desc="Adding Accepts"):
            # Check for leaf2leaf condition
            is_leaf2leaf = self.dataset.is_leaf2leaf(proposal)
            if only_leaf2leaf and not is_leaf2leaf:
                continue

            # Check if proposal satifies threshold
            if preds[proposal] < threshold:
                continue

            # Check if proposal creates a loop
            i, j = proposal
            if self.dataset.node_component_id[i] != self.dataset.node_component_id[j]:
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
                self.dataset.rejected.add(proposal)
                cnt += 1

        # Sanity check. Proposals within one component can never merge, so
        # drop them rather than re-score them.
        for proposal in self.dataset.list_proposals():
            i, j = proposal
            same_component = (
                self.dataset.node_component_id[i]
                == self.dataset.node_component_id[j]
            )
            if same_component or self.dataset.degree(i) > 2 or self.dataset.degree(j) > 2:
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
        # Generate predictions. No autocast: it keeps GroupNorm in fp32 so
        # it never sped up the backbone, and it would undo "half_vision".
        with torch.inference_mode():
            x = data.get_inputs().to(self.device)
            x["img"] = x["img"].contiguous(
                memory_format=torch.channels_last_3d
            )
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
        write_connections(self.dataset.merged_ids, self.output_dir)

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
