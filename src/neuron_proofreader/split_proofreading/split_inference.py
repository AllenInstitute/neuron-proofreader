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
import os
import torch

from neuron_proofreader.split_proofreading.split_datasets import (
    FragmentsDataset
)
from neuron_proofreader.machine_learning.gnn_models import VisionHGAT
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
        model_path,
        output_dir,
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
        model_path : str
            Path to checkpoint file containing model weights.
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
        self.model = VisionHGAT(config.ml.patch_shape)
        self.output_dir = output_dir
        self.soma_centroids = soma_centroids

        # Logger
        util.mkdir(self.output_dir)
        log_path = os.path.join(self.output_dir, "runtimes.txt")
        self.log_handle = open(log_path, 'a')
        self.log(log_preamble)

        # Load data
        self._load_data(fragments_path, img_path, segmentation_path)
        ml_util.load_model(self.model, model_path, device=config.ml.device)

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
            soma_centroids=self.soma_centroids
        )
        self.save_fragment_ids()

        # Connect fragments very close to soma
        self.log(f"# Soma Fragments: {len(self.dataset.graph.soma_ids)}")
        if len(self.soma_centroids) > 0:
            somas = self.soma_centroids
            results = self.dataset.graph.connect_soma_fragments(somas)
            self.log(results)

        # Log results
        elapsed, unit = util.time_writer(time() - t0)
        self.log(self.dataset.graph.get_summary(prefix="\nInitial"))
        self.log(f"Module Runtime: {elapsed:.2f} {unit}\n")

    # --- Core Routines ---
    def __call__(self, search_radius):
        """
        Executes the full inference pipeline.

        Parameters
        ----------
        search_radius : float
            Search radius (in microns) used to generate proposals.
        """
        # Main
        t0 = time()
        self.generate_proposals(search_radius)
        self.classify_proposals(self.config.ml.threshold)

        # Report results
        t, unit = util.time_writer(time() - t0)
        self.log(self.dataset.graph.get_summary(prefix="\nFinal"))
        self.log(f"Total Runtime: {t:.2f} {unit}\n")
        self.save_results()

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
        self.dataset.graph.generate_proposals(search_radius)
        n_proposals = format(self.dataset.graph.n_proposals(), ",")
        n_proposals_blocked = self.dataset.graph.n_proposals_blocked

        # Report results
        t, unit = util.time_writer(time() - t0)
        self.log(f"# Proposals: {n_proposals}")
        self.log(f"# Proposals Blocked: {n_proposals_blocked}")
        self.log(f"Module Runtime: {t:.2f} {unit}\n")

    def classify_proposals(self, accept_threshold, dt=0.05):
        """
        Classifies and iteratively merges accepted proposals using a
        decreasing confidence threshold.

        Parameters
        ----------
        accept_threshold : float
            Minimum confidence threshold for accepting proposals.
        dt : float, optional
            Step size for decreasing the confidence threshold at each
            iteration. Default is 0.05.
        """
        t0 = time()
        self.log("Step 3: Run Inference")

        # Main
        new_threshold = 0.99
        preds = self.predict_proposals()
        while True:
            # Update graph
            cur_threshold = new_threshold
            self.merge_proposals(preds, cur_threshold)

            # Update threshold
            new_threshold = max(cur_threshold - dt, accept_threshold)
            if cur_threshold == new_threshold:
                break
        n_accepts = len(self.dataset.graph.accepts)

        # Report results
        t, unit = util.time_writer(time() - t0)
        self.log(f"# Merges Blocked: {self.dataset.graph.n_merges_blocked}")
        self.log(f"# Accepted: {format(n_accepts, ',')}")
        self.log(f"% Accepted: {n_accepts / len(preds):.4f}")
        self.log(f"Module Runtime: {t:.4f} {unit}\n")

    def predict_proposals(self):
        """
        Runs inference over all proposals and saves model predictions.

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
        path = os.path.join(self.output_dir, "proposal_predictions.json")
        util.write_json(path, self.reformat_preds(preds))
        return preds

    def merge_proposals(self, preds, threshold):
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
        """
        for proposal in self.dataset.graph.get_sorted_proposals():
            # Check if proposal has been visited
            if proposal not in preds:
                continue

            # Check if proposal satifies threshold
            i, j = tuple(proposal)
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
        # Save temp result on local machine
        temp_dir = os.path.join(self.output_dir, "temp")
        self.dataset.graph.to_zipped_swcs(temp_dir, sampling_rate=2)

        # Merge ZIPs
        swc_dir = os.path.join(self.output_dir, "corrected-swcs")
        swc_path = os.path.join(swc_dir, "corrected-swcs.zip")
        util.mkdir(swc_dir)

        zip_paths = util.list_paths(temp_dir, extension=".zip")
        util.combine_zips(zip_paths, swc_path)
        util.rmdir(temp_dir)

        # Save additional info
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

    def reformat_preds(self, preds_dict):
        id_to_pred = dict()
        for proposal, pred in preds_dict.items():
            node1, node2 = tuple(proposal)
            id1 = self.dataset.graph.get_swc_id(node1)
            id2 = self.dataset.graph.get_swc_id(node2)
            id_to_pred[str((id1, id2))] = pred
        return id_to_pred

    def save_connections(self, round_id=None):
        """
        Writes the accepted proposals from the graph to a text file. Each line
        contains the two swc ids as comma separated values.
        """
        suffix = f"-{round_id}" if round_id else ""
        path = os.path.join(self.output_dir, f"connections{suffix}.txt")
        with open(path, "w") as f:
            for id_1, id_2 in self.dataset.graph.merged_ids:
                f.write(f"{id_1}, {id_2}" + "\n")

    def save_fragment_ids(self):
        path = f"{self.output_dir}/segment_ids.txt"
        segment_ids = list(self.dataset.graph.component_id_to_swc_id.values())
        util.write_list(path, segment_ids)
