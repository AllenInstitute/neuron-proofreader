"""
Created on Sat November 04 15:30:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code that executes the full GraphTrace inference pipeline.

    Inference Algorithm:
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

from datetime import datetime
from time import time
from torch.nn.functional import sigmoid
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
        brain_id : str
            Identifier for the whole-brain dataset.
        segmentation_id : str
            Identifier for the segmentation model that generated fragments.
        img_path : str
            Path to the whole-brain image stored in a GCS or S3 bucket.
        model_path : str
            Path to machine learning model parameters.
        output_dir : str
            Directory where the results of the inference will be saved.
        config : Config
            Configuration object containing parameters and settings required
            for the inference pipeline.
        segmentation_path : str, optional
            Path to segmentation stored in GCS bucket. The default is None.
        soma_centroids : List[Tuple[float]] or None, optional
            Physcial coordinates of soma centroids. The default is an empty
            list.
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
        swcs_path : str
            Path to SWC files used to build an instance of FragmentGraph,
            see "swc_util.Reader" for further documentation.
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
        self.dataset.graph.generate_proposals(search_radius)
        n_proposals = format(self.dataset.graph.n_proposals(), ",")
        n_proposals_blocked = self.dataset.graph.n_proposals_blocked

        # Report results
        t, unit = util.time_writer(time() - t0)
        self.log(f"# Proposals: {n_proposals}")
        self.log(f"# Proposals Blocked: {n_proposals_blocked}")
        self.log(f"Module Runtime: {t:.2f} {unit}\n")

    def classify_proposals(self, accept_threshold, dt=0.05):
        t0 = time()
        self.log("Step 3: Run Inference")

        # Main
        accepts = set()
        new_threshold = 0.99
        preds = self.predict_proposals()
        while True:
            # Update graph
            cur_threshold = new_threshold
            accepts.update(self.merge_proposals(preds, cur_threshold))

            # Update threshold
            new_threshold = max(cur_threshold - dt, accept_threshold)
            if cur_threshold == new_threshold:
                break

        # Report results
        t, unit = util.time_writer(time() - t0)
        self.log(f"# Merges Blocked: {self.dataset.graph.n_merges_blocked}")
        self.log(f"# Accepted: {format(len(accepts), ',')}")
        self.log(f"% Accepted: {len(accepts) / len(preds):.4f}")
        self.log(f"Module Runtime: {t:.4f} {unit}\n")

    def predict_proposals(self):
        # Main
        preds = dict()
        pbar = tqdm(total=self.dataset.graph.n_proposals(), desc="Inference")
        for data in self.dataset:
            preds.update(self.predict(data))
            pbar.update(data.n_proposals())

        # Save results
        path = os.path.join(self.output_dir, "proposal_predictions.json")
        util.write_json(path, reformat_preds(preds))
        return preds

    def merge_proposals(self, preds, threshold):
        accepts = list()
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
                accepts.append(proposal)
            del preds[proposal]
        return accepts

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
        self.write_metadata()
        self.log_handle.close()

        # Save result on s3 (if applicable)
        if self.s3_dict is not None:
            util.upload_dir_to_s3(
                self.output_dir,
                self.s3_dict["bucket_name"],
                self.s3_dict["prefix"]
            )

    # --- Helpers ---
    def log(self, txt):
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
        with torch.no_grad():
            device = self.config.ml.device
            x = data.get_inputs().to(device)
            hat_y = sigmoid(self.model(x))

        # Reformat predictions
        idx_to_id = data.idxs_proposals.idx_to_id
        hat_y = ml_util.tensor_to_list(hat_y)
        return {idx_to_id[i]: y_i for i, y_i in enumerate(hat_y)}

    def save_connections(self, round_id=None):
        """
        Writes the accepted proposals from the graph to a text file. Each line
        contains the two swc ids as comma separated values.
        """
        suffix = f"-{round_id}" if round_id else ""
        path = os.path.join(self.output_dir, f"connections{suffix}.txt")
        with open(path, "w") as f:
            for id_1, id_2 in self.graph.merged_ids:
                f.write(f"{id_1}, {id_2}" + "\n")

    def save_fragment_ids(self):
        path = f"{self.output_dir}/segment_ids.txt"
        segment_ids = list(self.dataset.graph.component_id_to_swc_id.values())
        util.write_list(path, segment_ids)

    def write_metadata(self):
        """
        Writes metadata about the current pipeline run to a JSON file.
        """
        metadata = {
            "date": datetime.today().strftime("%Y-%m-%d"),
            "min_fragment_size": f"{self.config.graph.min_size}um",
            "min_fragment_size_with_proposals": f"{self.config.graph.min_size_with_proposals}um",
            "node_spacing": self.config.graph.node_spacing,
            "remove_doubles": self.config.graph.remove_doubles,
            "use_somas": len(self.soma_centroids) > 0,
            "proposals_per_leaf": self.config.graph.proposals_per_leaf,
            "search_radius": f"{self.config.graph.search_radius}um",
            "model_name": os.path.basename(self.model_path),
            "accept_threshold": self.config.ml.threshold,
        }
        path = os.path.join(self.output_dir, "metadata.json")
        util.write_json(path, metadata)


# --- Helpers ---
def reformat_preds(preds_dict):
    return {str(k): v for k, v in preds_dict.items()}
