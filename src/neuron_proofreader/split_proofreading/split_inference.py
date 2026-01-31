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
      repeats multiple times by calling the routine "run_schedule" within the
      InferencePipeline class.

"""

from datetime import datetime
from time import time
from torch.nn.functional import sigmoid
from tqdm import tqdm

import networkx as nx
import numpy as np
import os
import torch

from neuron_proofreader.split_proofreading.split_datasets import (
    FragmentsDataset
)
from neuron_proofreader.split_proofreading.split_feature_extraction import (
    FeaturePipeline,
    HeteroGraphData
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
        self.model_path = model_path
        self.output_dir = output_dir
        self.soma_centroids = soma_centroids

        # Logger
        util.mkdir(self.output_dir)
        log_path = os.path.join(self.output_dir, "runtimes.txt")
        self.log_handle = open(log_path, 'a')
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
    def run(self, search_radius):
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
        """
        # Main
        t0 = time()
        self.log("\nStep 2: Generate Proposals")
        self.dataset.graph.generate_proposals(search_radius)
        n_proposals = format(self.graph.n_proposals(), ",")
        n_proposals_blocked = self.dataset.graph.n_proposals_blocked

        # Report results
        t, unit = util.time_writer(time() - t0)
        self.log(f"# Proposals: {n_proposals}")
        self.log(f"# Proposals Blocked: {n_proposals_blocked}")
        self.log(f"Module Runtime: {t:.2f} {unit}\n")

    def classify_proposals(self, accept_threshold):
        """
        Classifies proposals by calling "self.inference_engine". This routine
        generates features and runs a GNN to make predictions. Proposals with
        a prediction above "self.threshold" are accepted and added to the
        graph as an edge.
        """
        # Initializations
        self.log("Step 3: Run Inference")
        t0 = time()

        # Generate model predictions
        n_proposals = self.graph.n_proposals()
        inference_engine = InferenceEngine(
            self.graph,
            self.img_path,
            self.model_path,
            self.ml_config,
            segmentation_path=self.segmentation_path,
        )
        preds_dict = inference_engine.run()
        path = os.path.join(self.output_dir, "proposal_predictions.json")
        util.write_json(path, reformat_preds(preds_dict))

        # Update graph
        stop

        # Report results
        t, unit = util.time_writer(time() - t0)
        self.log(f"# Merges Blocked: {self.graph.n_merges_blocked}")
        self.log(f"# Accepted: {format(len(accepts), ',')}")
        self.log(f"% Accepted: {len(accepts) / n_proposals:.4f}")
        self.log(f"Module Runtime: {t:.4f} {unit}\n")

    def save_results(self):
        """
        Saves the processed results from running the inference pipeline,
        namely the corrected SWC files and a list of the merged SWC ids.
        """
        # Save temp result on local machine
        temp_dir = os.path.join(self.output_dir, "temp")
        self.graph.to_zipped_swcs(temp_dir, sampling_rate=2)

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
            "brain_id": self.brain_id,
            "segmentation_id": self.segmentation_id,
            "min_fragment_size": f"{self.graph_config.min_size}um",
            "min_fragment_size_with_proposals": f"{self.graph_config.min_size_with_proposals}um",
            "node_spacing": self.graph_config.node_spacing,
            "remove_doubles": self.graph_config.remove_doubles,
            "use_somas": len(self.soma_centroids) > 0,
            "complex_proposals": self.graph_config.complex_bool,
            "long_range_bool": self.graph_config.long_range_bool,
            "proposals_per_leaf": self.graph_config.proposals_per_leaf,
            "search_radius": f"{self.graph_config.search_radius}um",
            "model_name": os.path.basename(self.model_path),
            "accept_threshold": self.ml_config.threshold,
        }
        path = os.path.join(self.output_dir, "metadata.json")
        util.write_json(path, metadata)

    # --- Summaries ---
    def log(self, txt):
        print(txt)
        self.log_handle.write(txt)
        self.log_handle.write("\n")


class InferenceEngine:
    """
    Class that runs inference with a machine learning model that has been
    trained to classify edge proposals.
    """

    def __init__(
        self,
        dataset,
        model_path,
        ml_config,
    ):
        """
        Initializes an inference engine by loading images and setting class
        attributes.

        Parameters
        ----------
        img_path : str
            Path to image.
        model_path : str
            Path to machine learning model weights.
        ml_config : MLConfig
            Configuration object containing parameters and settings required
            for the inference.
        segmentation_path : str, optional
            Path to segmentation stored in GCS bucket. Default is None.
        """
        # Instance attributes
        self.batch_size = ml_config.batch_size
        self.device = ml_config.device
        self.model = VisionHGAT(ml_config.patch_shape)
        self.pbar = tqdm(total=graph.n_proposals(), desc="Inference")
        self.subgraph_sampler = self.get_subgraph_sampler(graph)

        # Feature generator
        self.feature_extractor = FeaturePipeline(
            graph,
            img_path,
            brightness_clip=ml_config.brightness_clip,
            patch_shape=ml_config.patch_shape,
            segmentation_path=segmentation_path
        )

        # Load weights
        ml_util.load_model(self.model, model_path, device=ml_config.device)

    def run(self):
        preds = dict()
        for subgraph in self.dataset:
            # Get model inputs
            features = self.feature_extractor(subgraph)
            data = HeteroGraphData(features)

            # Run model
            preds.update(self.predict(data))
            self.pbar.update(subgraph.n_proposals())
        return preds

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
            x = data.get_inputs().to(self.device)
            hat_y = sigmoid(self.model(x))

        # Reformat predictions
        idx_to_id = data.idxs_proposals.idx_to_id
        hat_y = ml_util.tensor_to_list(hat_y)
        return {idx_to_id[i]: y_i for i, y_i in enumerate(hat_y)}

    def update_graph(self, preds, high_threshold=0.9):
        """
        Determines which proposals to accept based on prediction scores and
        the specified threshold.

        Parameters
        ----------
        preds : dict
            Dictionary that maps proposal ids to probability generated from
            machine learning model.
        high_threshold : float, optional
            Threshold value for separating the best proposals from the rest.
            Default is 0.9.

        Returns
        -------
        list
            Proposals to be added as edges to "graph".
        """
        # Partition proposals into best and the rest
        preds = {k: v for k, v in preds.items() if v > self.threshold}
        best_proposals, proposals = self.separate_best(preds, high_threshold)

        # Determine which proposals to accept
        accepts = list()
        accepts.extend(self.add_accepts(best_proposals))
        accepts.extend(self.add_accepts(proposals))
        return accepts

    def separate_best(self, preds, high_threshold):
        """
        Splits "preds" into two separate dictionaries such that one contains
        the best proposals (i.e. simple proposals with high confidence) and
        the other contains all other proposals.

        Parameters
        ----------
        preds : dict
            Dictionary that maps proposal ids to probability generated from
            machine learning model.
        high_threshold : float
            Threshold on acceptance probability for proposals.

        Returns
        -------
        list
            Proposal IDs determined to be the best.
        list
            All other proposal IDs.
        """
        best_probs, probs = list(), list()
        best_proposals, proposals = list(), list()
        simple_proposals = self.graph.simple_proposals()
        for proposal, prob in preds.items():
            if proposal in simple_proposals and prob > high_threshold:
                best_proposals.append(proposal)
                best_probs.append(prob)
            else:
                proposals.append(proposal)
                probs.append(prob)
        best_idxs = np.argsort(best_probs)
        idxs = np.argsort(probs)
        return np.array(best_proposals)[best_idxs], np.array(proposals)[idxs]

    def add_accepts(self, proposals):
        """
        ...

        Parameters
        ----------
        proposals : list[frozenset]
            Proposals with predicted probability above threshold to be added
            to the graph.

        Returns
        -------
        List[frozenset]
            List of proposals that do not create a cycle when iteratively
            added to "graph".
        """
        accepts = list()
        for proposal in proposals:
            i, j = tuple(proposal)
            if not nx.has_path(self.graph, i, j):
                self.graph.merge_proposal(proposal)
                accepts.append(proposal)
        return accepts


# --- Helpers ---
def reformat_preds(preds_dict):
    return {str(k): v for k, v in preds_dict.items()}
