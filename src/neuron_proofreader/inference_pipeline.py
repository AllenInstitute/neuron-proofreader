"""
Created on Fri June 13 16:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code for running full neuron proofreading pipeline, including both split and
merge detection and correction.

"""

from neuron_proofreader.skeleton_graph import SkeletonGraph


class ProofreadPipeline:

    def __init__(
        self,
        fragments_path,
        graph_config,
        img_path,
        output_dir,
        log_preamble="",
        soma_centroids=list(),
    ):
        """
        Initializes an object that executes the full split proofreading
        pipeline.

        Parameters
        ----------
        fragments_path : str
            Path to SWC files to be loaded into graph.
        graph_config : GraphConfig
            Configuration object that contains parameters for building graph.
        img_path : str
            Path to whole-brain image corresponding to the given fragments.
        output_dir : str
            Directory where the results of the inference will be saved.
        log_preamble : str, optional
            String to be added to the beginning of log. Default is an empty
            string.
        soma_centroids : List[Tuple[float]], optional
            Physical coordinates of soma centroids. Default is an empty list.
        """
        # Instance attributes
        self.img_path = img_path
        self.output_dir = output_dir
        self.soma_centroids = soma_centroids

        # Logger
        util.mkdir(self.output_dir)
        log_path = os.path.join(self.output_dir, "summary.txt")
        self.log_handle = open(log_path, "a")
        self.log(log_preamble)

        # Load data
        self._load_data(fragments_path, img_path)

    def load_graph(self, fragments_path, config):
        """
        Loads a graph from the given fragments.

        Parameters
        ----------
        fragments_path : str
            Path to SWC files to be loaded into graph.
        config : GraphConfig
            Configuration object that contains parameters for building graph.
        """
        # Load data
        t0 = time()
        self.log("Step 1: Build Graph")
        self.graph = ProposalGraph(
            anisotropy=config.anisotropy,
            min_cable_length=config.min_cable_length,
            node_spacing=config.node_spacing,
            verbose=config.verbose,
        )

        # Save original graph state
        self.save_fragment_ids()
        self.save_graph("original_swcs")
        self.log("Initial Graph")
        self.log(self.graph)

        # Report runtime
        elapsed, unit = util.time_writer(time() - t0)
        self.log(f"Module Runtime: {elapsed:.2f} {unit}\n")

    # --- Split Proofreading ---
    def split_proofreading(self):
        pass

    # --- Merge Proofreading ---
    def merge_proofreading(self, mode):
        if mode == "heuristic":
            results = self.graph.remove_high_risk_merges()
            self.log(results)
        elif mode == "connected_somas":
            results = self.graph.remove_soma_merges()
            self.log(results)

    # --- Helpers ---
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
