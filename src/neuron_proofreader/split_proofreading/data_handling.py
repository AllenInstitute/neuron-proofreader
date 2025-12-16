"""
Created on Mon Dec 15 14:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Graph objects that encapsulate and preprocess data into inputs suitable for GNNs.

"""

from neuron_proofreader.utils import img_util


class GraphInput:

    def __init__(self, graph, proposals, img_path):
        # Instance attributes
        self.graph = graph
        self.proposals = proposals

        # Feature generation
        self.features = dict()
        self.patches = dict()

        # Image reader
        self.img_reader = img_util.TensorStoreReader(img_path)

    # --- Feature Generation ---
    def generate_features(self):
        pass

    # --- Getters ---
    def get_inputs(self):
        pass

    # --- Helpers ---
    def to(self):
        pass
