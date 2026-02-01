"""
Created on Frid Sept 15 16:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

This module defines a set of configuration classes used for setting storing
parameters used in neuron proofreading pipelines.

"""

from dataclasses import dataclass
from typing import Tuple

import os

from neuron_proofreader.utils import util


@dataclass
class ProposalGraphConfig:
    """
    Represents configuration settings related to graph properties and
    proposals generated.

    Attributes
    ----------
    anisotropy : Tuple[float]
        Scaling factors used to transform physical to image coordinates
        Default is (1.0, 1.0, 1.0).
    max_proposals_per_leaf : int
        Maximum number of proposals generated at leaf nodes. Default is 3.
    min_size : float
        Minimum path length (in microns) of SWC files loaded into a graph
        object. Default is 40.
    min_size_with_proposals : float
        Minimum path length (in microns) required for a fragment to have
        proposals generated from its leaf nodes. Default is 40.
    node_spacing : float
        Physcial spacing (in microns) between nodes. Default is 1.
    prune_depth : int
        Branches in graph less than "prune_depth" microns are pruned. Default
        is 24.
    remove_doubles : bool
        Indication of whether to remove fragments that are likely a double of
        another fragment. Default is True.
    remove_high_risk_merges : bool
        Indication of whether to remove high risk merge sites (i.e. close
        branching points). Default is False.
    trim_endpoints_bool : bool
        Indication of whether trim endpoints of branches with exactly one
        proposal. Default is True.
    verbose : bool
        Indication of whether to display a progress bar. Default is True.
    """

    anisotropy: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    max_proposals_per_leaf: int = 3
    min_size: float = 40.0
    min_size_with_proposals: float = 40.0
    node_spacing: float = 1.0
    proposals_per_leaf: int = 3
    prune_depth: float = 24.0
    remove_doubles: bool = True
    remove_high_risk_merges: bool = False
    trim_endpoints_bool: bool = True
    verbose: bool = True

    def to_dict(self):
        """
        Converts configuration attributes to a dictionary.

        Returns
        -------
        dict
            Dictionary containing configuration attributes.
        """
        attributes = dict()
        for k, v in vars(self).items():
            if isinstance(v, tuple):
                attributes[k] = list(v)
            else:
                attributes[k] = v
        return attributes

    def save(self, path):
        """
        Saves configuration attributes to a JSON file.
        """
        util.write_json(path, self.to_dict())


@dataclass
class MLConfig:
    """
    Configuration class for machine learning model parameters.

    Attributes
    ----------
    batch_size : int
        The number of samples processed in one batch during training or
        inference. Default is 64.
    brightness_clip : int
        Maximum brightness value that image intensities are clipped to.
        Default is 400.
    device : str
        Device to load model onto. Default is "cuda".
    model_name : str
        Name of model used to perform inference. Default is None.
    patch_shape : Tuple[int]
        Shape of image patch expected by vision model. Default is (96, 96, 96).
    shuffle : bool
        Indication of whether to shuffle batches. Default is False
    threshold : float
        A general threshold value used in classification. Default is 0.8.
    transform : bool
        Indication of whether to apply data augmentation to image patches.
        Default is False.
    """

    batch_size: int = 64
    brightness_clip: int = 400
    device: str = "cuda"
    model_name: str = None
    patch_shape: tuple = (96, 96, 96)
    shuffle: bool = False
    threshold: float = 0.8
    transform: bool = False

    def to_dict(self):
        """
        Converts configuration attributes to a dictionary.

        Returns
        -------
        dict
            Dictionary containing configuration attributes.
        """
        attributes = dict()
        for k, v in vars(self).items():
            if isinstance(v, tuple):
                attributes[k] = list(v)
            else:
                attributes[k] = v
        return attributes

    def save(self, path):
        """
        Saves configuration attributes to a JSON file.
        """
        util.write_json(path, self.to_dict())


@dataclass
class Config:
    """
    A configuration class for managing and storing settings related to graph
    and machine learning models.
    """

    def __init__(self, graph_config, ml_config):
        """
        Initializes a Config object which is used to manage settings used to
        run the proofreading pipeline.

        Parameters
        ----------
        graph_config : GraphConfig
            Instance of the "GraphConfig" class that contains configuration
            parameters for graph and proposal operations.
        ml_config : MLConfig
            An instance of the "MLConfig" class that includes configuration
            parameters for machine learning models.
        """
        self.graph = graph_config
        self.ml = ml_config

    def save(self, dir_path):
        """
        Saves configuration attributes to a JSON file.

        dir_path : str
            Path to directory to save JSON file.
        """
        
        self.graph.save(os.path.join(dir_path, "metadata_graph.json"))
        self.ml.save(os.path.join(dir_path, "metadata_ml.json"))
