"""
Created on Sat Sept 16 16:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

This module defines a set of configuration classes used for setting up various
aspects of a system involving graphs, proposals, and machine learning (ML).

"""
from dataclasses import dataclass
from typing import Tuple


@dataclass
class GraphConfig:
    """
    Represents configuration settings related to graph properties and
    proposals generated.

    Attributes
    ----------
    anisotropy : Tuple[float], optional
        Scaling factors applied to xyz coordinates to account for anisotropy
        of microscope. Note this instance of "anisotropy" is only used while
        reading SWC files. Default is (1.0, 1.0, 1.0).
    min_size : float, optional
        Minimum path length (in microns) of swc files which are stored as
        connected components in the FragmentsGraph. Default is 30.
    min_size_with_proposals : float, optional
        Minimum fragment path length required for proposals. Default is 0.
    node_spacing : int, optional
        Spacing (in microns) between nodes. Default is 5.
    proposals_per_leaf : int
        Maximum number of proposals generated for each leaf. Default is 3.
    prune_depth : int, optional
        Branches in graph less than "prune_depth" microns are pruned. Default
        is 16.
    remove_doubles : bool, optional
        ...
    remove_high_risk_merges : bool, optional
        Indication of whether to remove high risk merge sites (i.e. close
        branching points). Default is False.
    trim_endpoints_bool : bool, optional
        Indication of whether to endpoints of branches with exactly one
        proposal. Default is True.
    """

    anisotropy: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    max_proposals_per_leaf: int = 3
    min_size: float = 40.0
    min_size_with_proposals: float = 40.0
    node_spacing: int = 1
    proposals_per_leaf: int = 3
    prune_depth: float = 24.0
    remove_doubles: bool = True
    remove_high_risk_merges: bool = False
    search_radius: float = 20.0
    trim_endpoints_bool: bool = True
    verbose: bool = True


@dataclass
class MLConfig:
    """
    Configuration class for machine learning model parameters.

    Attributes
    ----------
    batch_size : int
        The number of samples processed in one batch during training or
        inference. Default is 64.
    threshold : float
        A general threshold value used for classification. Default is 0.6.
    """
    batch_size: int = 64
    brightness_clip: int = 400
    device: str = "cuda"
    patch_shape: tuple = (96, 96, 96)
    shuffle: bool = False
    transform: bool = False
    threshold: float = 0.8


class Config:
    """
    A configuration class for managing and storing settings related to graph
    and machine learning models.
    """

    def __init__(self, graph_config: GraphConfig, ml_config: MLConfig):
        """
        Initializes a Config object which is used to manage settings used to
        run the proofreading pipeline.

        Parameters
        ----------
        graph_config : GraphConfig
            Instance of the "GraphConfig" class that contains configuration
            parameters for graph and proposal operations, such as anisotropy,
            node spacing, and other graph-specific settings.
        ml_config : MLConfig
            An instance of the "MLConfig" class that includes configuration
            parameters for machine learning models.
        """
        self.graph = graph_config
        self.ml = ml_config
