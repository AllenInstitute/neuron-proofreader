"""
Created on Sat Sept 16 16:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

This module defines a set of configuration classes used for setting up various
aspects of a system involving graphs, proposals, and machine learning (ML).

"""
from dataclasses import dataclass, field
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
    complex_bool : bool
        Indication of whether to generate complex proposals, meaning proposals
        between leaf and non-leaf nodes. Default is False.
    long_range_bool : bool, optional
        Indication of whether to generate simple proposals within a scaled
        distance of "search_radius" from leaves without any proposals. Default
        is False.
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
    smooth_bool : bool, optional
        Indication of whether to smooth branches in the graph. Default is
        True.
    trim_endpoints_bool : bool, optional
        Indication of whether to endpoints of branches with exactly one
        proposal. Default is True.
    """

    anisotropy: Tuple[float] = field(default_factory=tuple)
    complex_bool: bool = False
    long_range_bool: bool = True
    min_size: float = 30.0
    min_size_with_proposals: float = 0
    node_spacing: int = 5
    proposals_per_leaf: int = 3
    prune_depth: float = 24.0
    remove_doubles: bool = False
    remove_high_risk_merges: bool = False
    search_radius: float = 20.0
    smooth_bool: bool = True
    trim_endpoints_bool: bool = True


@dataclass
class MLConfig:
    """
    Configuration class for machine learning model parameters.

    Attributes
    ----------
    batch_size : int
        The number of samples processed in one batch during training or
        inference. Default is 64.
    multiscale : int
        Level in the image pyramid that voxel coordinates must index into.
    threshold : float
        A general threshold value used for classification. Default is 0.6.
    model_type : str
        Type of machine learning model to use. Default is "GraphNeuralNet".
    """
    batch_size: int = 64
    device: str = "cpu"
    lr: float = 1e-4
    multiscale: int = 1
    n_epochs: int = 1000
    threshold: float = 0.6
    transform: bool = False
    validation_split: float = 0.15
    weight_decay: float = 1e-3


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
        self.graph_config = graph_config
        self.ml_config = ml_config
