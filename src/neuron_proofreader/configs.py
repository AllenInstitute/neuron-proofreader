"""
Created on Frid Sept 15 16:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Connfiguration classes used for setting storing parameters used in neuron
proofreading pipelines.

"""

from abc import ABC
from dataclasses import asdict, dataclass
from typing import Tuple

import os

from neuron_proofreader.utils import util


class Config(ABC):

    def to_dict(self):
        """
        Converts configuration attributes to a dictionary.

        Returns
        -------
        attrs : dict
            Dictionary containing configuration attributes.
        """
        attrs = asdict(self)
        for k, v in attrs.items():
            if isinstance(v, tuple):
                attrs[k] = list(v)
        return attrs

    def save(self, output_dir):
        """
        Saves configuration attributes to a JSON file.

        dir_path : str
            Path to directory to save JSON file.
        """
        path = os.path.join(output_dir, f"{self.name}.json")
        util.write_json(path, self.to_dict())


@dataclass
class GraphConfig(Config):
    """
    Configuration class for skeleton graph parameters.

    Attributes
    ----------
    anisotropy : Tuple[float]
        Scaling factors used to transform physical to image coordinates.
    min_cable_length : float
        Minimum path length (in microns) of SWC files loaded into graph.
    node_spacing : float
        Physcial spacing (in microns) between nodes.
    prune_depth : int
        ...
    remove_doubles : bool
        Indication of whether to remove fragments that are likely a double of
        another.
    verbose : bool
        Indication of whether to display a progress bar.
    """

    anisotropy: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    min_cable_length: float = 0.0
    min_swc_pts: int = 1
    name: str = "graph_config"
    node_spacing: float = 1.0
    prune_depth: float = 20.0
    remove_doubles: bool = True
    use_anisotropy: bool = True
    verbose: bool = False


@dataclass
class ImageConfig(Config):
    """
    Configuration class for image processing parameters.

    Attributes
    ----------
    brightness_clip : int
        Intensity value that voxel brightness is clipped to.
    percentiles : Tuple[float], optional
        Percentiles used to normalize patches.
    patch_shape : Tuple[int]
        Shape of patch to be read from image.
    transform : bool
        Indication of whether to use image augmentation.
    """

    brightness_clip: int = 400
    name: str = "image_config"
    percentiles: Tuple[float, float] = (1, 99.5)
    patch_shape: Tuple[int, int, int] = (128, 128, 128)
    transform: bool = False

    def set_train_mode(self):
        self.transform = True


@dataclass
class ProposalsConfig(Config):
    """
    Configuration class for skeleton graph parameters.

    Attributes
    ----------
    allow_nonleaf_proposals : bool
        Indication of whether to generate proposals between leaf and nodes
        with degree 2.
    proposals_per_leaf : int
        Maximum number of proposals generated at leaf nodes.
    trim_endpoints_bool : bool
        Indication of whether trim endpoints of isolated leaf-to-leaf
        proposals.
    """

    allow_nonleaf_proposals: bool = False
    proposals_per_leaf: int = 3
    trim_endpoints_bool: bool = True
