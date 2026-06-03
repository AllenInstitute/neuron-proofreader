"""
Created on Tue Jan 13 15:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code for parallelizing reading image patches from the cloud.

"""

from abc import ABC, abstractmethod

import numpy as np
import tensorstore as ts

from neuron_proofreader.machine_learning.image_augmentation import (
    ImageTransforms
)
from neuron_proofreader.utils import geometry_util, img_util, util


# --- Image Reading ---
class TensorStoreImage:
    """
    Class that reads images with the TensorStore library.
    """

    def __init__(self, img_path):
        """
        Instantiates a TensorStoreImage object.

        Parameters
        ----------
        img_path : str
            Path to image.
        """
        # Load image
        bucket_name, inner_path = util.parse_cloud_path(img_path)
        self.img = ts.open(
            {
                "driver": img_util.get_driver(img_path),
                "kvstore": {
                    "driver": img_util.get_storage_driver(img_path),
                    "bucket": bucket_name,
                    "path": inner_path,
                },
                "context": {
                    "cache_pool": {"total_bytes_limit": 1000000000},
                    "cache_pool#remote": {"total_bytes_limit": 1000000000},
                    "data_copy_concurrency": {"limit": 8},
                },
                "recheck_cached_data": "open",
            }
        ).result()

        # Check for Google segmentation
        if "from_google" in img_path:
            self.img = self.img[ts.d[:].transpose[3, 2, 1, 0]]

        # Check dimensions
        while self.img.ndim < 5:
            self.img = self.img[ts.newaxis, ...]

    def read(self, voxel, shape):
        """
        Reads a patch from an image given a voxel coordinate and patch shape.

        Parameters
        ----------
        voxel : Tuple[int]
            Center of image patch to be read.
        shape : Tuple[int]
            Shape of image patch to be read.

        Returns
        -------
        numpy.ndarray
            Image patch.
        """
        s = img_util.get_slices(voxel, shape)
        return self.img[(0, 0, *s)].read().result()

    def shape(self):
        """
        Gets the shape of image.

        Returns
        -------
        Tuple[int]
            Shape of image.
        """
        return self.img.shape


# --- Patch Loading ---
class PatchLoader(ABC):
    """
    A class for reading image patches and generating segment masks.
    """

    max_voxel_shift = 5

    def __init__(
        self,
        graph,
        img_path,
        brightness_clip=400,
        normalization_percentiles=(1, 99.5),
        patch_shape=(128, 128, 128),
        use_transform=False,
    ):
        """
        Instantiates a PatchLoader object.

        Parameters
        ----------
        graph : SkeletonGraph
            Graph used to compute patch voxel coordinates.
        img_path : str
            Path to whole-brain image.
        brightness_clip : int, optional
            Intensity value that voxel brightnesses are clipped to.
        normalization_percentiles : Tuple[float], optional
            Percentiles used to normalize patches. Default is (1, 99.5).
        patch_shape : Tuple[int], optional
            Shape of patch to be read from image. Default is (128, 128, 128).
        """
        # Instance attributes
        self.brightness_clip = brightness_clip
        self.graph = graph
        self.patch_shape = patch_shape
        self.percentiles = normalization_percentiles

        # Image operations
        self.img = TensorStoreImage(img_path)
        self.transform = ImageTransforms() if use_transform else None

    # --- Abstract Interface ---
    @abstractmethod
    def __call__(self):
        """
        Abstract method to be implemented by subclasses
        """
        pass

    @abstractmethod
    def compute_patch_specs(self):
        """
        Abstract method to be implemented by subclasses.
        """
        pass

    @abstractmethod
    def create_mask(self):
        """
        Abstract method to be implemented by subclasses.
        """
        pass

    # --- Core Routines ---
    def annotate_foreground(self, mask, nodes, offset, fill_val=1):
        visited = set()
        for i in nodes:
            voxel_i = self.graph.node_local_voxel(i, offset)
            for j in self.graph.neighbors(i):
                if frozenset({i, j}) not in visited and j in nodes:
                    voxel_j = self.graph.node_local_voxel(j, offset)
                    voxels = geometry_util.make_digital_line(voxel_i, voxel_j)
                    img_util.annotate_voxels(mask, voxels, fill_val=fill_val)
                    visited.add(frozenset({i, j}))

    def annotate_fragment(self, mask, subgraph, offset, fill_val=1):
        for node1, node2 in subgraph.edges:
            # Get local voxel coordinates
            voxel1 = subgraph.node_local_voxel(node1, offset)
            voxel2 = subgraph.node_local_voxel(node2, offset)

            # Populate mask
            voxels = geometry_util.make_digital_line(voxel1, voxel2)
            img_util.annotate_voxels(mask, voxels, fill_val=fill_val)

    def read_image(self, center, shape):
        """
        Reads the image patch specified by the given center and shape.

        Parameters
        ----------
        center : Tuple[int]
            Center of image patch to be read.
        shape : Tuple[int]
            Center of image patch to be read.

        Returns
        -------
        patch : numpy.ndarray
            Preprocessed image patch.
        """
        patch = self.img.read(center, shape)
        patch = np.minimum(patch, self.brightness_clip)
        patch = img_util.normalize(patch, percentiles=self.percentiles)
        return patch

    # --- Helpers ---
    def adjust_voxel(self, voxel):
        if self.transform:
            voxel += np.random.randint(
                -self.max_voxel_shift, self.max_voxel_shift + 1, size=3
            )
        return voxel

    @staticmethod
    def stack(img, mask):
        try:
            patches = np.stack([img, mask], axis=0)
        except ValueError:
            img = img_util.pad_to_shape(img, mask.shape)
            patches = np.stack([img, mask], axis=0)
        return patches


class DetectionPatchLoader(PatchLoader):

    def __init__(
        self,
        graph,
        img_path,
        brightness_clip=400,
        normalization_percentiles=(1, 99.5),
        patch_shape=(128, 128, 128),
        use_transform=False,
    ):
        # Call parent class
        super().__init__(
            graph,
            img_path,
            brightness_clip=brightness_clip,
            normalization_percentiles=normalization_percentiles,
            patch_shape=patch_shape,
            use_transform=use_transform,
        )

    # --- Implementation of Abstract Inferface ---
    def __call__(self, node):
        # Get patches
        center, shape = self.compute_patch_specs(node)
        img = self.read_image(center, shape)
        mask = self.create_mask(center, shape, node)
        patches = self.stack(img, mask)

        # Check whether to apply image augmentation
        if self.transform:
            patches = self.transform(patches)
        return patches

    def compute_patch_specs(self, node):
        voxel = self.graph.node_voxel(node)
        voxel = self.adjust_voxel(voxel)
        return voxel, self.patch_shape

    def create_mask(self, center, shape, node):
        # Initializations
        offset = img_util.get_offset(center, shape)
        depth = np.sqrt(2) * np.max(shape) / (2 * self.graph.anisotropy.min())
        nodes = self.get_foreground_nodes(node, depth)
        subgraph = self.graph.rooted_subgraph(node, depth)

        # Annotate mask
        mask = np.zeros(shape)
        self.annotate_foreground(mask, nodes, offset, fill_val=0.5)
        self.annotate_fragment(mask, subgraph, offset, fill_val=1)
        return mask

    # --- Helpers ---
    def get_foreground_nodes(self, node, radius):
        xyz = self.graph.node_xyz[node]
        nodes = self.graph.kdtree.query_ball_point(xyz, radius)
        return nodes


class ProposalPatchLoader(PatchLoader):
    pass
