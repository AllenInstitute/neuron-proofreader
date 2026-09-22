"""
Created on Tue Jan 13 15:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code for parallelizing reading image patches from the cloud.

"""

from abc import ABC, abstractmethod

import numpy as np
import tensorstore as ts

from neuron_proofreader.configs import ImageConfig
from neuron_proofreader.machine_learning.image_augmentation import (
    ImageTransforms,
)
from neuron_proofreader.utils import geometry_util, img_util, util


class TensorStoreImage:
    """
    Class that reads images with the TensorStore library.

    Attributes
    ----------
    cache_bytes : int
        Size of the decoded-chunk cache.
    """

    cache_bytes = 8_000_000_000

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
        self.img_path = img_path
        self.img = ts.open(
            {
                "driver": img_util.get_driver(img_path),
                "kvstore": {
                    "driver": img_util.get_storage_driver(img_path),
                    "bucket": bucket_name,
                    "path": inner_path,
                },
                "context": {
                    "cache_pool": {"total_bytes_limit": self.cache_bytes},
                    "data_copy_concurrency": {"limit": 8},
                    "gcs_request_concurrency": {"limit": 64},
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
        return self.wait(self.read_async(voxel, shape), voxel, shape)

    def read_async(self, voxel, shape):
        """
        Starts reading a patch without blocking. TensorStore fetches and
        decodes the chunks on its own thread pool, so many reads can be in
        flight without holding a Python thread each.

        Parameters
        ----------
        voxel : Tuple[int]
            Center of image patch to be read.
        shape : Tuple[int]
            Shape of image patch to be read.

        Returns
        -------
        tensorstore.Future
            Future that resolves to the image patch; pass it to "wait".
        """
        s = img_util.get_slices(voxel, shape, self.img.shape[2:])
        return self.img[(0, 0, *s)].read()

    def wait(self, future, voxel, shape):
        """
        Waits for a read started by "read_async" and returns the patch.
        """
        try:
            return future.result()
        except ValueError as e:
            if "OUT_OF_RANGE" in str(e):
                raise ValueError(
                    f"Out-of-bounds read from image: {self.img_path}\n"
                    f"  Requested center voxel: {tuple(voxel)}\n"
                    f"  Requested patch shape:  {tuple(shape)}\n"
                    f"  Image shape:            {tuple(self.img.shape)}\n"
                    f"  Original error: {e}"
                ) from e
            raise

    def shape(self):
        """
        Gets the shape of image.

        Returns
        -------
        Tuple[int]
            Shape of image.
        """
        return self.img.shape


class PatchLoader(ABC):
    """
    A class for reading image patches and generating segment masks.
    """

    max_voxel_shift = 5

    def __init__(self, graph, img_config):
        """
        Instantiates a PatchLoader object.

        Parameters
        ----------
        graph : SkeletonGraph
            Graph used to compute patch voxel coordinates.
        img_config : ImageConfig or None
            Config object with image processing parameters.
        img_path : str
            Path to whole-brain image.
        """
        self.config = img_config or ImageConfig()
        self.graph = graph
        self.img = TensorStoreImage(img_config.img_path)
        self.transform = ImageTransforms() if self.config.transform else None

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

    # --- Core Routines ---
    def annotate_foreground(self, mask, nodes, offset, fill_val=1):
        visited = set()
        for i in nodes:
            # Check whether to visit voxel
            voxel_i = self.graph.node_local_voxel(i, offset)
            if not img_util.is_contained(voxel_i, mask.shape):
                continue

            # Visit neighbors
            for j in self.graph.neighbors(i):
                if frozenset({i, j}) not in visited and j in nodes:
                    voxel_j = self.graph.node_local_voxel(j, offset)
                    voxels = geometry_util.make_digital_line(voxel_i, voxel_j)
                    img_util.annotate_voxels(mask, voxels, fill_val=fill_val)
                    visited.add(frozenset({i, j}))

    def annotate_fragment(self, mask, subgraph, offset, fill_val=1):
        for node1, node2 in subgraph.edge_list():
            # Get local voxel coordinates
            voxel1 = subgraph.node_local_voxel(node1, offset)
            voxel2 = subgraph.node_local_voxel(node2, offset)

            # Populate mask
            voxels = geometry_util.make_digital_line(voxel1, voxel2)
            img_util.annotate_voxels(mask, voxels, fill_val=fill_val)

    def create_mask(self, center, shape, node):
        # Initializations
        offset = img_util.get_offset(center, shape)
        depth = np.sqrt(2) * np.max(shape) / (2 * self.graph.anisotropy.min())
        nodes = self.get_foreground_nodes(node, depth)
        subgraph = self.graph.rooted_subgraph(node, depth)

        # Annotate mask
        mask = np.zeros(shape, dtype=np.float32)
        # self.annotate_foreground(mask, nodes, offset, fill_val=0.5)  TEMP
        self.annotate_fragment(mask, subgraph, offset, fill_val=1)
        return mask

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
        return self.finish_read(self.img.read_async(center, shape), center, shape)

    def finish_read(self, future, center, shape):
        """
        Waits for a patch read started with "self.img.read_async" and
        preprocesses it exactly as "read_image" does.
        """
        patch = self.img.wait(future, center, shape)
        patch = np.minimum(patch, self.brightness_clip)
        return img_util.normalize(patch, percentiles=self.percentiles).astype(
            np.float32
        )

    # --- Helpers ---
    def __getattr__(self, name):
        return getattr(self.config, name)

    def adjust_voxel(self, voxel):
        if self.transform:
            voxel += np.random.randint(
                -self.max_voxel_shift, self.max_voxel_shift + 1, size=3
            )
        return voxel

    def get_foreground_nodes(self, node, radius):
        xyz = self.graph.node_xyz[node]
        nodes = self.graph.kdtree.query_ball_point(xyz, radius)
        return nodes

    @staticmethod
    def stack(img, mask):
        try:
            patches = np.stack([img, mask], axis=0)
        except ValueError:
            img = img_util.pad_to_shape(img, mask.shape)
            patches = np.stack([img, mask], axis=0)
        return patches


class DetectionPatchLoader(PatchLoader):

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
        return node, patches

    def compute_patch_specs(self, node):
        voxel = self.graph.node_voxel(node)
        voxel = self.adjust_voxel(voxel)
        return voxel, self.patch_shape


class DetectionBatchLoader(PatchLoader):

    def __call__(self, nodes):
        # Compute patch info
        center, shape = self.compute_patch_specs(nodes)
        offset = center - shape // 2

        # Load patches
        img = self.read_image(center, shape)
        mask = self.create_mask(center, shape, nodes[len(nodes) // 2])
        return nodes, self.stack(img, mask), offset

    def compute_patch_specs(self, nodes):
        # Compute bounding box
        centers = np.array([self.graph.node_voxel(i) for i in nodes])
        buffer = np.array(self.patch_shape) // 2
        start = centers.min(axis=0) - buffer
        end = centers.max(axis=0) + buffer

        # Image patch location
        shape = (end - start).astype(int)
        center = (start + shape // 2).astype(int)
        return center, shape


class ProposalPatchLoader(PatchLoader):
    """
    Reads image patches centered on proposals. Each patch is a cube sized to
    contain both proposal endpoints plus padding.
    """

    def __init__(self, graph, img_config, padding=40):
        super().__init__(graph, img_config)
        self.padding = padding

    def __call__(self, proposal, read=None):
        """
        Loads the preprocessed image patch of a proposal.

        Parameters
        ----------
        proposal : Frozenset[int]
            Proposal that the patch is centered on.
        read : Tuple[tensorstore.Future, Tuple[int], Tuple[int]], optional
            Pending read for this proposal from "read_async". Default is
            None, in which case the read is issued here.

        Returns
        -------
        Tuple[numpy.ndarray, Tuple[int]]
            Image patch and its offset in the image.
        """
        future, center, shape = read or self.read_async(proposal)
        offset = img_util.get_offset(center, shape)
        img = self.finish_read(future, center, shape)
        return img, offset

    def read_async(self, proposal):
        """
        Starts reading the image patch of a proposal without blocking.

        Returns
        -------
        Tuple[tensorstore.Future, Tuple[int], Tuple[int]]
            Pending read with the patch center and shape, to be passed to
            "__call__".
        """
        center, shape = self.compute_patch_specs(proposal)
        return self.img.read_async(center, shape), center, shape

    def compute_patch_specs(self, proposal):
        node1, node2 = tuple(proposal)
        voxel1 = np.array(self.graph.node_voxel(node1))
        voxel2 = np.array(self.graph.node_voxel(node2))
        center = tuple(((voxel1 + voxel2) / 2).astype(int))
        length = int(np.max(np.abs(voxel2 - voxel1))) + 2 * self.padding
        return center, (length, length, length)
