"""
Created on Jan 26 5:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Routines for loading image patches from whole-brain exaSPIM datasets.

"""

from concurrent.futures import (
    as_completed,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
)
from torch.utils.data import IterableDataset

import fastremap
import numpy as np
import random
import torch

from neuron_proofreader.utils.img_util import TensorStoreReader
from neuron_proofreader.utils import swc_util, util


# --- Custom Datasets ---
class ExaspimDataset(IterableDataset):
    """
    A PyTorch Dataset for sampling 3D patches from whole-brain images. The
    dataset's __getitem__ method returns both raw image and segmentation
    patches. Optionally, the patch sampling maybe biased towards foreground
    regions.
    """

    def __init__(
        self,
        patch_shape,
        brightness_clip=400,
        boundary_buffer=5000,
        foreground_sampling_rate=0.5,
        n_examples_per_epoch=1000,
        normalization_percentiles=(1, 99.9),
        prefetch_foreground_sampling=32,
    ):
        """
        Instantiates an ExaspimDataset object.

        Parameters
        ----------
        patch_shape : Tuple[int]
            Shape of image patches to be read from image and segmentation.
        brightness_clip : int, optional
            Brightness intensity used as upper limit of image patch. Default
            is 400.
        boundary_buffer : int, optional
            Image patches are sampled at least "boundary_buffer" voxels away
            from boundary along each dimension. Default is 5000.
        foreground_sampling_rate : float, optional
            Rate at which image patches containing foreground objects are
            sampled. Default is 0.5.
        n_examples_per_epoch : int, optional
            Number of examples generated for each epoch. Default is 1000.
        normalization_percentiles : Tuple[float], optional
            Upper and lower bounds of percentiles used to normalize image.
            Default is (1, 99.9).
        prefetch_foreground_sampling : int, optional
            Number of image patches that are preloaded during foreground
            search in "self.sample_segmentation_voxel" and
            "self.sample_bright_voxel". Default is 32.
        """
        # Call parent class
        super(ExaspimDataset, self).__init__()

        # Class attributes
        self.boundary_buffer = boundary_buffer
        self.brightness_clip = brightness_clip
        self.foreground_sampling_rate = foreground_sampling_rate
        self.n_examples_per_epoch = n_examples_per_epoch
        self.normalization_percentiles = normalization_percentiles
        self.patch_shape = patch_shape
        self.prefetch_foreground_sampling = prefetch_foreground_sampling
        self.swc_reader = swc_util.Reader()

        # Data structures
        self.imgs = dict()
        self.segmentations = dict()
        self.skeletons = dict()

    # --- Ingest Data ---
    def ingest_brain(
        self, brain_id, img_path, segmentation_path=None, swc_path=None
    ):
        """
        Loads a brain image, label mask, and skeletons, then stores each in
        internal dictionaries.

        Parameters
        ----------
        brain_id : str
            Unique identifier for the brain corresponding to the image.
        img_path : str
            Path to whole-brain image to be read.
        segmentation_path : str, optional
            Path to segmentation. Default is None.
        swc_path : str, optional
            Path to SWC files. Default is None.
        """
        # Load data
        self.imgs[brain_id] = TensorStoreReader(img_path)
        self._load_segmentation(brain_id, segmentation_path)
        self._load_swcs(brain_id, swc_path)

        # Check image shapes
        shape1 = self.imgs[brain_id].shape()[2::]
        shape2 = self.segmentations[brain_id].shape()
        assert shape1 == shape2, f"img_shape={shape1}, mask_shape={shape2}"

    def _load_segmentation(self, brain_id, path):
        if path:
            self.segmentations[brain_id] = TensorStoreReader(path)

    def _load_swcs(self, brain_id, swc_path):
        if swc_path:
            # Initializations
            swc_dicts = self.swc_reader(swc_path)
            n_points = np.sum([len(d["xyz"]) for d in swc_dicts])

            # Extract skeleton voxels
            if n_points > 0:
                start = 0
                skeletons = np.zeros((n_points, 3), dtype=np.int32)
                for swc_dict in swc_dicts:
                    end = start + len(swc_dict["xyz"])
                    skeletons[start:end] = swc_dict["xyz"]
                    start = end
                self.skeletons[brain_id] = skeletons[:, [2, 1, 0]]

    # --- Sample Image Patches ---
    def __iter__(self):
        """
        Returns a pair of noisy and BM4D-denoised image patches, normalized
        according to percentile-based scaling.

        Returns
        -------
        img : numpy.ndarray
            Patch from raw image
        mask : numpy.ndarray
            Binarized mask from segmentation.
        """
        for _ in range(self.n_examples_per_epoch):
            # Get example
            brain_id = self.sample_brain()
            voxel = self.sample_voxel(brain_id)
            img = self.read_image(brain_id, voxel)
            mask = self.read_segmentation(brain_id, voxel)

            # Prepocess patches
            img = self.preprocess_image(img)
            mask = self.preprocess_mask(mask)
            yield img, mask

    def sample_brain(self):
        """
        Samples a brain ID from the loaded images.

        Returns
        -------
        brain_id : str
            Unique identifier of the sampled whole-brain.
        """
        return util.sample_once(self.imgs.keys())

    def sample_voxel(self, brain_id):
        """
        Samples a voxel from a brain volume, either foreground or interior.

        Parameters
        ----------
        brain_id : str
            Unique identifier of the sampled whole-brain.

        Returns
        -------
        Tuple[int]
            Voxel coordinate chosen according to the foreground or interior
            sampling strategy.
        """
        if random.random() < self.foreground_sampling_rate:
            return self.sample_foreground_voxel(brain_id)
        else:
            return self.sample_interior_voxel(brain_id)

    def sample_foreground_voxel(self, brain_id):
        """
        Samples a voxel likely to be part of the foreground of a neuron.

        Parameters
        ----------
        brain_id : str
            Unique identifier of a whole-brain.

        Returns
        -------
        Tuple[int]
            Voxel coordinate representing a likely foreground location.
        """
        if brain_id in self.skeletons and np.random.random() > 0.5:
            return self.sample_skeleton_voxel(brain_id)
        elif brain_id in self.segmentations:
            return self.sample_segmentation_voxel(brain_id)
        else:
            return self.sample_bright_voxel(brain_id)

    def sample_interior_voxel(self, brain_id):
        """
        Samples a random voxel coordinate from the interior of a 3D image
        volume, avoiding boundary regions.

        Parameters
        ----------
        brain_id : str
            Unique identifier of a whole-brain.

        Returns
        -------
        Tuple[int]
            Voxel coordinate sampled uniformly at random within the valid
            interior region of the image volume.
        """
        voxel = list()
        for s in self.imgs[brain_id].shape()[2::]:
            upper = s - self.boundary_buffer
            voxel.append(random.randint(self.boundary_buffer, upper))
        return tuple(voxel)

    def sample_skeleton_voxel(self, brain_id):
        """
        Samples a voxel coordinate near a skeleton point.

        Parameters
        ----------
        brain_id : str
            Unique identifier of a whole-brain.

        Returns
        -------
        Tuple[int]
            Voxel coordinate near a skeleton point.
        """
        idx = random.randint(0, len(self.skeletons[brain_id]) - 1)
        shift = np.random.randint(0, 16, size=3)
        return tuple(self.skeletons[brain_id][idx] + shift)

    def sample_segmentation_voxel(self, brain_id):
        """
        Sample a voxel coordinate whose corresponding segmentation patch
        contains a sufficiently large object.

        Parameters
        ----------
        brain_id : str
            Identifier for the image volume which must be a key in
            "self.segmentations".

        Returns
        -------
        best_voxel : Tuple[int]
            Voxel coordinate whose patch contains a sufficiently large object
            or had the largest object after 5 * self.prefetch attempts.
        """
        best_volume = 0
        best_voxel = self.sample_interior_voxel(brain_id)
        cnt = 0
        with ThreadPoolExecutor() as executor:
            while best_volume < 1600:
                # Read random image patches
                pending = dict()
                for _ in range(self.prefetch_foreground_sampling):
                    voxel = self.sample_interior_voxel(brain_id)
                    thread = executor.submit(
                        self.read_segmentation, brain_id, voxel
                    )
                    pending[thread] = voxel

                # Check if labels patch has large enough object
                for thread in as_completed(pending.keys()):
                    voxel = pending.pop(thread)
                    labels_patch = thread.result()
                    vals, cnts = fastremap.unique(
                        labels_patch, return_counts=True
                    )

                    if len(cnts) > 1:
                        volume = np.max(cnts[1:])
                        if volume > best_volume:
                            best_voxel = voxel
                            best_volume = volume

                # Check number of tries
                cnt += 1
                if cnt > 5:
                    break
        return best_voxel

    def sample_bright_voxel(self, brain_id):
        """
        Samples a voxel coordinate whose image patch is sufficiently bright.

        Parameters
        ----------
        brain_id : str
            Unique identifier of a whole-brain.

        Returns
        -------
        best_voxel : Tuple[int]
            Voxel coordinate whose patch is sufficiently bright or is the
            highest observed brightness after 4 * self.prefetch attempts.
        """
        best_brightness = 0
        best_voxel = self.sample_interior_voxel(brain_id)
        cnt = 0
        with ThreadPoolExecutor() as executor:
            while best_brightness < 1000:
                # Read random image patches
                pending = dict()
                for _ in range(self.prefetch_foreground_sampling):
                    voxel = self.sample_interior_voxel(brain_id)
                    thread = executor.submit(
                        self.read_image, brain_id, voxel
                    )
                    pending[thread] = voxel

                # Check if image patch is bright enough
                for thread in as_completed(pending.keys()):
                    voxel = pending.pop(thread)
                    img_patch = thread.result()
                    brightness = np.sum(img_patch > 100)
                    if brightness > best_brightness:
                        best_voxel = voxel
                        best_brightness = brightness

                # Check number of tries
                cnt += 1
                if cnt > 5:
                    break
        return best_voxel

    # --- Helpers ---
    def __len__(self):
        pass

    def preprocess_image(self, img):
        """
        Preprocesses the given image by clipping the intensity values and
        normalizing with a percentile-based scheme.

        Parameters
        ----------
        img : numpy.ndarray
            Image to be normalized

        Returns
        -------
        img : numpy.ndarray
            Normalized image.
        """
        # Clip
        img = np.minimum(img, self.brightness_clip)

        # Normalize
        mn, mx = np.percentile(img, self.normalization_percentiles)
        img = (img - mn) / (mx - mn + 1e-8)
        return np.clip(img, 0, 1)

    def preprocess_mask(self, mask):
        """
        Preprocesses the given segmentation mask by binarizing it.

        Parameters
        ----------
        img : numpy.ndarray
            Image to be normalized

        Returns
        -------
        img : numpy.ndarray
            Normalized image.
        """

        return (mask > 0).astype(int)

    def read_image(self, brain_id, voxel):
        """
        Reads an image patch from the given brain at the specified location.

        Parameters
        ----------
        brain_id : str
            Unique identifier of whole-brain dataset.
        voxel : Tuple[int]
            Center of image patch to be read.

        Returns
        -------
        numpy.ndarray
            Image patch.
        """
        return self.imgs[brain_id].read(voxel, self.patch_shape)

    def read_segmentation(self, brain_id, voxel):
        """
        Reads a segmentation patch from the given brain at the specified'
        location.

        Parameters
        ----------
        brain_id : str
            Unique identifier of whole-brain dataset.
        voxel : Tuple[int]
            Center of image patch to be read.

        Returns
        -------
        numpy.ndarray
            Segmentation patch.
        """
        return self.segmentations[brain_id].read(voxel, self.patch_shape)


# --- Custom Dataloader ---
class DataLoader:
    """
    DataLoader that uses multithreading to fetch image patches from the cloud
    to form batches.

    Attributes
    ----------
    dataset : torch.utils.data.Dataset
        Dataset to iterated over.
    batch_size : int
        Number of examples in each batch.
    patch_shape : Tuple[int]
        Shape of image patch expected by the model.
    """

    def __init__(self, dataset, batch_size=16):
        """
        Instantiates a DataLoader object.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            Dataset to iterated over.
        batch_size : int, optional
            Number of examples in each batch. Default is 16.
        """
        # Instance attributes
        self.dataset = dataset
        self.batch_size = batch_size
        self.patch_shape = dataset.patch_shape

    def __iter__(self):
        """
        Iterates over the dataset and yields batches of examples.

        Returns
        -------
        iterator
            Yields batches of examples.
        """
        for idx in range(0, len(self.dataset), self.batch_size):
            yield self._load_batch(idx)

    def _load_batch(self, start_idx):
        # Compute batch size
        n_remaining_examples = len(self.dataset) - start_idx
        batch_size = min(self.batch_size, n_remaining_examples)

        # Generate batch
        with ProcessPoolExecutor() as executor:
            # Assign processs
            processes = list()
            for idx in range(start_idx, start_idx + batch_size):
                processes.append(
                    executor.submit(self.dataset.__getitem__, idx)
                )

            # Process results
            img_patches = np.zeros((batch_size, 1,) + self.patch_shape)
            mask_patches = np.zeros((batch_size, 1,) + self.patch_shape)
            for i, process in enumerate(as_completed(processes)):
                img, mask = process.result()
                img_patches[i, 0, ...] = img
                mask_patches[i, 0, ...] = mask
        return to_tensor(img_patches), to_tensor(mask_patches)


# --- Helpers ---
def to_tensor(arr):
    """
    Converts the given NumPy array to a torch tensor.

    Parameters
    ----------
    arr : numpy.ndarray
        Array to be converted.

    Returns
    -------
    torch.Tensor
        Array converted to a torch tensor.
    """
    return torch.tensor(arr, dtype=torch.float)
