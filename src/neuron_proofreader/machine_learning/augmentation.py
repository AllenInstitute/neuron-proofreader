"""
Created on Tue Jan 21 14:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Routines for applying image augmentation during training.

"""

from scipy.ndimage import rotate, zoom

import numpy as np
import random


class ImageTransforms:
    """
    Class that applies a sequence of transforms to a 3D image and
    segmentation patch.
    """
    def __init__(self, use_geometric=True, use_intensity=True):
        """
        Initializes an ImageTransforms instance that applies augmentation to
        an image and segmentation patch.

        Parameters
        ----------
        use_geometric : bool, optional
            Indication of whether to use geometric transforms. Default is
            True.
        use_intensity : bool, optional
            Indication of whether to use transforms that alter the image
            intensity distribution. Default is True.
        """
        # Instance attributes
        self.transforms = list()
        if use_geometric:
            self.transforms.extend([RandomFlip3D(), RandomRotation3D()])

        if use_intensity:
            self.transforms.extend([RandomNoise3D(), RandomContrast3D()])

    def __call__(self, patches):
        """
        Applies geometric transforms to the input image and segmentation
        patch.

        Parameters
        ----------
        patches : numpy.ndarray
            Image with the shape (2, H, W, D), where "patches[0, ...]" is from
            the input image and "patches[1, ...]" is from the segmentation.
        """
        for transform in self.transforms:
            transform(patches)


# --- Geometric Transforms ---
class RandomFlip3D:
    """
    Randomly flips a 3D image along one or more axes.
    """

    def __init__(self, axes=(0, 1, 2)):
        """
        Initializes a RandomFlip3D transformer.

        Parameters
        ----------
        axes : Tuple[float], optional
            Tuple of integers representing the axes along which to flip the
            image. Default is (0, 1, 2).
        """
        self.axes = axes

    def __call__(self, patches):
        """
        Applies random flipping to the input image and segmentation patch.

        Parameters
        ----------
        patches : numpy.ndarray
            Image with the shape (2, H, W, D), where "patches[0, ...]" is from
            the input image and "patches[1, ...]" is from the segmentation.

        Returns
        -------
        patches : numpy.ndarray
            Flipped 3D image and segmentation patch.
        """
        for axis in self.axes:
            if random.random() > 0.5:
                patches[0, ...] = np.flip(patches[0, ...], axis=axis)
                patches[1, ...] = np.flip(patches[1, ...], axis=axis)
        return patches


class RandomRotation3D:
    """
    Applies random rotation along a randomly chosen axis.
    """

    def __init__(self, angles=(-90, 90), axes=((0, 1), (0, 2), (1, 2))):
        """
        Initializes a RandomRotation3D transformer.

        Parameters
        ----------
        angles : Tuple[int], optional
            Maximum angle of rotation. Default is (-45, 45).
        axis : Tuple[Tuple[int]], optional
            Axes to apply rotation. Default is ((0, 1), (0, 2), (1, 2))
        """
        self.angles = angles
        self.axes = axes

    def __call__(self, patches):
        """
        Rotates the input image and segmentation patch.

        Parameters
        ----------
        patches : numpy.ndarray
            Image with the shape (2, H, W, D), where "patches[0, ...]" is from
            the input image and "patches[1, ...]" is from the segmentation.

        Returns
        -------
        patches : numpy.ndarray
            Rotated 3D image and segmentation patch.
        """
        for axes in self.axes:
            if random.random() > 0.5:
                angle = random.uniform(*self.angles)
                patches[0, ...] = rotate3d(patches[0, ...], angle, axes)
                patches[1, ...] = rotate3d(patches[1, ...], angle, axes)
        return patches


class RandomScale3D:
    """
    Applies random scaling along each axis.
    """

    def __init__(self, scale_range=(0.9, 1.1)):
        """
        Initializes a RandomScale3D transformer.

        Parameters
        ----------
        scale_range : Tuple[float], optional
            Range of scaling factors. Default is (0.9, 1.1).
        """
        self.scale_range = scale_range

    def __call__(self, patches):
        """
        Applies random rescaling to the input 3D image.

        Parameters
        ----------
        patches : numpy.ndarray
            Image with the shape (2, H, W, D), where "patches[0, ...]" is from
            the input image and "patches[1, ...]" is from the segmentation.

        Returns
        -------
        patches : numpy.ndarray
            Rescaled 3D image and segmentation patch.
        """
        # Sample new image shape
        alpha = np.random.uniform(self.scale_range[0], self.scale_range[1])
        new_shape = (
            int(patches.shape[1] * alpha),
            int(patches.shape[2] * alpha),
            int(patches.shape[3] * alpha),
        )

        # Compute the zoom factors
        shape = patches.shape[1:]
        zoom_factors = [
            new_dim / old_dim for old_dim, new_dim in zip(shape, new_shape)
        ]

        # Rescale images
        patches[0, ...] = zoom(patches[0, ...], zoom_factors, order=3)
        patches[1, ...] = zoom(patches[1, ...], zoom_factors, order=3)
        return patches


# --- Intensity Transforms ---
class RandomContrast3D:
    """
    Adjusts the contrast of a 3D image by scaling voxel intensities.
    """

    def __init__(self, factor_range=(0.8, 1.2)):
        """
        Initializes a RandomContrast3D transformer.

        Parameters
        ----------
        factor_range : Tuple[float], optional
            Tuple of integers representing the range of contrast factors.
            Default is (0.8, 1.2).
        """
        self.factor_range = factor_range

    def __call__(self, img_patch):
        """
        Applies contrast to the input 3D image.

        Parameters
        ----------
        img_patch : numpy.ndarray
            Image to which contrast will be added.

        Returns
        -------
        numpy.ndarray
            Contrasted 3D image.
        """
        factor = random.uniform(*self.factor_range)
        return np.clip(img_patch * factor, 0, 1)


class RandomNoise3D:
    """
    Adds random Gaussian noise to a 3D image.
    """

    def __init__(self, max_std=0.05):
        """
        Initializes a RandomNoise3D transformer.

        Parameters
        ----------
        max_std : float, optional
            Maximum standard deviation of the Gaussian noise distribution.
            Default is 0.05.
        """
        self.max_std = max_std

    def __call__(self, img_patch):
        """
        Adds Gaussian noise to the input 3D image.

        Parameters
        ----------
        img_patch : np.ndarray
            Image to which noise will be added.

        Returns
        -------
        img_patch : numpy.ndarray
            Noisy 3D image.
        """
        std = self.max_std * random.random()
        noise = np.random.normal(0, std, img_patch.shape)
        img_patch = img_patch + noise
        return img_patch


# --- Helpers ---
def rotate3d(img_patch, angle, axes):
    """
    Rotates a 3D image patch around the specified axes by a given angle.

    Parameters
    ----------
    img_patch : numpy.ndarray
        Image to be rotated.
    angle : float
        Angle (in degrees) by which to rotate the image patch around the
        specified axes.
    axes : Tuple[int]
        Tuple representing the two axes of rotation.

    Returns
    -------
    img_patch : numpy.ndarray
        Rotated 3D image patch.
    """
    img_patch = rotate(
        img_patch,
        angle,
        axes=axes,
        mode="grid-mirror",
        reshape=False,
        order=0,
    )
    return img_patch
