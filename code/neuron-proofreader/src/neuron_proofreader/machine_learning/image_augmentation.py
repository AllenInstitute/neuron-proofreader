"""
Created on Tue Jan 21 14:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Routines for applying image augmentation during training.

"""

from scipy.ndimage import gaussian_filter, rotate, zoom

import numpy as np
import random


# --- Image Augmentation ---
class ImageTransforms:
    """
    Class that applies a sequence of transforms to a 3D image and segmentation
    patch.
    """

    def __init__(self):
        """
        Initializes an ImageTransforms instance that applies augmentation to
        an image and segmentation patch.
        """
        # Instance attributes
        self.geometric_transform = [RandomFlip3D(), RandomRotation3D()]
        self.intensity1_transform = [RandomNoise3D(), RandomContrast3D()]
        self.intensity2_transform = [RandomSmooth3D(), RandomContrast3D()]

    def __call__(self, patches):
        """
        Applies geometric transforms to the input image and segmentation
        patch.

        Parameters
        ----------
        patches : numpy.ndarray
            Image with the shape (2, H, W, D), where the first channel is the
            raw image and second is a mask.
        """
        # Apply geometric transform
        patches = self.apply(patches, self.geometric_transform)

        # Apply intensity transform
        transform = random.choice(
            [
                self.intensity1_transform,
                self.intensity2_transform,
            ]
        )
        patches = self.apply(patches, transform)
        patches = np.clip(patches, 0, 1)
        return patches

    def apply(self, patches, transforms):
        for transform in transforms:
            patches = transform(patches)
        return patches


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
            Axes along which to flip the image. Default is (0, 1, 2).
        """
        self.axes = axes

    def __call__(self, patches):
        """
        Applies random flipping to the input image and segmentation patch.

        Parameters
        ----------
        patches : numpy.ndarray
            Image with the shape (2, H, W, D), where the first channel is the
            raw image and second is a mask.
        """
        for axis in self.axes:
            if random.random() > 0.5:
                patches[0] = np.flip(patches[0], axis=axis)
                patches[1] = np.flip(patches[1], axis=axis)
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
            Maximum angle of rotation. Default is (-90, 90).
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
            Image with the shape (2, H, W, D), where the first channel is the
            raw image and second is a mask.
        """
        for axes in self.axes:
            if random.random() < 0.5:
                angle = random.uniform(*self.angles)
                patches[0] = self.rotate3d(patches[0], angle, axes, False)
                patches[1] = self.rotate3d(patches[1], angle, axes, True)
        return patches

    @staticmethod
    def rotate3d(img, angle, axes, is_mask=False):
        """
        Rotates a 3D image patch around the specified axes by a given angle.

        Parameters
        ----------
        img : numpy.ndarray
            Image to be rotated.
        angle : float
            Angle (in degrees) by which to rotate the image patch around the
            specified axes.
        axes : Tuple[int]
            Two axes of rotation.
        is_mask : bool, optional
            True if the image is a mask.
        """
        multipler = 4 if is_mask else 1
        img = rotate(
            multipler * img,
            angle,
            axes=axes,
            mode="grid-mirror",
            reshape=False,
            order=0,
        )
        img /= multipler
        return img


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
            Image with the shape (2, H, W, D), where the first channel is the
            raw image and second is a mask.

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
        patches[0] = zoom(patches[0], zoom_factors, order=1)
        patches[1] = zoom(patches[1], zoom_factors, order=0)
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
            Range of contrast factors. Default is (0.8, 1.2).
        """
        self.factor_range = factor_range

    def __call__(self, patches):
        """
        Applies contrast to an image.

        Parameters
        ----------
        Image with the shape (2, H, W, D), where the first channel is the
            raw image and second is a mask.

        Returns
        -------
        numpy.ndarray
            Contrasted image.
        """
        factor = random.uniform(*self.factor_range)
        patches[0] = patches[0] * factor
        return patches


class RandomNoise3D:
    """
    Adds random Gaussian noise to a 3D image.
    """

    def __init__(self, max_std=0.2):
        """
        Initializes a RandomNoise3D transformer.

        Parameters
        ----------
        max_std : float, optional
            Maximum standard deviation of the Gaussian noise distribution.
            Default is 0.3.
        """
        self.max_std = max_std

    def __call__(self, patches):
        """
        Adds Gaussian noise to the input 3D image.

        Parameters
        ----------
        patches : numpy.ndarray
            Image with the shape (2, H, W, D), where the first channel is the
            raw image and second is a mask.
        """
        std = self.max_std * random.random()
        patches[0] += np.random.uniform(-std, std, patches[0].shape)
        return patches


class RandomSmooth3D:
    """
    Applies Gaussian smoothing to a 3D image.
    """

    def __init__(self, max_sigma=1):
        """
        Initializes a GaussianSmooth3D transformer.

        Parameters
        ----------
        max_sigma : float, optional
            Maximum standard deviation of the Gaussian kernel.
            Default is 0.8.
        """
        self.max_sigma = max_sigma

    def __call__(self, patches):
        """
        Applies Gaussian smoothing to an image.

        Parameters
        ----------
        patches : numpy.ndarray
            Image with the shape (2, H, W, D), where the first channel is the
            raw image and second is a mask.

        Returns
        -------
        numpy.ndarray
            Smoothed image.
        """
        sigma = random.uniform(0, self.max_sigma)
        patches[0] = gaussian_filter(patches[0], sigma=sigma)
        return patches
