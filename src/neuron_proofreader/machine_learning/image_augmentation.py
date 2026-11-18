"""
Created on Tue Jan 21 14:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Routines for applying image augmentation during training.

"""

from scipy.ndimage import gaussian_filter

import numpy as np
import random
import threading
import torch
import torch.nn.functional as F

from neuron_proofreader.utils import img_util


# --- Image Augmentation ---
class ImageTransforms:
    """
    Class that applies a sequence of transforms to a 3D image and segmentation
    patch.
    """

    def __init__(self, intensity=True):
        """
        Initializes an ImageTransforms instance that applies augmentation to
        an image and segmentation patch.

        Parameters
        ----------
        intensity : bool, optional
            Whether to apply the intensity transforms, which act on
            normalized values in [0, 1]. False applies the geometric
            transforms only, for pipelines that augment intensities on the
            raw image instead (see RawIntensityTransforms). Default is True.
        """
        # Instance attributes
        self.intensity = intensity
        self.affine = RandomAffine3D()
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
        if self.intensity:
            transform = random.choice(
                [
                    self.intensity1_transform,
                    self.intensity2_transform,
                ]
            )
            patches = self.apply(patches, transform)
            np.clip(patches, 0, 1, out=patches)
        return np.ascontiguousarray(patches)

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
        # Views, not copies; ImageTransforms makes the result contiguous once
        for axis in self.axes:
            if random.random() > 0.5:
                patches = np.flip(patches, axis=axis + 1)
        return patches


class RandomRotation3D:
    """
    Applies random 90-degree rotations along each axis pair.
    """

    def __init__(self, axes=((0, 1), (0, 2), (1, 2))):
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
                k = random.randint(1, 3)
                patches = np.rot90(
                    patches, k=k, axes=(axes[0] + 1, axes[1] + 1)
                )
        return patches


class RandomAffine3D:
    """
    Resamples an image and mask patch to the model's patch shape through a
    random rotation and isotropic scaling about the patch center, in the
    same interpolation pass that otherwise only resizes them. Flips and
    90-degree rotations show a site in 48 axis-aligned orientations at one
    scale; this covers the orientations and sizes in between.

    Voxels sampled from outside the input are zero, which is the background
    level of a normalized patch. The ball inscribed in the input is always
    covered, so only the periphery of the output is affected.
    """

    def __init__(self, scale_range=(0.85, 1.15), p=0.75):
        """
        Initializes a RandomAffine3D transformer.

        Parameters
        ----------
        scale_range : Tuple[float], optional
            Range of the log-uniform magnification. Default is (0.85, 1.15).
        p : float, optional
            Probability of applying the transform. Otherwise the patches are
            only resized, as they are at inference, so the model also trains
            on patches without empty corners. Default is 0.75.
        """
        self.log_range = (np.log(scale_range[0]), np.log(scale_range[1]))
        self.p = p

    def __call__(self, img, mask, shape):
        """
        Resamples the image and mask to "shape".

        Parameters
        ----------
        img : numpy.ndarray
            Normalized image patch with a cubic shape.
        mask : numpy.ndarray
            Mask patch with the same shape as "img".
        shape : Tuple[int]
            Shape of the output patches.

        Returns
        -------
        Tuple[numpy.ndarray]
            Resampled image (linear interpolation) and mask (nearest).
        """
        if random.random() > self.p:
            img = img_util.resize(img, shape)
            return img, img_util.resize_nearest(mask, shape)

        scale = np.exp(random.uniform(*self.log_range))
        theta = torch.zeros(1, 3, 4)
        theta[0, :, :3] = torch.from_numpy(random_rotation() / scale)
        size = (1, 1) + tuple(int(s) for s in shape)
        grid = F.affine_grid(theta, size, align_corners=False)
        return self.sample(img, grid, "bilinear"), self.sample(mask, grid, "nearest")

    @staticmethod
    def sample(patch, grid, mode):
        x = torch.from_numpy(np.ascontiguousarray(patch, dtype=np.float32))
        out = F.grid_sample(x[None, None], grid, mode=mode, align_corners=False)
        return out[0, 0].numpy()


def random_rotation():
    """
    Samples a 3D rotation matrix uniformly, from a random unit quaternion.

    Returns
    -------
    numpy.ndarray
        Rotation matrix with shape (3, 3).
    """
    q = np.array([random.gauss(0, 1) for _ in range(4)])
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


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
        patches[0] *= factor
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
        # The legacy global RandomState serializes the 32 loader threads on
        # one lock; a Generator per thread does not
        self._local = threading.local()

    def __call__(self, patches):
        """
        Adds uniform noise to the input 3D image.

        Parameters
        ----------
        patches : numpy.ndarray
            Image with the shape (2, H, W, D), where the first channel is the
            raw image and second is a mask.
        """
        rng = getattr(self._local, "rng", None)
        if rng is None:
            rng = self._local.rng = np.random.default_rng()
        std = self.max_std * random.random()
        noise = rng.random(patches[0].shape, dtype=np.float32)
        noise *= 2 * std
        noise -= std
        patches[0] += noise
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


# --- Raw Intensity Transforms ---
class RawIntensityTransforms:
    """
    Intensity augmentation applied to the raw image before the asinh
    normalization, in units of the patch's own noise level.

    The transforms above act on normalized values, which the asinh stretch
    compresses nonlinearly: a contrast factor there is a power law on the raw
    image, and additive noise there is far larger in raw units on bright
    voxels than on dim ones. On the raw image each transform keeps one
    meaning on every brain: a gain change is the same structure recorded
    brighter or dimmer, and noise is sensor noise.
    """

    def __init__(self, gain_range=(0.5, 2.0), max_noise=2.0, min_sigma=2.0):
        """
        Initializes a RawIntensityTransforms instance.

        Parameters
        ----------
        gain_range : Tuple[float], optional
            Range of the log-uniform gain factor. Default is (0.5, 2).
        max_noise : float, optional
            Maximum standard deviation of the added noise, in units of the
            patch noise. Default is 2.
        min_sigma : float, optional
            Lower bound on the patch noise in raw intensity units, so that a
            patch whose background is flat still gets noise. Default is 2.
        """
        self.gain = RandomGain(gain_range)
        self.noise = RandomRawNoise(max_noise, min_sigma)

    def __call__(self, img, bg, sigma):
        """
        Applies noise, then a gain change, to a raw image.

        Parameters
        ----------
        img : numpy.ndarray
            Raw float32 image, modified in place.
        bg : float
            Background level estimated before augmentation.
        sigma : float
            Noise standard deviation estimated before augmentation. The
            normalization uses these same values, so a gain change makes the
            patch look brighter instead of being normalized away.

        Returns
        -------
        numpy.ndarray
            Augmented raw image.
        """
        img = self.noise(img, bg, sigma)
        return self.gain(img, bg, sigma)


class RandomGain:
    """
    Scales the signal above background by a log-uniform factor, so that
    doubling and halving are equally likely.
    """

    def __init__(self, gain_range=(0.5, 2.0)):
        self.log_range = (np.log(gain_range[0]), np.log(gain_range[1]))

    def __call__(self, img, bg, sigma):
        gain = np.float32(np.exp(random.uniform(*self.log_range)))
        img -= np.float32(bg)
        img *= gain
        img += np.float32(bg)
        return img


class RandomRawNoise:
    """
    Adds Gaussian noise with a standard deviation of up to "max_std" times
    the noise level of the patch, which is taken to be at least "min_sigma".
    """

    def __init__(self, max_std=2.0, min_sigma=2.0):
        self.max_std = max_std
        self.min_sigma = min_sigma
        # See RandomNoise3D: a Generator per thread avoids the global lock
        self._local = threading.local()

    def __call__(self, img, bg, sigma):
        rng = getattr(self._local, "rng", None)
        if rng is None:
            rng = self._local.rng = np.random.default_rng()
        sigma = max(sigma, self.min_sigma)
        std = np.float32(self.max_std * sigma * random.random())
        noise = rng.standard_normal(img.shape, dtype=np.float32)
        noise *= std
        img += noise
        return img
