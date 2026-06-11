"""
Created on Wed June 11 12:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code for applying data augmentation to 3D space curves.

"""

import numpy as np
import random


class CurveTransforms:
    """
    Class that applies a sequence of transforms to a 3D space curve.
    """
    def __init__(self):
        """
        Initializes a CurveTransforms instance that applies augmentation to
        a 3D space curve.
        """
        self.transforms = [
            RandomRotation3D(),  RandomMirror3D(), RandomJitter3D(),
        ]

    def __call__(self, curve):
        """
        Applies transforms to the input curve.

        Parameters
        ----------
        curve : numpy.ndarray
            Array of shape (N, 3) representing N points in 3D space (x, y, z).
        """
        # Check whether to reverse path
        if random.random() > 0.5:
            curve = np.flip(curve)

        # Apply transforms
        for transform in self.transforms:
            curve = transform(curve)
        return curve


# --- Noise Transforms ---
class RandomJitter3D:
    """
    Randomly adds Gaussian noise to each point in a 3D curve.
    """
    def __init__(self, sigma=1, p=0.5):
        """
        Initializes a RandomJitter3D transformer.

        Parameters
        ----------
        sigma : float, optional
            Standard deviation of the Gaussian noise. Default is 0.01.
        p : float, optional
            Probability of applying the transform. Default is 0.5.
        """
        self.sigma = sigma
        self.p = p

    def __call__(self, curve):
        """
        Applies random jitter to the input curve.

        Parameters
        ----------
        curve : numpy.ndarray
            Array of shape (N, 3) representing N points in 3D space.

        Returns
        -------
        numpy.ndarray
            Jittered curve of shape (N, 3).
        """
        if random.random() > self.p:
            return curve
        noise = np.random.normal(0, self.sigma, size=curve.shape)
        return curve + noise


# --- Geometric Transforms ---
class RandomRotation3D:
    """
    Applies a random 3D rotation to a curve about a random axis.
    """
    def __init__(self, max_angle=np.pi, p=0.5):
        """
        Initializes a RandomRotation3D transformer.

        Parameters
        ----------
        max_angle : float, optional
            Maximum rotation angle in radians. Default is pi (full rotation).
        p : float, optional
            Probability of applying the transform. Default is 0.5.
        """
        self.max_angle = max_angle
        self.p = p

    def _rotation_matrix(self, axis, angle):
        """
        Computes the Rodrigues rotation matrix for a given axis and angle.

        Parameters
        ----------
        axis : numpy.ndarray
            Unit vector of shape (3,) representing the rotation axis.
        angle : float
            Rotation angle in radians.

        Returns
        -------
        numpy.ndarray
            Rotation matrix of shape (3, 3).
        """
        c, s = np.cos(angle), np.sin(angle)
        x, y, z = axis
        return np.array([
            [c + x*x*(1-c),   x*y*(1-c) - z*s, x*z*(1-c) + y*s],
            [y*x*(1-c) + z*s, c + y*y*(1-c),   y*z*(1-c) - x*s],
            [z*x*(1-c) - y*s, z*y*(1-c) + x*s, c + z*z*(1-c)],
        ])

    def __call__(self, curve):
        """
        Applies a random rotation to the input curve about its centroid.

        Parameters
        ----------
        curve : numpy.ndarray
            Array of shape (N, 3) representing N points in 3D space.

        Returns
        -------
        numpy.ndarray
            Rotated curve of shape (N, 3).
        """
        if random.random() > self.p:
            return curve
        axis = np.random.randn(3)
        axis /= np.linalg.norm(axis)
        angle = random.uniform(-self.max_angle, self.max_angle)
        R = self._rotation_matrix(axis, angle)
        centroid = curve.mean(axis=0)
        return (curve - centroid) @ R.T + centroid


class RandomMirror3D:
    """
    Randomly mirrors a 3D curve along one or more axes about its centroid.
    """
    def __init__(self, axes=(0, 1, 2), p=0.5):
        """
        Initializes a RandomMirror3D transformer.

        Parameters
        ----------
        axes : Tuple[int], optional
            Axes to consider for mirroring. Default is (0, 1, 2).
        p : float, optional
            Per-axis probability of mirroring. Default is 0.5.
        """
        self.axes = axes
        self.p = p

    def __call__(self, curve):
        """
        Applies random mirroring to the input curve.

        Parameters
        ----------
        curve : numpy.ndarray
            Array of shape (N, 3) representing N points in 3D space.

        Returns
        -------
        numpy.ndarray
            Mirrored curve of shape (N, 3).
        """
        curve = curve.copy()
        centroid = curve.mean(axis=0)
        for axis in self.axes:
            if random.random() > self.p:
                continue
            curve[:, axis] = 2 * centroid[axis] - curve[:, axis]
        return curve
