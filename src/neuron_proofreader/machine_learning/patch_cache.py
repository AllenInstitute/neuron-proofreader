"""
Created on Sun Nov 1 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

On-disk cache of raw image patches for merge detection training.

Reading a patch from cloud storage dominates the cost of a training example,
while the set of sites is fixed across epochs once the nonmerge pool is
frozen. The cache stores the raw (unnormalized) uint16 window around each
site with a margin, so the training-time voxel shift can be cropped out of
the cached window, plus a per-brain manifest of the sites so later runs
rebuild the same pool.

Layout
------
<cache_dir>/<brain_id>/sites.json
<cache_dir>/<brain_id>/<z>_<y>_<x>_<d>x<h>x<w>.npy
"""

import json
import os
import threading

import numpy as np


class PatchCache:
    """
    Cache of raw image windows for a single brain.

    Parameters
    ----------
    cache_dir : str
        Root directory shared by all brains.
    brain_id : str
        Brain whose windows are stored in <cache_dir>/<brain_id>.
    margin : int
        Voxels added on every side of the requested patch shape, so a
        shifted patch can be cropped out of the stored window.
    """

    def __init__(self, cache_dir, brain_id, margin):
        self.brain_dir = os.path.join(cache_dir, brain_id)
        self.margin = margin
        self.sites_path = os.path.join(self.brain_dir, "sites.json")
        try:
            os.makedirs(self.brain_dir, exist_ok=True)
            self.read_only = not os.access(self.brain_dir, os.W_OK)
        except OSError:
            self.read_only = True

    # --- Patches ---
    def window_shape(self, shape):
        return tuple(int(s) + 2 * self.margin for s in shape)

    def patch_path(self, voxel, shape):
        name = "_".join(str(int(v)) for v in voxel)
        window = "x".join(str(s) for s in self.window_shape(shape))
        return os.path.join(self.brain_dir, f"{name}_{window}.npy")

    def contains(self, voxel, shape):
        return os.path.exists(self.patch_path(voxel, shape))

    def get(self, voxel, shape):
        """
        Returns the stored window around "voxel", or None on a miss.
        """
        path = self.patch_path(voxel, shape)
        return np.load(path) if os.path.exists(path) else None

    def put(self, voxel, shape, window):
        if self.read_only:
            return
        path = self.patch_path(voxel, shape)
        tmp_path = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
        with open(tmp_path, "wb") as f:
            np.save(f, window)
        os.replace(tmp_path, path)

    # --- Site Manifest ---
    def has_sites(self):
        return os.path.exists(self.sites_path)

    def load_sites(self):
        """
        Returns
        -------
        Dict[str, List[dict]]
            "merge_sites" and "nonmerge_sites", each a list of records with
            "xyz" and "filename".
        """
        with open(self.sites_path) as f:
            return json.load(f)

    def save_sites(self, merge_sites, nonmerge_sites):
        """
        Parameters
        ----------
        merge_sites, nonmerge_sites : pandas.DataFrame
            Site tables with "xyz" and "filename" columns.
        """
        if self.read_only:
            raise RuntimeError(
                f"Patch cache {self.brain_dir} has no sites.json and is "
                "read-only; build the cache before using it."
            )
        sites = {
            "merge_sites": self._records(merge_sites),
            "nonmerge_sites": self._records(nonmerge_sites),
        }
        with open(self.sites_path, "w") as f:
            json.dump(sites, f)

    @staticmethod
    def _records(sites):
        if len(sites) == 0:
            return []
        return [
            {"xyz": [float(v) for v in xyz], "filename": filename}
            for xyz, filename in zip(sites["xyz"], sites["filename"])
        ]
