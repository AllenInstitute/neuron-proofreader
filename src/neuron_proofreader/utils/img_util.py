"""
Created on Fri May 8 11:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Helper routines for reading and processing images.

"""

from abc import ABC, abstractmethod
from fastremap import mask_except, renumber, unique
from matplotlib.colors import ListedColormap
from scipy.ndimage import zoom

import json
import matplotlib.pyplot as plt
import numpy as np
import tensorstore as ts

from neuron_proofreader.utils import util


class ImageReader(ABC):
    """
    Abstract class to create image readers classes.
    """

    def __init__(self, img_path):
        """
        Instantiates an ImageReader object.

        Parameters
        ----------
        img_path : str
            Path to image.
        is_segmentation : bool, optional
            Indication of whether image is a segmentation.
        """
        self.img_path = img_path
        self._load_image()

    @abstractmethod
    def _load_image(self):
        """
        This method should be implemented by subclasses to load the image
        based on img_path.
        """
        pass

    def read(self, center, shape):
        """
        Reads an image patch center at the given voxel coordinate.

        Parameters
        ----------
        center : Tuple[int]
            Center of image patch to be read.
        shape : Tuple[int]
            Shape of image patch to be read.

        Returns
        -------
        numpy.ndarray
            Image patch.
        """
        s = get_slices(center, shape)
        return self.img[s] if self.img.ndim == 3 else self.img[(0, 0, *s)]

    def read_voxel(self, voxel, thread_id=None):
        """
        Reads the intensity value at a given voxel.

        Parameters
        ----------
        voxel : Tuple[int]
            Voxel to be read.
        thread_id : Any
            Identifier associated with output. Default is None.

        Returns
        -------
        int
            Intensity value at voxel.
        """
        return thread_id, self.img[voxel]

    def shape(self):
        """
        Gets the shape of image.

        Returns
        -------
        Tuple[int]
            Shape of image.
        """
        return self.img.shape


class TensorStoreReader(ImageReader):
    """
    Class that reads an image with TensorStore library.
    """

    def __init__(self, img_path):
        """
        Instantiates a TensorStoreReader object.

        Parameters
        ----------
        img_path : str
            Path to image.
        """
        self.driver = self.get_driver(img_path)
        super().__init__(img_path)

    def get_driver(self, img_path):
        """
        Gets the storage driver needed to read the image.

        Parameters
        ----------
        img_path : str
            Path to image

        Returns
        -------
        str
            Storage driver needed to read the image.
        """
        if ".zarr" in img_path:
            return "zarr"
        elif ".n5" in img_path:
            return "n5"
        elif is_precomputed(img_path):
            return "neuroglancer_precomputed"
        else:
            raise ValueError(f"Unsupported image format: {img_path}")

    def _load_image(self):
        """
        Loads image using the TensorStore library.
        """
        # Extract metadata
        bucket_name, path = util.parse_cloud_path(self.img_path)
        storage_driver = get_storage_driver(self.img_path)

        # Load image
        self.img = ts.open(
            {
                "driver": self.driver,
                "kvstore": {
                    "driver": storage_driver,
                    "bucket": bucket_name,
                    "path": path,
                },
                "context": {
                    "cache_pool": {"total_bytes_limit": 1000000000},
                    "cache_pool#remote": {"total_bytes_limit": 1000000000},
                    "data_copy_concurrency": {"limit": 8},
                },
                "recheck_cached_data": "open",
            }
        ).result()

        # Check whether to absorb channel
        if bucket_name == "allen-nd-goog" and is_precomputed(self.img_path):
            self.img = self.img[ts.d["channel"][0]]
            self.img = self.img[ts.d[0].transpose[2]]
            self.img = self.img[ts.d[0].transpose[1]]

    def read(self, center, shape):
        """
        Reads an image patch center at the given voxel coordinate.

        Parameters
        ----------
        center : Tuple[int]
            Center of image patch to be read.
        shape : Tuple[int]
            Shape of image patch to be read.

        Returns
        -------
        numpy.ndarray
            Image patch.
        """
        try:
            return super().read(center, shape).read().result()
        except Exception:
            print(f"Unable to read image patch at {center} w/ shape {shape}!")
            return np.ones(shape)

    def read_voxel(self, voxel, thread_id):
        """
        Reads the intensity value at a given voxel.

        Parameters
        ----------
        voxel : Tuple[int]
            Voxel to be read.
        thread_id : Any
            Identifier associated with output.

        Returns
        -------
        int
            Intensity value at voxel.
        """
        return thread_id, int(self.img[voxel].read().result())


# --- Visualization ---
def make_segmentation_colormap(mask, seed=42):
    """
    Creates a matplotlib ListedColormap for a segmentation mask. Ensures label
    0 maps to black and all other labels get distinct random colors.

    Parameters
    ----------
    mask : numpy.ndarray
        Segmentation mask with integer labels. Assumes label 0 is background.
    seed : int, optional
        Random seed for color reproducibility. Default is 42.

    Returns
    -------
    ListedColormap
        Colormap with black for background and unique colors for other labels.
    """
    n_labels = int(mask.max()) + 1
    rng = np.random.default_rng(seed)
    colors = [(0, 0, 0)]
    colors += list(rng.uniform(0.2, 1.0, size=(n_labels - 1, 3)))
    return ListedColormap(colors)


def plot_image_and_segmentation_mips(img, segmentation, output_path=None):
    """
    Plots a 6x3 grid with image MIPs on the top and segmentation MIPs on the
    bottom.

    Parameters
    ----------
    img : numpy.ndarray
        Input 3D image to generate MIPs from.
    segmentation : numpy.ndarray
        Segmentation to generate MIPs from.
    output_path : None or str, optional
        Path to save MIPs as a PNG if provided. Default is None.
    """
    # Initializations
    vmax = np.percentile(img, 99.9)
    axes_names = ["XY", "XZ", "YZ"]
    cmap = make_segmentation_colormap(segmentation)

    fig, axs = plt.subplots(2, 3)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    # Image MIPs
    for i in range(3):
        mip = np.max(img, axis=i)
        ax = axs[0, i]
        ax.imshow(mip, vmax=vmax, aspect='equal')
        ax.set_title(axes_names[i], fontsize=16)
        ax.set_xticks([])
        ax.set_yticks([])

    # Segmentation MIPs
    for i in range(3):
        mip = np.max(segmentation, axis=i)
        ax = axs[1, i]
        ax.imshow(mip, cmap=cmap, interpolation="none", aspect='equal')
        ax.set_title(axes_names[i], fontsize=16)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()

    # Save figure if path provided
    if output_path:
        plt.savefig(output_path, dpi=200)

    plt.show()
    plt.close(fig)


def plot_mips(img, vmax=None):
    """
    Plots the Maximum Intensity Projections (MIPs) of a 3D image along the XY,
    XZ, and YZ axes.

    Parameters
    ----------
    img : numpy.ndarray
        Input 3D image to generate MIPs from.
    vmax : None or float
        Brightness intensity used as upper limit of the colormap. Default is
        None.
    """
    vmax = vmax or np.percentile(img, 99.9)
    fig, axs = plt.subplots(1, 3, figsize=(10, 4))
    axs_names = ["XY", "XZ", "YZ"]
    for i in range(3):
        mip = np.max(img, axis=i)
        axs[i].imshow(mip, vmax=vmax)
        axs[i].set_title(axs_names[i], fontsize=16)
        axs[i].set_xticks([])
        axs[i].set_yticks([])
    plt.tight_layout()
    plt.show()


def plot_segmentation_mips(segmentation):
    """
    Plots maximum intensity projections (MIPs) of a segmentation.

    Parameters
    ----------
    segmentation : numpy.ndarray
        Segmentation to generate MIPs from.
    """
    # Initialize plot
    fig, axs = plt.subplots(1, 3, figsize=(10, 4))
    axs_names = ["XY", "XZ", "YZ"]
    cmap = make_segmentation_colormap(segmentation)

    # Plot MIPs
    for i in range(3):
        mip = np.max(segmentation, axis=i)

        axs[i].imshow(mip, cmap=cmap, interpolation="none")
        axs[i].set_title(axs_names[i], fontsize=16)
        axs[i].set_xticks([])
        axs[i].set_yticks([])
    plt.tight_layout()


# --- Helpers ---
def annotate_voxels(img, voxels, kernel_size=3, val=1):
    """
    Annotates a set of voxel coordinates in a 3D image by filling a patch
    around each voxel with a given value.

    Parameters
    ----------
    img : numpy.ndarray
        Image to modify in-place.
    voxels : Iterable[Tuple[int]]
        Voxel coordinates to annotate.
    kernel_size : int, optional
        Size of kernel used to fill around each voxel. Default is 3.
    val : int, optional
        Value to write into each patch. Default is 1.
    """
    buffer = (kernel_size - 1) // 2
    shape = (kernel_size, kernel_size, kernel_size)
    for voxel in voxels:
        if is_contained(voxel, img.shape, buffer=buffer):
            s = get_slices(voxel, shape)
            img[s] = val


def compute_iou3d(c1, c2, s1, s2):
    """
    Computes IoU between two 3D axis-aligned boxes.

    Parameters
    ----------
    center1 : Tuple[int]
        3D center coordinate of box 1.
    center2 : Tuple[int]
        3D center coordinate of box 2.
    shape1 : Tuple[int]
        Shape of box for center 1.
    shape2 : Tuple[int]
        Shape of box for center 2.

    Returns
    -------
    float
        IoU between the boxes
    """
    c1, s1, c2, s2 = map(np.asarray, (c1, s1, c2, s2))
    min1, max1 = c1 - s1 / 2, c1 + s1 / 2
    min2, max2 = c2 - s2 / 2, c2 + s2 / 2

    overlap_min = np.maximum(min1, min2)
    overlap_max = np.minimum(max1, max2)
    overlap = np.maximum(overlap_max - overlap_min, 0)
    inter = np.prod(overlap)
    vol1 = np.prod(s1)
    vol2 = np.prod(s2)
    union = vol1 + vol2 - inter
    return inter / union if union > 0 else 0


def find_img_path(bucket_name, root_dir, brain_id):
    """
    Finds the path to a whole-brain dataset stored in a GCS bucket.

    Parameters:
    ----------
    bucket_name : str
        Name of the GCS bucket where the images are stored.
    root_dir : str
        Path to the directory in the GCS bucket where the image is expected to
        be located.
    dataset_name : str
        Name of the dataset to be searched for within the subdirectories.

    Returns:
    -------
    str
        Path of the found dataset subdirectory within the specified GCS bucket.
    """
    for subdir in util.list_gcs_subdirectories(bucket_name, root_dir):
        if brain_id in subdir:
            img_path = f"gs://{bucket_name}/{subdir}whole-brain/fused.zarr"
            return img_path
    raise f"Dataset not found in {bucket_name} - {root_dir}"


def get_contained_voxels(voxels, shape, buffer=0):
    """
    Gets voxels from the given list contained in an image specifed by "shape"
    and "buffer".

    Parameters
    ----------
    voxels : numpy.ndarray
        Array containing voxel coordinates.
    shape : Tuple[int]
        Shape of image patch.
    buffer : int, optional
        Constant value added/subtracted from the max/min coordinates of the
        bounding box. Default is 0.

    Returns
    -------
    List[Tuple[int]]
        Voxels from the given list contained in an image specifed by "shape"
        and "buffer".
    """
    return [v for v in voxels if is_contained(v, shape, buffer)]


def get_minimal_bbox(voxels, buffer=0):
    """
    Gets the min and max coordinates of a bounding box that contains "voxels".

    Parameters
    ----------
    voxels : numpy.ndarray
        Array containing voxel coordinates.
    buffer : int, optional
        Constant value added/subtracted from the max/min coordinates of the
        bounding box. Default is 0.

    Returns
    -------
    bbox : Dict[str, numpy.ndarray]
        Bounding box.
    """
    bbox = {
        "min": np.floor(np.min(voxels, axis=0) - buffer).astype(int),
        "max": np.ceil(np.max(voxels, axis=0) + buffer).astype(int),
    }
    return bbox


def get_neighbors(voxel, shape):
    """
    Gets the neighbors of a given voxel coordinate.

    Parameters
    ----------
    voxel : Tuple[int]
        Voxel coordinate in a 3D image.
    shape : Tuple[int]
        Shape of the 3D image that voxel is contained within.

    Returns
    -------
    neighbors : List[Tuple[int]]
         Voxel coordinates of the 26 neighbors of the given voxel.
    """
    # Initializations
    x, y, z = voxel
    depth, height, width = shape

    # Iterate over the possible offsets for x, y, and z
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                # Skip the (0, 0, 0) offset
                if dx == 0 and dy == 0 and dz == 0:
                    continue

                # Calculate the neighbor's coordinates
                nx, ny, nz = x + dx, y + dy, z + dz

                # Check if the neighbor is within the bounds of the 3D image
                if 0 <= nx < depth and 0 <= ny < height and 0 <= nz < width:
                    neighbors.append((nx, ny, nz))
    return neighbors


def get_offset(center, shape):
    """
    Computes the spatial offset of a crop given its center and shape.

    Parameters
    ----------
    center : Tuple[int]
        Center voxel coordinate of the crop.
    shape : Tuple[int]
        Shape of the crop.

    Returns
    -------
    Tuple[int]
        Offset of the crop.
    """
    return tuple([c - s // 2 for c, s in zip(center, shape)])


def get_slices(center, shape):
    """
    Gets the start and end indices of the chunk to be read.

    Parameters
    ----------
    center : Tuple[int]
        Center of image patch to be read.
    shape : Tuple[int]
        Shape of image patch to be read.

    Return
    ------
    Tuple[slice]
        Slice objects used to index into the image.
    """
    start = [int(c - d // 2) for c, d in zip(center, shape)]
    return tuple(slice(s, s + d) for s, d in zip(start, shape))


def get_storage_driver(img_path):
    """
    Gets the storage driver needed to read the image.

    Parameters
    ----------
    img_path : str
        Image path to be checked.

    Returns
    -------
    str
        Storage driver needed to read the image.
    """
    if util.is_s3_path(img_path):
        return "s3"
    elif util.is_gcs_path(img_path):
        return "gcs"
    else:
        raise ValueError(f"Unsupported path type: {img_path}")


def is_contained(voxel, shape, buffer=0):
    """
    Checks if the given voxel is within bounds of a given shape, considering a
    buffer.

    Parameters
    ----------
    voxel : Tuple[int]
        Voxel coordinates to be checked.
    shape : Tuple[int]
        Shape of image.
    buffer : int, optional
        Number of voxels to pad the bounds by when checking containment.
        Default 0.

    Returns
    -------
    bool
        True if the voxel is within bounds (with buffer) on all axes, False
        otherwise.
    """
    contained_above = all(0 <= v + buffer < s for v, s in zip(voxel, shape))
    contained_below = all(0 <= v - buffer < s for v, s in zip(voxel, shape))
    return contained_above and contained_below


def is_patch_contained(center, patch_shape, image_shape):
    """
    Checks if the given image patch defined by "center" and "patch_shape" is
    contained in the image defined by "image_shape".

    Parameters
    ----------
    voxel : Tuple[int]
        Voxel coordinates to be checked.
    patch_shape : Tuple[int]
        Shape of patch.
    image_shape : Tuple[int], optional
        Shape of image containing the patch.

    Returns
    -------
    bool
        True if the patch is contained in the image.
    """
    # Convert to arrays
    center = np.asarray(center)
    patch_shape = np.asarray(patch_shape)
    image_shape = np.asarray(image_shape)

    # Compute patch vertices
    half = patch_shape // 2
    start = center - half
    end = start + patch_shape
    return np.all(start >= 0) and np.all(end <= image_shape)


def is_precomputed(img_path):
    """
    Checks if the path points to a Neuroglancer precomputed dataset.

    Parameters
    ----------
    img_path : str
        Path to be checked (can be local, GCS, or S3).

    Returns
    -------
    bool
        True if the path appears to be a Neuroglancer precomputed dataset.
    """
    try:
        # Build kvstore spec
        bucket_name, path = util.parse_cloud_path(img_path)
        kv = {"driver": "gcs", "bucket": bucket_name, "path": path}

        # Open the info file
        store = ts.KvStore.open(kv).result()
        raw = store.read(b"info").result()

        # Only proceed if the key exists and has content
        if raw.state != "missing" and raw.value:
            info = json.loads(raw.value.decode("utf8"))
            is_valid_type = info.get("type") in ("image", "segmentation")
            if isinstance(info, dict) and is_valid_type and "scales" in info:
                return True
        return False
    except Exception:
        return False


def normalize(img, percentiles=(1, 99.5)):
    """
    Normalizes an image using a percentile-based scheme and clips values to
    [0, 1].

    Parameters
    ----------
    img : numpy.ndarray
        Image to be normalized.
    percentiles : Tuple[float], optional
        Upper and lower percentiles used to normalize the given image. Default
        is (1, 99.5).

    Returns
    -------
    img : numpy.ndarray
        Normalized image.
    """
    mn, mx = np.percentile(img, percentiles)
    return np.clip((img - mn) / (mx - mn + 1e-5), 0, 1)


def pad_to_shape(img, target_shape, pad_value=0):
    """
    Pads a NumPy image array to the specified target shape.

    Parameters
    ----------
    img : numpy.ndarray
        Input image with shape (D, H, W).
    target_shape : Tuple[int]
        Desired output shape
    pad_value : float, optional
        Value to use for padding. Default is 0.

    Returns
    -------
    numpy.ndarray
        Padded image with shape equal to target_shape.
    """
    pads = list()
    for s, t in zip(img.shape, target_shape):
        pads.append(((t - s) // 2, (t - s + 1) // 2))
    return np.pad(img, pads, mode='constant', constant_values=pad_value)


def remove_small_segments(segmentation, min_size):
    """
    Removes small segments from a segmentation.

    Parameters
    ----------
    segmentation : numpy.ndarray
        Integer array representing a segmentation mask. Each unique
        nonzero value corresponds to a distinct segment.
    min_size : int
        Minimum size (in voxels) for a segment to be kept.

    Returns
    -------
    segmentation : numpy.ndarray
        New segmentation of the same shape as the input, with only the
        retained segments renumbered contiguously.
    """
    ids, cnts = unique(segmentation, return_counts=True)
    ids = [i for i, cnt in zip(ids, cnts) if cnt > min_size and i != 0]
    ids = mask_except(segmentation, ids)
    segmentation, _ = renumber(ids, preserve_zero=True, in_place=True)
    return segmentation


def resize(img, new_shape, is_segmentation=False):
    """
    Resize a 3D image to the specified new shape using linear interpolation.

    Parameters
    ----------
    img : numpy.ndarray
        Input 3D image array with shape (depth, height, width).
    new_shape : Tuple[int]
        Desired output shape as (new_depth, new_height, new_width).
    is_segmentation : bool, optional
        Indication of whether the image represents a segmentation mask.

    Returns
    -------
    numpy.ndarray
        Resized 3D image with shape equal to "new_shape".
    """
    # Set parameters
    order = 0 if is_segmentation else 3
    multiplier = 4 if is_segmentation else 1
    zoom_factors = np.array(new_shape) / np.array(img.shape)

    # Resize image
    img = zoom(multiplier * img, zoom_factors, order=order)
    return img / multiplier


def to_physical(voxel, anisotropy, offset=(0, 0, 0)):
    """
    Converts a voxel coordinate to a physical coordinate by applying the
    anisotropy scaling factors.

    Parameters
    ----------
    voxel : ArrayLike
        Voxel coordinate to be converted.
    anisotropy : ArrayLike
        Image to physical coordinates scaling factors to account for the
        anisotropy of the microscope.
    offset : Tuple[int], optional
        Shift to be applied to "voxel". Default is (0, 0, 0).

    Returns
    -------
    Tuple[float]
        Physical coordinate.
    """
    voxel = voxel[::-1]
    return tuple([voxel[i] * anisotropy[i] - offset[i] for i in range(3)])


def to_voxels(xyz, anisotropy, multiscale=0):
    """
    Converts coordinate from a physical to voxel space.

    Parameters
    ----------
    xyz : ArrayLike
        Physical coordiante to be converted.
    anisotropy : ArrayLike
        Image to physical coordinates scaling factors to account for the
        anisotropy of the microscope.
    multiscale : int, optional
        Level in the image pyramid that the voxel coordinate must index into.
        Default is 0.

    Returns
    -------
    Tuple[int]
        Voxel coordinate.
    """
    scaling_factor = 1.0 / 2 ** multiscale
    voxel = [int(scaling_factor * xyz[i] / anisotropy[i]) for i in range(3)]
    return tuple(voxel[::-1])
