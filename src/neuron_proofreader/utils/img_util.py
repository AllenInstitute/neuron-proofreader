"""
Created on Fri May 8 11:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Helper routines for reading and processing images.

"""

from abc import ABC, abstractmethod
from cloudvolume import CloudVolume
from scipy.ndimage import zoom

import numpy as np
import matplotlib.pyplot as plt
import s3fs
import tensorstore as ts
import zarr

from neuron_proofreader.utils import util


class ImageReader(ABC):
    """
    Abstract class to create image readers classes.
    """

    def __init__(self, img_path):
        """
        Instantiates ImageReader object.

        Parameters
        ----------
        img_path : str
            Path to image.
        is_segmentation : bool, optional
            Indication of whether image is a segmentation.
        """
        self.img = None
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
        Instantiates TensorStoreReader object.

        Parameters
        ----------
        img_path : str
            Path to image.
        """
        self.driver = self.init_driver(img_path)
        super().__init__(img_path)

    def init_driver(self, img_path):
        """
        Gets the storage driver needed to read the image.

        Returns
        -------
        str
            Storage driver needed to read the image.
        """
        if ".zarr" in img_path:
            return "zarr"
        elif ".n5" in img_path:
            return "n5"
        elif is_neuroglancer_precomputed(img_path):
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

        # Check whether to permute axes
        if bucket_name == "allen-nd-goog":
            self.img = self.img[ts.d["channel"][0]]
            #self.img = self.img[ts.d[0].transpose[2]]
            #self.img = self.img[ts.d[0].transpose[1]]

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


class ZarrReader(ImageReader):
    """
    Class that reads image with Zarr library.
    """

    def __init__(self, img_path):
        """
        Instantiates ZarrReader object.

        Parameters
        ----------
        img_path : str
            Path to image.
        """
        super().__init__(img_path)

    def _load_image(self):
        """
        Loads image using the Zarr library.
        """
        store = s3fs.S3Map(root=self.img_path, s3=s3fs.S3FileSystem(anon=True))
        self.img = zarr.open(store, mode="r")
        assert self.img.ndim in (3, 5), f"Invalid Img Shape {self.img.shape}!"


def init_reader(img_path):
    """
    Initializes an image reader based on where image is stored in cloud.

    Parameters
    ----------
    img_path : str
        Path to image.

    Returns
    -------
    ImageReader
        Image reader.
    """
    if util.is_s3_path(img_path):
        return ZarrReader(img_path)
    else:
        return TensorStoreReader(img_path)


# --- Helpers ---
def fill_path(img, path, val=-1):
    """
    Fills a given path in a 3D image array with a specified value.

    Parameters
    ----------
    img : numpy.ndarray
        The 3D image array to fill the path in.
    path : iterable
        A list or iterable containing 3D coordinates (x, y, z) representing
        the path.
    val : int, optional
        The value to fill the path with. Default is -1.

    Returns
    -------
    numpy.ndarray
        The modified image array with the path filled with the specified value.
    """
    for xyz in path:
        x, y, z = tuple(np.floor(xyz).astype(int))
        img[x - 2: x + 3, y - 2: y + 3, z - 2: z + 3] = val
    return img


def find_img_path(bucket_name, root_dir, dataset_name):
    """
    Finds the path to an image in a GCS bucket for the dataset given by
    "dataset_name".

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
        if dataset_name in subdir:
            return subdir + "whole-brain/fused.zarr/"
    raise f"Dataset not found in {bucket_name} - {root_dir}"


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
    Dict[str, numpy.ndarray]
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
    List[Tuple[int]]
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
    start = [c - d // 2 for c, d in zip(center, shape)]
    return tuple(slice(s, s + d) for s, d in zip(start, shape))


def is_contained(voxel, shape, buffer=0):
    """
    Check whether a voxel is within bounds of a given shape, considering a
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


def is_neuroglancer_precomputed(path):
    """
    Checks if the path points to a neuroglancer precomputed dataset.

    Parameters
    ----------
    path : str
        Path to be checked.

    Returns
    -------
    bool
        Indication of whether the path points to a neuroglancer precomputed
        dataset.
    """
    try:
        vol = CloudVolume(path)
        return all(k in vol.info for k in ["data_type", "scales", "type"])
    except Exception:
        return False


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


def normalize(img):
    """
    Normalizes an image so that the minimum and maximum intensity values are 0
    and 1.

    Parameters
    ----------
    img : numpy.ndarray
        Image to be normalized.

    Returns
    -------
    numpy.ndarray
        Normalized image.
    """
    try:
        mn, mx = np.percentile(img, [1, 99.9])
        return np.clip((img - mn) / max((mx - mn), 1), 0, 1)
    except Exception as e:
        print("Image Normalization Failed:", e)
        return np.zeros(img.shape)


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


def plot_mips(img, vmax=None):
    """
    Plots the Maximum Intensity Projections (MIPs) of a 3D image along the XY,
    XZ, and YZ axes.

    Parameters
    ----------
    img : numpy.ndarray
        Input 3D image to generate MIPs from.
    vmax : None or float
        Brightness intensity used as upper limit of the colormap. Default is None.
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


def resize(img, new_shape):
    """
    Resize a 3D image to the specified new shape using linear interpolation.

    Parameters
    ----------
    img : numpy.ndarray
        Input 3D image array with shape (depth, height, width).
    new_shape : Tuple[int]
        Desired output shape as (new_depth, new_height, new_width).

    Returns
    -------
    numpy.ndarray
        Resized 3D image with shape equal to "new_shape".
    """
    depth, height, width = img.shape
    zoom_factors = np.array(new_shape) / np.array([depth, height, width])
    resized_img = zoom(img, zoom_factors, order=1)
    return resized_img


def to_physical(voxel, anisotropy, shift=(0, 0, 0)):
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
    shift : Tuple[int], optional
        Shift to be applied to "voxel". Default is (0, 0, 0).

    Returns
    -------
    Tuple[float]
        Physical coordinate.
    """
    voxel = voxel[::-1]
    return tuple([voxel[i] * anisotropy[i] - shift[i] for i in range(3)])


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
