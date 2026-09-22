"""
Created on Fri May 8 11:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Helper routines for reading and processing images.

"""

from matplotlib.colors import ListedColormap

import json
import matplotlib.pyplot as plt
import numpy as np
import tensorstore as ts
import torch
import torch.nn.functional as F

from neuron_proofreader.utils import util


# --- Visualization ---
def make_segmentation_colormap(mask, seed=42):
    """
    Creates a matplotlib ListedColormap for a segmentation. Ensures label 0
    maps to black and all other labels get distinct random colors.

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
        ax.imshow(mip, vmax=vmax, aspect="equal")
        ax.set_title(axes_names[i], fontsize=16)
        ax.set_xticks([])
        ax.set_yticks([])

    # Segmentation MIPs
    for i in range(3):
        mip = np.max(segmentation, axis=i)
        ax = axs[1, i]
        ax.imshow(mip, cmap=cmap, interpolation="none", aspect="equal")
        ax.set_title(axes_names[i], fontsize=16)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()

    # Save figure if path provided
    if output_path:
        plt.savefig(output_path, dpi=200)

    plt.show()
    plt.close(fig)


def plot_mips(img, vmax=None, output_path=None):
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
    # Generate plot
    vmax = vmax or np.percentile(img, 99.9)
    fig, axs = plt.subplots(1, 3, figsize=(10, 4))
    axs_names = ["XY", "XZ", "YZ"]
    for i in range(3):
        mip = np.max(img, axis=i)
        axs[i].imshow(mip, vmax=vmax)
        axs[i].set_title(axs_names[i], fontsize=16)
        axs[i].set_xticks([])
        axs[i].set_yticks([])

    # Display result
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    else:
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
def annotate_voxels(img, voxels, kernel_size=3, fill_val=1):
    """
    Annotates voxel coordinates in a 3D image by filling a patch around each
    voxel with a given value.

    Parameters
    ----------
    img : numpy.ndarray
        Image to modify in-place.
    voxels : Iterable[Tuple[int]]
        Voxel coordinates to annotate.
    kernel_size : int, optional
        Size of kernel used to fill around each voxel. Default is 3.
    fill_val : int, optional
        Fill value. Default is 1.
    """
    voxels = np.asarray(voxels, dtype=np.int64).reshape(-1, 3)
    if len(voxels) == 0:
        return

    # Keep voxels whose full kernel lies inside the image (same test as
    # is_contained(voxel, img.shape, buffer))
    buffer = (kernel_size - 1) // 2
    shape = np.asarray(img.shape)
    keep = np.all(voxels - buffer >= 0, axis=1)
    keep &= np.all(voxels + buffer < shape, axis=1)
    voxels = voxels[keep]
    if len(voxels) == 0:
        return

    # Fill kernel around every voxel in one vectorized assignment. The kernel
    # spans [c - k//2, c - k//2 + k), matching get_slices.
    offsets = np.arange(kernel_size) - kernel_size // 2
    grid = np.stack(np.meshgrid(offsets, offsets, offsets, indexing="ij"), -1)
    idx = (voxels[:, None, :] + grid.reshape(1, -1, 3)).reshape(-1, 3)
    img[idx[:, 0], idx[:, 1], idx[:, 2]] = fill_val


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
    # Extract shapes
    c1, s1, c2, s2 = map(np.asarray, (c1, s1, c2, s2))
    min1, max1 = c1 - s1 / 2, c1 + s1 / 2
    min2, max2 = c2 - s2 / 2, c2 + s2 / 2

    # Compute overlap
    overlap_min = np.maximum(min1, min2)
    overlap_max = np.minimum(max1, max2)
    overlap = np.maximum(overlap_max - overlap_min, 0)
    union = np.prod(s1) + np.prod(s2) - np.prod(overlap)
    return np.prod(overlap) / union if union > 0 else 0


def find_img_path(root_prefix, brain_id):
    """
    Finds the path to a whole-brain dataset stored in a GCS bucket.

    Parameters:
    ----------
    root_prefrix : str
        Path to the directory in the GCS bucket where the image is expected to
        be located.
    dataset_name : str
        Name of the dataset to be searched for within the subdirectories.

    Returns:
    -------
    str
        Path of the found dataset subdirectory within the specified GCS bucket.
    """
    bucket_name, _ = util.parse_cloud_path(root_prefix)
    for prefix in util.list_gcs_subprefixes(root_prefix):
        if brain_id in prefix:
            img_path = f"gs://{bucket_name}/{prefix}whole-brain/fused.zarr"
            return img_path
    raise f"Dataset not found in {root_prefix}"


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


def get_driver(img_path):
    """
    Gets the driver needed to read the image.

    Parameters
    ----------
    img_path : str
        Path to image.

    Returns
    -------
    str
        Storage driver needed to read the image.
    """
    if ".zarr" in img_path:
        return "zarr"
    elif is_precomputed(img_path):
        return "neuroglancer_precomputed"
    raise Exception(f"Invalid image path at {img_path}")


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


def get_slices(center, shape, img_shape=None):
    """
    Gets the start and end indices of the chunk to be read, clamped to the
    image bounds.

    Parameters
    ----------
    center : Tuple[int]
        Center of image patch to be read.
    shape : Tuple[int]
        Shape of image patch to be read.
    img_shape : Tuple[int], optional
        Spatial shape of the image. If provided, the end of each slice is
        clamped to it. Default is None.

    Return
    ------
    Tuple[slice]
        Slice objects used to index into the image.
    """
    start = [max(0, int(c - d // 2)) for c, d in zip(center, shape)]
    end = [s + d for s, d in zip(start, shape)]
    if img_shape is not None:
        end = [min(e, int(n)) for e, n in zip(end, img_shape)]
    return tuple(slice(s, e) for s, e in zip(start, end))


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
        Normalized image as float32. The arithmetic is done in float32 (the
        precision every consumer casts to anyway), which halves the memory
        traffic of the float64 version.
    """
    mn, mx = np.percentile(img, percentiles)
    out = img.astype(np.float32)
    out -= np.float32(mn)
    out /= np.float32(mx - mn + 1e-5)
    return np.clip(out, 0, 1, out=out)


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
    return np.pad(img, pads, mode="constant", constant_values=pad_value)


def resize(img, new_shape):
    """
    Resizes a 3D image to the new shape using linear interpolation.

    Parameters
    ----------
    img : numpy.ndarray
        Input 3D image array with shape (depth, height, width).
    new_shape : Tuple[int]
        Desired output shape as (new_depth, new_height, new_width).

    Returns
    -------
    numpy.ndarray
        Resized 3D image.

    Notes
    -----
    Reproduces scipy.ndimage.zoom(img, factors, order=1, prefilter=False),
    which the split model was trained with, using torch's trilinear
    interpolation (20-30x faster on the CPU). Both map output index i to
    input coordinate i * (in - 1) / (out - 1). One scipy quirk is kept on
    purpose: when that product rounds to slightly more than in - 1 for the
    last output index, scipy treats the coordinate as out of bounds and
    zero-fills the entire last plane along that axis.
    """
    new_shape = tuple(int(s) for s in new_shape)
    if tuple(img.shape) == new_shape:
        return np.array(img, dtype=np.float32, copy=True)

    x = torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32))
    out = F.interpolate(
        x[None, None], size=new_shape, mode="trilinear", align_corners=True
    )[0, 0].numpy()

    # scipy border quirk (see Notes)
    for axis, (n_in, n_out) in enumerate(zip(img.shape, new_shape)):
        if n_out > 1:
            zoom_factor = np.float64(n_in - 1) / np.float64(n_out - 1)
            if np.float64(n_out - 1) * zoom_factor > n_in - 1:
                idx = [slice(None)] * 3
                idx[axis] = -1
                out[tuple(idx)] = 0
    return out


def resize_nearest(mask, new_shape):
    """
    Resizes a 3D label mask with nearest-neighbor sampling, reproducing
    skimage.transform.resize(mask, shape, order=0, preserve_range=True,
    anti_aliasing=False), which samples at grid centers
    (i + 0.5) * in / out - 0.5.

    Parameters
    ----------
    mask : numpy.ndarray
        Mask to be resized.
    new_shape : Tuple[int]
        Desired output shape.

    Returns
    -------
    numpy.ndarray
        Resized mask with the same dtype as the input.
    """
    new_shape = tuple(int(s) for s in new_shape)
    if tuple(mask.shape) == new_shape:
        return np.array(mask, copy=True)
    x = torch.from_numpy(np.ascontiguousarray(mask, dtype=np.float32))
    out = F.interpolate(x[None, None], size=new_shape, mode="nearest-exact")
    return out[0, 0].numpy().astype(mask.dtype)


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


def to_voxels(xyz, anisotropy):
    """
    Converts coordinate from a physical to voxel space.

    Parameters
    ----------
    xyz : ArrayLike
        Physical coordinate to be converted.
    anisotropy : ArrayLike
        Image to physical coordinates scaling factors to account for the
        anisotropy of the microscope.

    Returns
    -------
    Tuple[int]
        Voxel coordinate.
    """
    voxel = [int(xyz[i] / anisotropy[i]) for i in range(3)]
    return tuple(voxel[::-1])
