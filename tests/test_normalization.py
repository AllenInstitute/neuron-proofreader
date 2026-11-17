"""
Tests for the asinh patch normalization and the raw-image intensity
augmentation that goes with it.
"""

import numpy as np
import pytest

from neuron_proofreader.configs import ImageConfig
from neuron_proofreader.machine_learning.image_augmentation import (
    ImageTransforms,
    RawIntensityTransforms,
)
from neuron_proofreader.utils import img_util


def sparse_patch(bg=20.0, sigma=4.0, peak=600.0, shape=(32, 32, 32), seed=0):
    """
    Gaussian background noise with one bright tube through the center, like
    a neurite in a sparse light microscopy volume.
    """
    rng = np.random.default_rng(seed)
    img = rng.normal(bg, sigma, shape)
    z, y = np.mgrid[: shape[0], : shape[1]]
    tube = np.exp(-((z - shape[0] / 2) ** 2 + (y - shape[1] / 2) ** 2) / 4.5)
    img += peak * tube[:, :, None]
    return np.clip(np.round(img), 0, 65535).astype(np.uint16)


def test_background_stats_ignore_foreground():
    patch = sparse_patch(bg=20, sigma=4, peak=600)
    mode, sigma = img_util.background_stats(patch)
    assert mode == pytest.approx(20, abs=1)
    assert sigma == pytest.approx(4, rel=0.25)

    # Float path agrees with the integer path
    mode_f, sigma_f = img_util.background_stats(patch.astype(np.float32))
    assert mode_f == mode
    assert sigma_f == pytest.approx(sigma, abs=0.8)


def test_asinh_lut_matches_float_path():
    patch = sparse_patch()
    lut = img_util.normalize_asinh_lut(patch)
    direct = img_util.normalize_asinh(patch)
    assert lut.dtype == np.float32
    np.testing.assert_allclose(lut, direct, atol=1e-6)


def test_asinh_anchors():
    patch = sparse_patch(peak=600)
    mode, sigma = img_util.background_stats(patch)
    out = img_util.normalize_asinh(patch, scale=100.0, stats=(mode, sigma))

    # Background maps to 0 and the stretch is asinh((x - mode) / scale)
    assert np.median(out) == pytest.approx(0, abs=0.05)
    anchors = np.array([mode, mode + 100, mode + 600], dtype=np.float32)
    np.testing.assert_allclose(
        img_util.normalize_asinh(anchors, scale=100.0, stats=(mode, sigma)),
        [0, np.arcsinh(1), np.arcsinh(6)],
        atol=1e-6,
    )

    # The peak is compressed, not clipped: monotonic over the whole range
    ramp = np.arange(0, 5000, dtype=np.float32)
    assert np.all(np.diff(img_util.normalize_asinh(ramp, stats=(mode, sigma))) > 0)


def test_asinh_is_invariant_to_camera_offset():
    patch = sparse_patch().astype(np.float32)
    np.testing.assert_allclose(
        img_util.normalize_asinh(patch + 37),
        img_util.normalize_asinh(patch),
        atol=1e-6,
    )


def test_asinh_flat_patch_maps_to_zero():
    flat = np.full((16, 16, 16), 7, dtype=np.uint16)
    assert np.all(img_util.normalize_asinh_lut(flat) == 0)


def test_normalize_lut_still_matches_normalize():
    patch = sparse_patch()
    np.testing.assert_allclose(
        img_util.normalize_lut(patch, 400),
        img_util.normalize(np.minimum(patch, 400)),
        atol=1e-6,
    )


def test_raw_transforms_keep_shape_dtype_and_background():
    patch = sparse_patch().astype(np.float32)
    mode, sigma = img_util.background_stats(patch)
    for _ in range(10):
        out = RawIntensityTransforms()(patch.copy(), mode, sigma)
        assert out.shape == patch.shape and out.dtype == np.float32
        # Gain scales about the background, noise is zero mean and
        # smoothing preserves the mean, so the background stays put
        assert np.median(out) == pytest.approx(mode, abs=sigma)


def test_geometric_only_transform_does_not_clip():
    patches = np.stack(
        [np.full((8, 8, 8), 2.0, np.float32), np.zeros((8, 8, 8), np.float32)]
    )
    out = ImageTransforms(intensity=False)(patches)
    assert out.shape == patches.shape
    assert out[0].min() == 2.0 and out[0].max() == 2.0


def test_image_config_rejects_unknown_normalization():
    assert ImageConfig().normalization == "percentile"
    assert ImageConfig(normalization="asinh").normalization == "asinh"
    with pytest.raises(ValueError):
        ImageConfig(normalization="arcsinh")
