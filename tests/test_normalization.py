"""
Tests for the asinh patch normalization and the intensity augmentation
that goes with it.
"""

import numpy as np
import pytest

from neuron_proofreader.configs import ImageConfig
from neuron_proofreader.machine_learning.image_augmentation import (
    ImageTransforms,
    RandomAffine3D,
    random_rotation,
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


def tube_mask(shape=(32, 32, 32), width=5):
    """Mask of a tube of "width" voxels around the neurite of sparse_patch."""
    mask = np.zeros(shape, dtype=np.float32)
    lo = [n // 2 - width // 2 for n in shape[:2]]
    mask[lo[0]: lo[0] + width, lo[1]: lo[1] + width, :] = 1
    return mask


def asinh_stats(patch, quantile=0.7):
    bg, sigma = img_util.background_stats(patch)
    scale = img_util.foreground_scale(patch, tube_mask(), bg, sigma, quantile)
    return bg, scale


def test_background_stats_ignore_zero_padding():
    patch = sparse_patch(bg=20, sigma=4)
    patch[:20] = 0
    assert img_util.background_stats(patch)[0] == pytest.approx(20, abs=1)
    assert img_util.background_stats(np.zeros((4, 4, 4), np.uint16)) == (0, 0)


def test_foreground_scale_follows_the_masked_neurite():
    bg, scale = asinh_stats(sparse_patch(peak=600))
    assert 100 < scale < 600

    # A higher quantile sits closer to the core of the fiber
    assert asinh_stats(sparse_patch(peak=600), 0.9)[1] > scale

    # A fragment on background is floored by the noise, not stretched
    noise = sparse_patch(peak=0)
    bg, sigma = img_util.background_stats(noise)
    assert img_util.foreground_scale(
        noise, tube_mask(), bg, sigma
    ) == pytest.approx(5 * sigma)
    empty = np.zeros_like(tube_mask())
    assert img_util.foreground_scale(noise, empty, bg, 0.0) == 1.0


def test_asinh_lut_matches_float_path():
    patch = sparse_patch()
    bg, scale = asinh_stats(patch)
    lut = img_util.normalize_asinh_lut(patch, bg, scale)
    direct = img_util.normalize_asinh(patch, bg, scale)
    assert lut.dtype == np.float32
    np.testing.assert_allclose(lut, direct, atol=1e-6)


def test_asinh_anchors():
    patch = sparse_patch(peak=600)
    bg, scale = asinh_stats(patch)
    out = img_util.normalize_asinh(patch, bg, scale)

    # Background maps to 0 and the stretch is asinh((x - bg) / scale)
    assert np.median(out) == pytest.approx(0, abs=0.05)
    anchors = np.array([bg, bg + scale, bg + 6 * scale], dtype=np.float32)
    np.testing.assert_allclose(
        img_util.normalize_asinh(anchors, bg, scale),
        [0, np.arcsinh(1), np.arcsinh(6)],
        atol=1e-5,
    )

    # The peak is compressed, not clipped: monotonic over the whole range
    ramp = np.arange(0, 5000, dtype=np.float32)
    assert np.all(np.diff(img_util.normalize_asinh(ramp, bg, scale)) > 0)


def test_asinh_is_invariant_to_brightness_and_camera_offset():
    dim = sparse_patch(bg=20, sigma=0, peak=300)
    bright = 3 * dim + 37
    np.testing.assert_allclose(
        img_util.normalize_asinh(bright, *asinh_stats(bright)),
        img_util.normalize_asinh(dim, *asinh_stats(dim)),
        atol=1e-5,
    )


def test_normalize_lut_still_matches_normalize():
    patch = sparse_patch()
    np.testing.assert_allclose(
        img_util.normalize_lut(patch, 400),
        img_util.normalize(np.minimum(patch, 400)),
        atol=1e-6,
    )


def test_raw_noise_keeps_shape_dtype_and_background():
    patch = sparse_patch().astype(np.float32)
    mode, sigma = img_util.background_stats(patch)
    for _ in range(10):
        out = ImageTransforms("asinh").raw_noise(patch.copy(), sigma)
        assert out.shape == patch.shape and out.dtype == np.float32
        assert np.median(out) == pytest.approx(mode, abs=sigma)


def test_fg_quantile_is_sampled_on_both_sides():
    transform = ImageTransforms("asinh")
    draws = [transform.sample_fg_quantile(0.7) for _ in range(200)]
    assert min(draws) >= 0.5 and max(draws) <= 0.9
    assert min(draws) < 0.6 and max(draws) > 0.8


def test_geometric_only_transform_does_not_clip():
    patches = np.stack(
        [np.full((8, 8, 8), 2.0, np.float32), np.zeros((8, 8, 8), np.float32)]
    )
    out = ImageTransforms("asinh")(patches)
    assert out.shape == patches.shape
    assert out[0].min() == 2.0 and out[0].max() == 2.0


def test_image_config_rejects_unknown_normalization():
    assert ImageConfig().normalization == "percentile"
    assert ImageConfig(normalization="asinh").normalization == "asinh"
    with pytest.raises(ValueError):
        ImageConfig(normalization="arcsinh")


def test_raw_noise_has_a_floor_on_flat_background():
    flat = np.full((16, 16, 16), 7, dtype=np.float32)
    noise = ImageTransforms("asinh").raw_noise
    out = [noise(flat.copy(), 0.0).std() for _ in range(10)]
    assert max(out) > 0.5


def test_random_rotation_is_a_rotation():
    for _ in range(10):
        rot = random_rotation()
        np.testing.assert_allclose(rot @ rot.T, np.eye(3), atol=1e-5)
        assert np.linalg.det(rot) == pytest.approx(1, abs=1e-5)


def test_affine_resamples_image_and_mask_together():
    rng = np.random.default_rng(0)
    img = rng.random((32, 32, 32), dtype=np.float32)
    mask = np.zeros((32, 32, 32), dtype=np.float32)
    mask[12:20, 12:20, 12:20] = 1
    for _ in range(10):
        out_img, out_mask = RandomAffine3D(p=1.0)(img, mask, (24, 24, 24))
        assert out_img.shape == out_mask.shape == (24, 24, 24)
        assert out_img.dtype == out_mask.dtype == np.float32
        assert set(np.unique(out_mask)) <= {0.0, 1.0}
        # The center of the patch is fixed by the rotation and scaling
        assert out_mask[11:13, 11:13, 11:13].all()
        assert out_img.min() >= 0 and out_img.max() <= 1


def test_affine_skipped_is_a_plain_resize():
    img = np.random.default_rng(0).random((32, 32, 32), dtype=np.float32)
    mask = (img > 0.5).astype(np.float32)
    out_img, out_mask = RandomAffine3D(p=0.0)(img, mask, (24, 24, 24))
    np.testing.assert_array_equal(out_img, img_util.resize(img, (24, 24, 24)))
    np.testing.assert_array_equal(
        out_mask, img_util.resize_nearest(mask, (24, 24, 24))
    )


def make_loader(monkeypatch, train):
    from neuron_proofreader.machine_learning import image_dataloader

    monkeypatch.setattr(image_dataloader, "TensorStoreImage", lambda path: None)
    config = ImageConfig(normalization="asinh")
    if train:
        config.set_train_mode()
    loader = image_dataloader.DetectionPatchLoader(None, config)
    config.set_val_mode()
    return loader


def test_patch_loader_keeps_train_transform_after_val_mode(monkeypatch):
    loader = make_loader(monkeypatch, train=True)
    assert loader.transform.normalization == "asinh"


def test_patch_loader_asinh_needs_the_mask(monkeypatch):
    loader = make_loader(monkeypatch, train=False)
    patch = sparse_patch()
    with pytest.raises(ValueError):
        loader.normalize(patch)

    # Validation is deterministic and puts the masked neurite near asinh(1)
    out = loader.normalize(patch, tube_mask())
    np.testing.assert_array_equal(out, loader.normalize(patch, tube_mask()))
    level = np.quantile(out[tube_mask() > 0], 0.7)
    assert level == pytest.approx(np.arcsinh(1), abs=0.02)


def test_patch_loader_asinh_training_varies_brightness(monkeypatch):
    loader = make_loader(monkeypatch, train=True)
    patch = sparse_patch()
    peaks = [loader.normalize(patch, tube_mask()).max() for _ in range(20)]
    assert max(peaks) - min(peaks) > 0.2
