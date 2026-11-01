import numpy as np
import pytest

from quito_building_lab.training import pair_files, segmentation_metrics, spatial_split


def test_spatial_split_whole_blocks_and_reproducible():
    cents = [(x * 10.0, y * 10.0) for x in range(40) for y in range(40)]
    a = spatial_split(cents, 100.0, seed=1)
    assert a == spatial_split(cents, 100.0, seed=1)
    blocks = {}
    for (x, y), s in zip(cents, a, strict=True):
        blocks.setdefault((int(x // 100), int(y // 100)), set()).add(s)
    assert all(len(v) == 1 for v in blocks.values())  # no block shared between splits
    assert {"train", "val", "test"} <= set(a)


def test_pairing_requires_matching_stems(tmp_path):
    (tmp_path / "i").mkdir()
    (tmp_path / "m").mkdir()
    (tmp_path / "i" / "a.tif").write_bytes(b"")
    (tmp_path / "m" / "b.tif").write_bytes(b"")
    with pytest.raises(ValueError):
        pair_files(str(tmp_path / "i"), str(tmp_path / "m"))


def test_segmentation_metrics():
    t = np.zeros((10, 10), bool)
    t[:5] = True
    assert segmentation_metrics(t, t)["iou"] == 1.0
    p = np.zeros((10, 10), bool)
    p[:5, :5] = True
    assert segmentation_metrics(p, t)["iou"] == pytest.approx(0.5)
