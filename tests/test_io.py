import numpy as np

from src.io import (
    DATA_ROOT,
    N_AXES,
    N_TIMESTEPS,
    TIMESTEPS_MS,
    kfold_split,
    load_labels,
    load_sample,
)


def test_timesteps_grid():
    assert TIMESTEPS_MS.tolist() == list(range(-400, 1, 40))
    assert len(TIMESTEPS_MS) == N_TIMESTEPS == 11


def test_load_sample_shape_and_no_nan():
    arr = load_sample("TRAIN_00001")
    assert arr.shape == (N_TIMESTEPS, N_AXES)
    assert np.isfinite(arr).all()


def test_load_sample_subset_consistency():
    for sid in ["TRAIN_00001", "TRAIN_05000", "TRAIN_09999", "TEST_00001", "TEST_05000"]:
        split = "train" if sid.startswith("TRAIN") else "test"
        arr = load_sample(sid, split=split)
        assert arr.shape == (N_TIMESTEPS, N_AXES)
        assert np.isfinite(arr).all()


def test_labels_match_train_files():
    ids, y = load_labels()
    assert len(ids) == 10000
    assert y.shape == (10000, N_AXES)
    assert np.isfinite(y).all()
    train_files = {p.stem for p in (DATA_ROOT / "train").glob("*.csv")}
    assert set(ids) == train_files


def test_kfold_split_partition():
    ids = [f"TRAIN_{i:05d}" for i in range(1, 10001)]
    folds = kfold_split(ids, k=5, seed=42)
    assert len(folds) == 5
    val_all = []
    for tr, va in folds:
        assert len(tr) == 8000
        assert len(va) == 2000
        assert len(set(tr.tolist()) & set(va.tolist())) == 0
        val_all.extend(va.tolist())
    assert sorted(val_all) == list(range(10000))


def test_kfold_split_determinism():
    ids = [f"TRAIN_{i:05d}" for i in range(1, 101)]
    f1 = kfold_split(ids, k=5, seed=42)
    f2 = kfold_split(ids, k=5, seed=42)
    for (a_tr, a_va), (b_tr, b_va) in zip(f1, f2):
        assert np.array_equal(a_tr, b_tr)
        assert np.array_equal(a_va, b_va)


def test_kfold_split_stride_pattern():
    ids = [f"S_{i:04d}" for i in range(20)]
    folds = kfold_split(ids, k=5)
    assert folds[0][1].tolist() == [0, 5, 10, 15]
    assert folds[1][1].tolist() == [1, 6, 11, 16]
    assert folds[4][1].tolist() == [4, 9, 14, 19]
