"""Our NumPy metrics vs sklearn's, plus leakage-safety checks of data_prep."""

import numpy as np
import pandas as pd
import pytest
from sklearn import metrics as skm

from src import data_prep as dp
from src import evaluate as ev


@pytest.fixture
def preds():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    s = y + rng.normal(0, 1.0, 500)
    s[:50] = np.round(s[:50])          # create ties
    return y, (s > 0.5).astype(int), s


def test_regression_metrics():
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=100), rng.normal(size=100)
    assert ev.rmse(a, b) == pytest.approx(np.sqrt(skm.mean_squared_error(a, b)))
    assert ev.mae(a, b) == pytest.approx(skm.mean_absolute_error(a, b))
    assert ev.r2(a, b) == pytest.approx(skm.r2_score(a, b))


def test_classification_metrics(preds):
    y, yhat, s = preds
    np.testing.assert_array_equal(ev.confusion_matrix(y, yhat), skm.confusion_matrix(y, yhat))
    p, r, f = ev.precision_recall_f1(y, yhat)
    assert p == pytest.approx(skm.precision_score(y, yhat))
    assert r == pytest.approx(skm.recall_score(y, yhat))
    assert f == pytest.approx(skm.f1_score(y, yhat))
    assert ev.roc_auc(y, s) == pytest.approx(skm.roc_auc_score(y, s))
    assert ev.average_precision(y, s) == pytest.approx(skm.average_precision_score(y, s))


def test_kfold_partitions_rows():
    folds = ev.kfold_indices(103, k=5, strata=np.arange(103) % 2)
    test_rows = np.sort(np.concatenate([te for _, te in folds]))
    np.testing.assert_array_equal(test_rows, np.arange(103))
    for tr, te in folds:
        assert len(np.intersect1d(tr, te)) == 0


def test_tier_threshold_comes_from_train_only():
    train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_tr, thr = dp.make_tier_label(train)
    assert thr == 3.0
    y_te, thr2 = dp.make_tier_label(np.array([100.0, 0.0]), thr)
    assert thr2 == thr and list(y_te) == [1, 0]


def test_standardize_uses_train_stats():
    Xtr = np.array([[0.0, 1.0], [2.0, 1.0]])
    Xte = np.array([[4.0, 5.0]])
    a, b = dp.standardize(Xtr, Xte)
    np.testing.assert_allclose(a.mean(0), 0)
    np.testing.assert_allclose(b, [[3.0, 4.0]])   # (4-1)/1, constant column: (5-1)/1


def test_split_is_disjoint_and_stratified():
    strata = np.repeat(np.arange(10), 100)
    tr, va, te = dp.stratified_split_indices(strata, 0.15, 0.15, seed=0)
    assert len(np.intersect1d(tr, te)) == 0 and len(np.intersect1d(va, te)) == 0
    assert len(tr) + len(va) + len(te) == 1000
    assert np.all(np.bincount(strata[te]) == 15)


def test_feature_builder_has_no_leakage_columns():
    df = pd.DataFrame({
        "price": [1e5, 2e5, 3e5], "area_m2": [50.0, 80.0, 120.0], "rooms": [1.0, 2.0, np.nan],
        "floor": [1.0, 5.0, np.nan], "total_floors": [5.0, 9.0, np.nan], "land_sot": [np.nan, np.nan, 3.0],
        "lat": [40.4] * 3, "lng": [49.8] * 3, "category": ["new_build", "old_build", "house"],
        "district": ["Yasamal r."] * 3, "neighbourhood": ["x"] * 3, "near_metro": [1.0, 0.0, 0.0],
        "repaired": [1.0, np.nan, 0.0], "bill_of_sale": [1.0] * 3, "mortgage": [0.0] * 3,
        "vip": [0.0] * 3, "featured": [0.0] * 3, "is_agent": [1.0] * 3, "is_complex": [0.0] * 3,
    })
    X, y, names = dp.FeatureBuilder(min_count=1).fit(df).transform(df)
    assert not np.isnan(X).any()
    assert not any("price" in n for n in names)
    np.testing.assert_allclose(y, np.log(df["price"]))
