"""Correctness tests for the from-scratch decision tree (sklearn used as the reference only)."""

import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from src.decision_tree import DecisionTree, entropy_from_counts, gini_from_counts


def sk_splits(sk):
    """Pre-order (depth, feature, threshold) of sklearn's internal nodes."""
    t = sk.tree_
    out, stack = [], [(0, 0)]
    while stack:
        node, depth = stack.pop()
        if t.children_left[node] == -1:
            continue
        out.append((depth, int(t.feature[node]), float(t.threshold[node])))
        stack += [(t.children_right[node], depth + 1), (t.children_left[node], depth + 1)]
    return out


# A tiny, fully-determined example: integer features (thresholds land on .5,
# exactly representable in sklearn's float32) and no ties in the gains.
X_TOY = np.array([[1, 7], [2, 3], [3, 8], [4, 1], [5, 6], [6, 2],
                  [7, 9], [8, 4], [9, 5], [10, 12], [11, 10], [12, 11]], dtype=float)
Y_CLS = np.array([0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1])
Y_REG = np.array([1.0, 1.2, 3.1, 0.7, 2.9, 0.9, 4.2, 1.5, 3.3, 6.0, 5.5, 7.1])


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_classifier_matches_sklearn_splits(criterion):
    ours = DecisionTree("classification", criterion).fit(X_TOY, Y_CLS)
    sk = DecisionTreeClassifier(criterion=criterion, random_state=0).fit(X_TOY, Y_CLS)
    assert ours.splits() == sk_splits(sk)
    np.testing.assert_array_equal(ours.predict(X_TOY), sk.predict(X_TOY))
    np.testing.assert_allclose(ours.predict_proba(X_TOY), sk.predict_proba(X_TOY))


@pytest.mark.parametrize("max_depth", [1, 2, 3])
def test_regressor_matches_sklearn_splits(max_depth):
    ours = DecisionTree("regression", "mse", max_depth=max_depth).fit(X_TOY, Y_REG)
    sk = DecisionTreeRegressor(max_depth=max_depth, random_state=0).fit(X_TOY, Y_REG)
    assert [(d, f, round(t, 4)) for d, f, t in ours.splits()] == \
           [(d, f, round(t, 4)) for d, f, t in sk_splits(sk)]
    np.testing.assert_allclose(ours.predict(X_TOY), sk.predict(X_TOY))


def test_fully_grown_regressor_matches_sklearn_predictions():
    # Grown to purity, 2-sample nodes can be split equally well on either
    # feature (a gain TIE); sklearn breaks ties with a random feature
    # permutation, we take the lowest feature index. The training-set
    # predictions are identical either way.
    ours = DecisionTree("regression", "mse").fit(X_TOY, Y_REG)
    sk = DecisionTreeRegressor(random_state=0).fit(X_TOY, Y_REG)
    np.testing.assert_allclose(ours.predict(X_TOY), sk.predict(X_TOY))
    assert ours.get_n_leaves() == sk.get_n_leaves()


def test_tie_breaking_prefers_lowest_feature_index():
    # column 1 is an exact copy of column 0 -> every split ties
    X = np.repeat(np.arange(8.0).reshape(-1, 1), 2, axis=1)
    y = np.array([0, 0, 0, 1, 1, 1, 1, 1])
    t = DecisionTree("classification", "gini").fit(X, y)
    assert t.splits() == [(0, 0, 2.5)]


def test_feature_importances_match_sklearn():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    ours = DecisionTree("classification", "gini", max_depth=4).fit(X, y)
    sk = DecisionTreeClassifier(max_depth=4, random_state=0).fit(X, y)
    np.testing.assert_allclose(ours.feature_importances_, sk.feature_importances_, atol=1e-6)


def test_impurity_formulas():
    assert gini_from_counts(np.array([5.0, 5.0])) == pytest.approx(0.5)
    assert entropy_from_counts(np.array([5.0, 5.0])) == pytest.approx(1.0)
    assert entropy_from_counts(np.array([4.0, 0.0])) == pytest.approx(0.0)
    t = DecisionTree("regression", "mse")
    assert t._impurity(np.array([1.0, 3.0])) == pytest.approx(1.0)


@pytest.fixture
def noisy():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(400, 5))
    y = (X[:, 0] * X[:, 1] + 0.3 * rng.normal(size=400) > 0).astype(int)
    return X, y


@pytest.mark.parametrize("param,small,large", [
    ("max_depth", 3, None),
    ("min_samples_split", 2, 60),
    ("min_samples_leaf", 1, 25),
    ("min_impurity_decrease", 0.0, 0.01),
])
def test_each_hyperparameter_changes_the_tree(noisy, param, small, large):
    X, y = noisy
    a = DecisionTree("classification", "gini", **{param: small}).fit(X, y)
    b = DecisionTree("classification", "gini", **{param: large}).fit(X, y)
    assert a.get_n_leaves() != b.get_n_leaves()


def test_min_samples_leaf_respected(noisy):
    X, y = noisy
    t = DecisionTree("classification", "gini", min_samples_leaf=17).fit(X, y)
    assert min(n.n_samples for n in t._nodes() if n.is_leaf()) >= 17


def test_max_depth_respected(noisy):
    X, y = noisy
    assert DecisionTree("classification", "entropy", max_depth=4).fit(X, y).get_depth() <= 4


def test_predict_proba_sums_to_one_and_labels_preserved(noisy):
    X, y = noisy
    labels = np.where(y == 1, "premium", "standard")
    t = DecisionTree("classification", "gini", max_depth=3).fit(X, labels)
    p = t.predict_proba(X)
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    assert set(t.predict(X)) <= {"premium", "standard"}


def test_adjacent_float_values_split_correctly():
    # midpoint of two neighbouring doubles rounds onto the larger one
    a = 40.0
    b = np.nextafter(a, 50.0)
    X = np.array([[a], [a], [b], [b]])
    y = np.array([0, 0, 1, 1])
    t = DecisionTree("classification", "gini").fit(X, y)
    np.testing.assert_array_equal(t.predict(X), y)
    assert t.get_n_leaves() == 2


def test_pure_data_gives_single_leaf():
    t = DecisionTree("classification", "gini").fit(np.arange(10).reshape(-1, 1), np.ones(10, int))
    assert t.get_n_leaves() == 1
