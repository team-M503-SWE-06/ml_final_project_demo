"""Behaviour tests for the from-scratch Pegasos SVM."""

import numpy as np
import pytest
from sklearn.svm import SVC

from src.svm import PegasosSVM, RandomFourierFeatures, RFFPegasosSVM


@pytest.fixture
def blobs():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(-1.5, 1.0, size=(200, 2)), rng.normal(1.5, 1.0, size=(200, 2))])
    y = np.r_[np.zeros(200, int), np.ones(200, int)]
    return X, y


def test_objective_decreases(blobs):
    X, y = blobs
    m = PegasosSVM(lambda_=1e-2, n_epochs=30, batch_size=1).fit(X, y)
    h = m.loss_history_
    assert h[-1] < h[0]
    assert h[-1] <= min(h[1:5]) + 1e-3          # converged, not diverging


def test_matches_sklearn_svc_on_toy_split(blobs):
    X, y = blobs
    lam = 1e-2
    ours = PegasosSVM(lambda_=lam, n_epochs=200, batch_size=None, project=False).fit(X, y)
    # same objective: C = 1 / (lambda * n)
    sk = SVC(kernel="linear", C=1.0 / (lam * len(y))).fit(X, y)
    agree = np.mean(ours.predict(X) == sk.predict(X))
    assert agree > 0.97
    cos = ours.w @ sk.coef_[0] / (np.linalg.norm(ours.w) * np.linalg.norm(sk.coef_[0]))
    assert cos > 0.99
    assert ours.norm_w_ == pytest.approx(np.linalg.norm(sk.coef_[0]), rel=0.15)


def test_support_vectors_are_on_or_inside_margin(blobs):
    X, y = blobs
    m = PegasosSVM(lambda_=1e-2, n_epochs=30).fit(X, y)
    sv = m.support_vector_mask(X, y)
    ypm = np.where(y == 1, 1, -1)
    assert np.all(ypm[sv] * m.decision_function(X[sv]) <= 1)
    assert 0 < sv.sum() < len(y)
    assert m.margin_width_ == pytest.approx(2 / np.linalg.norm(m.w))


def test_lambda_controls_margin(blobs):
    X, y = blobs
    wide = PegasosSVM(lambda_=1.0, n_epochs=20).fit(X, y).margin_width_
    narrow = PegasosSVM(lambda_=1e-3, n_epochs=20).fit(X, y).margin_width_
    assert wide > narrow


def test_labels_are_mapped_back(blobs):
    X, y = blobs
    labels = np.where(y == 1, "premium", "standard")
    m = PegasosSVM(lambda_=1e-2, n_epochs=5).fit(X, labels)
    assert set(m.predict(X)) <= {"premium", "standard"}
    assert np.mean(m.predict(X) == labels) > 0.9


def test_rff_approximates_rbf_kernel():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 3))
    gamma = 0.5
    Z = RandomFourierFeatures(n_components=20000, gamma=gamma).fit(X).transform(X)
    K = np.exp(-gamma * ((X[:, None] - X[None]) ** 2).sum(-1))
    assert np.abs(Z @ Z.T - K).max() < 0.05


def test_rff_svm_solves_xor():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(600, 2))
    y = (X[:, 0] * X[:, 1] > 0).astype(int)
    linear = PegasosSVM(lambda_=1e-3, n_epochs=20).fit(X, y)
    rff = RFFPegasosSVM(n_components=300, gamma=2.0, lambda_=1e-4, n_epochs=40).fit(X, y)
    assert np.mean(linear.predict(X) == y) < 0.7
    assert np.mean(rff.predict(X) == y) > 0.9
