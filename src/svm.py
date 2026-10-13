"""
svm.py — a soft-margin SVM built FROM SCRATCH via PEGASOS (NumPy only).

No QP / SMO solver. Pegasos minimises the regularised hinge-loss primal by
(mini-batch) stochastic sub-gradient descent:

    J(w, b) = (lambda/2) * ||w||^2 + (1/n) * sum_i max(0, 1 - y_i (w . x_i + b))

with labels y_i in {-1, +1}. Per step t on a mini-batch B (|B| = 1 is the
original Pegasos), with A = {i in B : y_i (w . x_i + b) < 1}:

    w <- (1 - eta_t*lambda) w + (eta_t/|B|) * sum_{i in A} y_i x_i

Bias: following the project statement we fold b into w by appending a
constant 1 feature (so b receives the same update). Note this means b is
also (weakly) regularised; J is reported for exactly the objective we
optimise.

Learning-rate schedules (a hyperparameter):
    "pegasos"     eta_t = 1 / (lambda * t)        (the paper's schedule)
    "invscaling"  eta_t = eta0 / sqrt(t)
    "constant"    eta_t = eta0

Optional Pegasos projection: after each step, w is projected onto the ball
||w|| <= 1/sqrt(lambda), which contains the optimum.

Optional suffix averaging (average=True): return the average of the iterates
from the second half of training instead of the last iterate. The Pegasos
convergence guarantee is for averaged iterates; in practice it removes the
step-to-step noise of the 1/(lambda t) schedule when lambda is small.

Non-linear extension — choice (ii) RANDOM FOURIER FEATURES:
    phi(x) = sqrt(2/D) cos(W x + c),   W_ij ~ N(0, 2*gamma),  c_j ~ U[0, 2*pi)
so that phi(x).phi(z) ~= exp(-gamma ||x - z||^2) (RBF kernel, Rahimi & Recht
2007). We then run the SAME linear Pegasos on phi(X). Why RFF rather than
kernelised Pegasos: training/prediction cost is O(n*D) instead of O(n^2)
kernel evaluations, which matters with ~40k training rows, and we reuse the
tested linear solver unchanged.

scikit-learn's SVC / SGDClassifier are BASELINES only — not used here.
"""

from __future__ import annotations

import numpy as np


class PegasosSVM:
    def __init__(
        self,
        lambda_: float = 1e-4,          # regularisation strength (C = 1/(lambda*n))
        n_epochs: int = 20,             # passes over the training data
        batch_size: int | None = 32,    # 1 = classic Pegasos; None = full batch
        lr_schedule: str = "pegasos",   # "pegasos" | "invscaling" | "constant"
        eta0: float = 0.1,              # base rate for invscaling / constant
        fit_intercept: bool = True,
        project: bool = True,
        average: bool = True,
        random_state: int = 42,
    ):
        self.lambda_ = lambda_
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr_schedule = lr_schedule
        self.eta0 = eta0
        self.fit_intercept = fit_intercept
        self.project = project
        self.average = average
        self.random_state = random_state
        self.w = None              # weight vector (without the bias)
        self.b = 0.0               # bias

    # --- labels ----------------------------------------------------------------
    def _to_pm1(self, y):
        """Map 2-class labels to {-1, +1} (larger label -> +1). Stores the mapping."""
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("PegasosSVM is a binary classifier")
        return np.where(y == self.classes_[1], 1.0, -1.0)

    def _augment(self, X):
        X = np.asarray(X, dtype=float)
        return np.hstack([X, np.ones((len(X), 1))]) if self.fit_intercept else X

    def _eta(self, t):
        if self.lr_schedule == "pegasos":
            return 1.0 / (self.lambda_ * t)
        if self.lr_schedule == "invscaling":
            return self.eta0 / np.sqrt(t)
        if self.lr_schedule == "constant":
            return self.eta0
        raise ValueError(f"unknown lr_schedule {self.lr_schedule!r}")

    def objective(self, Xa, ypm, w):
        """J(w) = lambda/2 ||w||^2 + mean hinge, on augmented X."""
        hinge = np.maximum(0.0, 1.0 - ypm * (Xa @ w))
        return 0.5 * self.lambda_ * float(w @ w) + float(hinge.mean())

    # --- training ----------------------------------------------------------------
    def fit(self, X, y):
        """
        Run (mini-batch) Pegasos for n_epochs passes. Each epoch shuffles the
        data and walks through it in mini-batches; the objective J on the full
        training set is recorded after every epoch in loss_history_.
        """
        ypm = self._to_pm1(np.asarray(y))
        Xa = self._augment(X)
        n, d = Xa.shape
        bs = n if self.batch_size is None else int(self.batch_size)
        rng = np.random.default_rng(self.random_state)
        w = np.zeros(d)
        radius = 1.0 / np.sqrt(self.lambda_)
        t = 0
        w_sum, n_avg = np.zeros(d), 0
        avg_from = self.n_epochs // 2                           # average the second half
        self.loss_history_ = [self.objective(Xa, ypm, w)]
        for epoch in range(self.n_epochs):
            perm = rng.permutation(n)
            for start in range(0, n, bs):
                idx = perm[start:start + bs]
                t += 1
                eta = self._eta(t)
                Xb, yb = Xa[idx], ypm[idx]
                active = yb * (Xb @ w) < 1.0                    # hinge is "on" for these
                grad_hinge = (yb[active, None] * Xb[active]).sum(axis=0) / len(idx)
                w = (1.0 - eta * self.lambda_) * w + eta * grad_hinge
                if self.project:
                    nw = np.linalg.norm(w)
                    if nw > radius:
                        w *= radius / nw
                if self.average and epoch >= avg_from:
                    w_sum += w
                    n_avg += 1
            # objective of the model we would return if we stopped now
            w_out = w_sum / n_avg if n_avg else w
            self.loss_history_.append(self.objective(Xa, ypm, w_out))
        w = w_sum / n_avg if n_avg else w
        self.n_steps_ = t
        if self.fit_intercept:
            self.w, self.b = w[:-1].copy(), float(w[-1])
        else:
            self.w, self.b = w.copy(), 0.0
        return self

    # --- margin / support vectors -------------------------------------------------
    @property
    def norm_w_(self) -> float:
        return float(np.linalg.norm(self.w))

    @property
    def margin_width_(self) -> float:
        """Geometric width of the margin band: 2 / ||w||."""
        return 2.0 / self.norm_w_

    def support_vector_mask(self, X, y):
        """Training points on or inside the margin: y_i (w.x_i + b) <= 1."""
        ypm = np.where(np.asarray(y) == self.classes_[1], 1.0, -1.0)
        return ypm * self.decision_function(X) <= 1.0

    # --- inference ------------------------------------------------------------------
    def decision_function(self, X):
        """Signed score w . x + b for each row."""
        return np.asarray(X, dtype=float) @ self.w + self.b

    def predict(self, X):
        """Class labels: sign of the decision function mapped back to the original labels."""
        return np.where(self.decision_function(X) >= 0, self.classes_[1], self.classes_[0])


class RandomFourierFeatures:
    """Random features whose inner products approximate the RBF kernel exp(-gamma ||x-z||^2)."""

    def __init__(self, n_components: int = 500, gamma: float = 0.1, random_state: int = 42):
        self.n_components = n_components
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X):
        d = np.asarray(X).shape[1]
        rng = np.random.default_rng(self.random_state)
        self.W_ = rng.normal(0.0, np.sqrt(2.0 * self.gamma), size=(d, self.n_components))
        self.c_ = rng.uniform(0.0, 2 * np.pi, size=self.n_components)
        return self

    def transform(self, X):
        return np.sqrt(2.0 / self.n_components) * np.cos(np.asarray(X, dtype=float) @ self.W_ + self.c_)


class RFFPegasosSVM:
    """Non-linear SVM: random Fourier feature map + the linear PegasosSVM above."""

    def __init__(self, n_components: int = 500, gamma: float = 0.1, random_state: int = 42,
                 **svm_kwargs):
        self.rff = RandomFourierFeatures(n_components, gamma, random_state)
        self.svm = PegasosSVM(random_state=random_state, **svm_kwargs)

    def fit(self, X, y):
        Z = self.rff.fit(X).transform(X)
        self.svm.fit(Z, y)
        self.classes_ = self.svm.classes_
        self.loss_history_ = self.svm.loss_history_
        return self

    def decision_function(self, X):
        return self.svm.decision_function(self.rff.transform(X))

    def predict(self, X):
        return self.svm.predict(self.rff.transform(X))

    @property
    def margin_width_(self) -> float:
        return self.svm.margin_width_
