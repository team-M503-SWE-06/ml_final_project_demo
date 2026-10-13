"""
decision_tree.py — a CART-style decision tree built FROM SCRATCH (NumPy only).

Supports both tasks:
  - classification (criterion = "gini" or "entropy")  -> price tier
  - regression     (criterion = "mse")                -> log(price)

Split search
------------
For a node with samples S and a feature j, sort S by x_j once. Every split
"x_j <= t" with t between two consecutive DISTINCT sorted values sends a
prefix of the sorted order left and the suffix right. With prefix sums we
get the statistics of ALL candidate splits of that feature at once:
  - classification: cumulative class counts  -> p_k(left), p_k(right)
  - regression:     cumulative sum(y), sum(y^2) -> SSE(left), SSE(right)
and score each candidate by the impurity decrease (information gain)

    IG = I(parent) - N_L/N * I(left) - N_R/N * I(right).

The threshold is the midpoint between the two neighbouring values (the
same convention as scikit-learn). The best (feature, threshold) over all
features wins; ties go to the lowest feature index, then the lowest
threshold (sklearn breaks feature ties by a random permutation instead).

scikit-learn is NOT used anywhere in this file.
"""

from __future__ import annotations

import numpy as np


class Node:
    """One node of the tree."""
    __slots__ = ("feature", "threshold", "left", "right", "value",
                 "impurity", "n_samples", "depth")

    def __init__(self):
        self.feature = None      # int: feature index to split on (internal)
        self.threshold = None    # float: go left if x[feature] <= threshold
        self.left = None         # Node
        self.right = None        # Node
        self.value = None        # leaf/node prediction: class counts or mean
        self.impurity = 0.0
        self.n_samples = 0
        self.depth = 0

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


# --- impurity functions (on class-count / moment arrays, vectorised) ----------
def gini_from_counts(counts):
    """G = 1 - sum_k p_k^2 ; counts shape (..., K)."""
    n = counts.sum(axis=-1, keepdims=True)
    p = counts / np.maximum(n, 1)
    return 1.0 - np.sum(p * p, axis=-1)


def entropy_from_counts(counts):
    """H = -sum_k p_k log2 p_k  (with 0 log 0 = 0)."""
    n = counts.sum(axis=-1, keepdims=True)
    p = counts / np.maximum(n, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(p > 0, np.log2(p), 0.0)
    return -np.sum(p * logp, axis=-1)


def mse_from_moments(n, s, ss):
    """I_reg = 1/N sum (y - ybar)^2 = ss/n - (s/n)^2."""
    n = np.maximum(n, 1)
    return np.maximum(ss / n - (s / n) ** 2, 0.0)


class DecisionTree:
    def __init__(
        self,
        task: str = "classification",   # "classification" | "regression"
        criterion: str = "gini",        # "gini" | "entropy" | "mse"
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        random_state: int = 42,
    ):
        self.task = task
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state   # unused: the search is exhaustive & deterministic
        self.root: Node | None = None

    # --- impurity -----------------------------------------------------------
    def _impurity(self, y) -> float:
        """Gini / entropy (classification) or MSE/variance (regression) of labels y."""
        if self.task == "regression":
            y = np.asarray(y, dtype=float)
            return float(mse_from_moments(len(y), y.sum(), (y * y).sum()))
        counts = np.bincount(y, minlength=self.n_classes_).astype(float)
        f = gini_from_counts if self.criterion == "gini" else entropy_from_counts
        return float(f(counts))

    def _node_value(self, y):
        if self.task == "regression":
            return float(np.mean(y))
        return np.bincount(y, minlength=self.n_classes_).astype(float)

    def _best_split(self, X, y):
        """
        Search all features & candidate thresholds; return
        (feature, threshold, impurity_decrease) of the split with the largest
        information gain, or None if no valid split exists.
        """
        n, d = X.shape
        msl = self.min_samples_leaf
        parent = self._impurity(y)
        best = None
        best_gain = -np.inf

        # sizes of the left child for split position i (left = first i+1 rows)
        n_left = np.arange(1, n, dtype=float)
        n_right = n - n_left
        size_ok = (n_left >= msl) & (n_right >= msl)
        if not size_ok.any():
            return None

        if self.task == "classification":
            onehot = np.eye(self.n_classes_)[y]                 # (n, K)
            total = onehot.sum(axis=0)
            imp = gini_from_counts if self.criterion == "gini" else entropy_from_counts
        else:
            yf = y.astype(float)

        for j in range(d):
            order = np.argsort(X[:, j], kind="stable")
            xs = X[order, j]
            # candidate positions: between two DISTINCT values, both children big enough
            pos = np.flatnonzero(size_ok & (xs[1:] > xs[:-1]))
            if pos.size == 0:
                continue
            nl, nr = n_left[pos], n_right[pos]
            if self.task == "classification":
                left = np.cumsum(onehot[order], axis=0)[pos]    # class counts left of each cut
                right = total - left
                child = (nl * imp(left) + nr * imp(right)) / n
            else:
                ys = yf[order]
                cs, css = np.cumsum(ys), np.cumsum(ys * ys)
                sl, ssl = cs[pos], css[pos]
                sr, ssr = cs[-1] - sl, css[-1] - ssl
                child = (nl * mse_from_moments(nl, sl, ssl)
                         + nr * mse_from_moments(nr, sr, ssr)) / n
            gain = parent - child
            k = int(np.argmax(gain))                            # first max = lowest threshold
            if gain[k] > best_gain + 1e-12:                     # strict: lower feature wins ties
                best_gain = gain[k]
                i = pos[k]
                t = (xs[i] + xs[i + 1]) / 2.0
                if t >= xs[i + 1]:          # adjacent floats: the midpoint rounds up onto
                    t = xs[i]               # the right value -> use the left value instead
                best = (j, t, float(gain[k]))
        return best

    # --- grow / predict -----------------------------------------------------
    def _grow(self, X, y, depth: int) -> Node:
        """Recursively build the tree, respecting the stopping rules."""
        node = Node()
        node.n_samples = len(y)
        node.depth = depth
        node.value = self._node_value(y)
        node.impurity = self._impurity(y)

        # stopping rules (each one is a hyperparameter that changes the tree)
        if (self.max_depth is not None and depth >= self.max_depth) \
                or len(y) < self.min_samples_split \
                or len(y) < 2 * self.min_samples_leaf \
                or node.impurity <= 1e-12:                      # pure node
            return node

        split = self._best_split(X, y)
        if split is None:
            return node
        j, t, gain = split
        # sklearn's definition: weighted decrease N_t/N * IG must reach the minimum
        weighted = len(y) / self.n_samples_ * gain
        if gain <= 1e-12 or weighted + 1e-12 < self.min_impurity_decrease:
            return node

        mask = X[:, j] <= t
        if mask.all() or not mask.any():                        # defensive: never create empty children
            return node
        node.feature, node.threshold = j, t
        self.feature_importances_[j] += len(y) * gain           # N_t * IG = total impurity decrease
        node.left = self._grow(X[mask], y[mask], depth + 1)
        node.right = self._grow(X[~mask], y[~mask], depth + 1)
        return node

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if self.task not in ("classification", "regression"):
            raise ValueError("task must be 'classification' or 'regression'")
        if self.task == "classification":
            if self.criterion not in ("gini", "entropy"):
                raise ValueError("classification criterion must be 'gini' or 'entropy'")
            self.classes_, y = np.unique(y, return_inverse=True)  # labels -> 0..K-1
            self.n_classes_ = len(self.classes_)
        else:
            if self.criterion != "mse":
                raise ValueError("regression criterion must be 'mse'")
            y = y.astype(float)
        self.n_samples_, self.n_features_in_ = X.shape
        self.feature_importances_ = np.zeros(self.n_features_in_)
        self.root = self._grow(X, y, depth=0)
        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ /= total
        return self

    def _predict_one(self, x, node: Node):
        while not node.is_leaf():
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.value

    def _leaf_values(self, X):
        """Route all rows at once: partition index sets down the tree."""
        X = np.asarray(X, dtype=float)
        out = [None] * len(X)
        stack = [(self.root, np.arange(len(X)))]
        while stack:
            node, idx = stack.pop()
            if node.is_leaf():
                for i in idx:
                    out[i] = node.value
                continue
            m = X[idx, node.feature] <= node.threshold
            stack.append((node.left, idx[m]))
            stack.append((node.right, idx[~m]))
        return out

    def predict(self, X):
        vals = self._leaf_values(X)
        if self.task == "regression":
            return np.asarray(vals, dtype=float)
        counts = np.vstack(vals)
        return self.classes_[np.argmax(counts, axis=1)]          # majority class

    def predict_proba(self, X):
        """Classification only: class probabilities = leaf class frequencies."""
        if self.task != "classification":
            raise AttributeError("predict_proba is only defined for classification")
        counts = np.vstack(self._leaf_values(X))
        return counts / counts.sum(axis=1, keepdims=True)

    # --- introspection -------------------------------------------------------
    def _nodes(self):
        stack = [self.root]
        while stack:
            node = stack.pop()
            yield node
            if not node.is_leaf():
                stack += [node.right, node.left]

    def get_depth(self) -> int:
        return max(n.depth for n in self._nodes())

    def get_n_leaves(self) -> int:
        return sum(n.is_leaf() for n in self._nodes())

    def node_count(self) -> int:
        return sum(1 for _ in self._nodes())

    def splits(self):
        """Pre-order list of (depth, feature, threshold) for internal nodes."""
        return [(n.depth, n.feature, n.threshold) for n in self._nodes() if not n.is_leaf()]
