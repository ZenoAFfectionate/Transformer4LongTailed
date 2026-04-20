"""Extreme Learning Machine (ELM) classifier head.

An ELM is a single-hidden-layer feedforward network with randomly generated
hidden weights/biases and a closed-form least-squares solution for the output
layer. This module exposes :class:`ELM`, a scikit-learn-style estimator whose
``fit`` / ``predict`` / ``predict_proba`` contract matches the BLS / ARBN
classifiers already used by ``train_stage2`` — so it can be plugged in as a
stage2 classifier without any training-loop changes.

Key features beyond a vanilla ELM:

* Ridge-regularised output weights with dual/primal formulation (whichever
  linear system is smaller) and Cholesky solve + SVD fallback.
* Optional class-frequency reweighting (``cls_num_list`` + ``class_weight_beta``)
  so long-tail distributions can down-weight head classes during the solve,
  mirroring ARBN's adaptive scheme.
* Deterministic randomness via ``random_state``.
* Softmax probabilities from the regression scores for calibration / ECE.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array


DEFAULT_DTYPE = np.float32


ACTIVATIONS = {
    'linear': lambda x: x,
    'relu': lambda x: np.maximum(0.0, x),
    'sigmoid': lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0))),
    'tanh': lambda x: np.tanh(x),
    'leaky_relu': lambda x: np.where(x > 0, x, 0.01 * x),
}


class ELM(BaseEstimator):
    """Closed-form Extreme Learning Machine classifier.

    Parameters
    ----------
    n_hidden : int
        Number of hidden neurons in the random projection layer.
    n_classes : int
        Number of output classes.
    activation : str
        Hidden-layer activation. One of ``linear``, ``relu``, ``sigmoid``,
        ``tanh``, ``leaky_relu``.
    reg : float
        L2 regularisation strength added to the normal equations.
    cls_num_list : sequence of int, optional
        Per-class training counts. When provided, sample-weights for the
        least-squares solve are set to ``(1 / n_c) ** class_weight_beta``.
    class_weight_beta : float
        Exponent controlling how aggressively tail classes are re-weighted.
    random_state : int, optional
        Seed for the random hidden weights/biases.
    orthogonalize : bool
        If True, orthogonalise the input-to-hidden weights via QR to stabilise
        the hidden representation.
    """

    def __init__(
        self,
        n_hidden: int = 1024,
        n_classes: int = 10,
        activation: str = 'relu',
        reg: float = 1e-2,
        cls_num_list: Optional[list] = None,
        class_weight_beta: float = 0.5,
        random_state: Optional[int] = None,
        orthogonalize: bool = False,
    ):
        if activation not in ACTIVATIONS:
            raise ValueError(
                f"Unknown activation '{activation}'. "
                f"Expected one of: {sorted(ACTIVATIONS.keys())}"
            )
        self.n_hidden = int(n_hidden)
        self.n_classes = int(n_classes)
        self.activation = activation
        self.reg = float(reg)
        self.cls_num_list = cls_num_list
        self.class_weight_beta = float(class_weight_beta)
        self.random_state = random_state
        self.orthogonalize = bool(orthogonalize)

        # Parameters populated by ``fit``.
        self.W_in_: Optional[np.ndarray] = None
        self.b_in_: Optional[np.ndarray] = None
        self.beta_: Optional[np.ndarray] = None
        self.is_fitted: bool = False
        self.class_weights_: Optional[np.ndarray] = None

        self._initialise_class_weights()
        self._log_init()

    def _initialise_class_weights(self) -> None:
        if self.cls_num_list is None:
            self.class_weights_ = np.ones(self.n_classes, dtype=DEFAULT_DTYPE)
            return

        counts = np.asarray(self.cls_num_list, dtype=np.float64)
        if counts.shape[0] != self.n_classes:
            raise ValueError(
                'Length of cls_num_list must match n_classes for ELM weighting.'
            )
        if np.any(counts <= 0):
            raise ValueError('cls_num_list must contain strictly positive counts.')
        weights = np.power(1.0 / counts, self.class_weight_beta)
        weights = weights / weights.mean()  # keep scale comparable to unweighted solve
        self.class_weights_ = weights.astype(DEFAULT_DTYPE, copy=False)

    def _log_init(self) -> None:
        print('=' * 50)
        print('Initializing ELM Classifier')
        print(f'  Hidden units: {self.n_hidden}')
        print(f'  Activation: {self.activation}')
        print(f'  Regularization: {self.reg}')
        print(f'  Classes: {self.n_classes}')
        reweight = 'enabled' if self.cls_num_list is not None else 'disabled'
        print(
            f'  Class reweighting: {reweight}'
            f' (beta={self.class_weight_beta})'
        )
        print(f'  Orthogonalize hidden weights: {self.orthogonalize}')
        print('=' * 50, '\n')

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _rng(self) -> np.random.Generator:
        return np.random.default_rng(self.random_state)

    def _initialise_hidden(self, input_dim: int) -> None:
        rng = self._rng()
        W = rng.uniform(-1.0, 1.0, size=(input_dim, self.n_hidden)).astype(DEFAULT_DTYPE)
        if self.orthogonalize and self.n_hidden <= input_dim:
            # QR orthogonalisation keeps hidden activations well-scaled.
            Q, _ = np.linalg.qr(W)
            W = Q.astype(DEFAULT_DTYPE, copy=False)
        b = rng.uniform(-0.5, 0.5, size=(self.n_hidden,)).astype(DEFAULT_DTYPE)
        self.W_in_ = W
        self.b_in_ = b

    def _hidden_activations(self, X: np.ndarray) -> np.ndarray:
        return ACTIVATIONS[self.activation](X @ self.W_in_ + self.b_in_)

    def _encode_targets(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y)
        if y.ndim == 1:
            Y = np.eye(self.n_classes, dtype=DEFAULT_DTYPE)[y.astype(np.int64)]
        else:
            if y.shape[1] != self.n_classes:
                raise ValueError(
                    'One-hot targets must have the same second dimension as n_classes.'
                )
            Y = y.astype(DEFAULT_DTYPE, copy=False)
        return Y

    def _sample_weights(self, y_labels: np.ndarray) -> Optional[np.ndarray]:
        if self.cls_num_list is None:
            return None
        weights = self.class_weights_[y_labels.astype(np.int64)]
        return weights.astype(DEFAULT_DTYPE, copy=False)

    def _ridge_solve(self, H: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Solve ``min ||H beta - Y||^2 + reg ||beta||^2`` in closed form."""
        n, m = H.shape
        print(f'  Solving ELM ridge regression on H of shape {H.shape}')
        start = time.time()
        reg_eye_n = self.reg * np.eye(n, dtype=DEFAULT_DTYPE)
        reg_eye_m = self.reg * np.eye(m, dtype=DEFAULT_DTYPE)
        try:
            if n < m:
                # Dual form: beta = H^T (H H^T + reg I)^-1 Y
                A = H @ H.T + reg_eye_n
                L = cholesky(A, lower=True)
                z = solve_triangular(L, Y, lower=True)
                z = solve_triangular(L.T, z, lower=False)
                beta = H.T @ z
            else:
                # Primal form: beta = (H^T H + reg I)^-1 H^T Y
                A = H.T @ H + reg_eye_m
                L = cholesky(A, lower=True)
                rhs = H.T @ Y
                z = solve_triangular(L, rhs, lower=True)
                beta = solve_triangular(L.T, z, lower=False)
        except np.linalg.LinAlgError:
            print('  Cholesky failed, falling back to SVD pseudoinverse.')
            U, s, Vt = np.linalg.svd(H, full_matrices=False)
            s_inv = s / (s ** 2 + self.reg)
            beta = Vt.T @ (s_inv[:, None] * (U.T @ Y))
        print(f'  ELM solve complete in {time.time() - start:.3f}s')
        return beta.astype(DEFAULT_DTYPE, copy=False)

    # ------------------------------------------------------------------
    # sklearn-style interface
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ELM':
        X = check_array(X, ensure_2d=True, dtype=DEFAULT_DTYPE)
        y_labels = np.asarray(y).reshape(-1)
        if X.shape[0] != y_labels.shape[0]:
            raise ValueError('X and y must have the same number of samples.')

        if self.is_fitted:
            print('  ELM already fitted. Refitting with new data.')

        self._initialise_hidden(X.shape[1])
        H = self._hidden_activations(X)
        Y = self._encode_targets(y_labels)

        # Optional sample reweighting: scale rows of H and Y by sqrt(w_i).
        sample_weights = self._sample_weights(y_labels)
        if sample_weights is not None:
            sqrt_w = np.sqrt(sample_weights).reshape(-1, 1)
            H = H * sqrt_w
            Y = Y * sqrt_w

        self.beta_ = self._ridge_solve(H, Y)
        self.is_fitted = True
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError('ELM is not fitted. Call fit() first.')
        X = check_array(X, ensure_2d=True, dtype=DEFAULT_DTYPE)
        H = self._hidden_activations(X)
        return H @ self.beta_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        scores = scores - scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.decision_function(X), axis=1)

    def reset(self) -> None:
        self.W_in_ = None
        self.b_in_ = None
        self.beta_ = None
        self.is_fitted = False


__all__ = ['ELM', 'ACTIVATIONS']
