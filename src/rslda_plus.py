"""NumPy reimplementation of the RSLDA+ alternating optimization.

The update structure follows the authors' released MATLAB reference code.
This file is an AI-produced educational reimplementation, not the authors'
official implementation.
"""

from __future__ import annotations

import numpy as np


class RSLDAPlus:
    """Robust and sparse LDA projection for samples stored row-wise.

    Parameters mirror the reference implementation. The projection dimension
    may be any value up to the input feature count; the experiment uses nine
    dimensions to match the usual ten-class Fisher-LDA baseline.
    """

    def __init__(
        self,
        n_components: int,
        lambda1: float = 1e-5,
        lambda2: float = 1e-5,
        lambda3: float = 1e-5,
        beta: float = 0.1,
        rho: float = 1.1,
        max_iter: int = 60,
        tol: float = 1e-6,
        regu: float = 1e-5,
    ) -> None:
        self.n_components = n_components
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.beta = beta
        self.rho = rho
        self.max_iter = max_iter
        self.tol = tol
        self.regu = regu
        self.n_iter_ = 0
        self.objective_ = []
        self.n_nonzero_features_ = 0
        self.converged_ = False
        self.primal_residual1_inf_ = np.inf
        self.primal_residual2_inf_ = np.inf

    @staticmethod
    def _scatter(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return within-class and between-class scatter matrices."""
        mean_all = x.mean(axis=0)
        n_features = x.shape[1]
        sw = np.zeros((n_features, n_features), dtype=float)
        sb = np.zeros_like(sw)
        for label in np.unique(y):
            xi = x[y == label]
            mean_i = xi.mean(axis=0)
            centered = xi - mean_i
            sw += centered.T @ centered / max(len(xi), 1)
            delta = mean_i - mean_all
            sb += len(xi) * np.outer(delta, delta)
        return sw, sb

    @staticmethod
    def _solve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        try:
            return np.linalg.solve(a, b)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(a, b, rcond=1e-10)[0]

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RSLDAPlus":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y)
        if x.ndim != 2 or y.ndim != 1 or len(x) != len(y):
            raise ValueError("x must be 2-D, y 1-D, and both must have equal rows")
        if not np.isfinite(x).all():
            raise ValueError("x contains non-finite values")

        self.mean_ = x.mean(axis=0)
        xc = (x - self.mean_).T  # features x samples, matching the paper code
        n_features, n_samples = xc.shape
        labels = np.unique(y)
        if not 1 <= self.n_components <= n_features:
            raise ValueError("n_components must be in [1, number of features]")
        d = self.n_components
        sw, sb = self._scatter(x, y)
        sw_minus_sb = sw - self.regu * sb

        # PCA eigenvectors initialize P, as in the released MATLAB code.
        covariance = xc @ xc.T
        eigenvalues, eigenvectors = np.linalg.eigh((covariance + covariance.T) / 2)
        order = np.argsort(eigenvalues)[::-1][:d]
        p_init = eigenvectors[:, order]
        p = p_init.copy()
        q = np.ones((n_features, d), dtype=float)
        e = np.zeros_like(xc)
        f = np.zeros_like(q)
        a1 = np.zeros_like(xc)
        a2 = np.zeros_like(q)
        beta = self.beta
        identity = np.eye(n_features)

        m = xc - e + a1 / beta
        q1 = 2 * sw_minus_sb + beta * (xc @ xc.T) + beta * identity
        q2 = beta * xc @ m.T @ p_init
        q = self._solve(q1, q2)

        previous_objective = None
        for iteration in range(1, self.max_iter + 1):
            if iteration == 1:
                p = p_init.copy()
            else:
                m = xc - e + a1 / beta
                cross = m @ xc.T @ q
                u, _, vt = np.linalg.svd(cross, full_matrices=False)
                p = u @ vt

            if iteration > 1:
                m = xc - e + a1 / beta
                q1 = 2 * sw_minus_sb + beta * (xc @ xc.T) + beta * identity
                q2 = beta * xc @ m.T @ p
                q = self._solve(q1, q2)
                row_norms = np.linalg.norm(q, axis=1)
                # Hard threshold from the authors' reference implementation.
                q[row_norms <= np.sqrt(self.lambda1 / 0.1736), :] = 0.0

            e0 = xc - p @ q.T @ xc + a1 / beta
            e = e0 * (np.abs(e0) > np.sqrt(self.lambda3 / beta))
            f0 = q - a2 / beta
            f = f0 * (np.abs(f0) > np.sqrt(self.lambda2 / beta))

            residual1 = xc - p @ q.T @ xc - e
            residual2 = q - f
            self.primal_residual1_inf_ = float(np.max(np.abs(residual1)))
            self.primal_residual2_inf_ = float(np.max(np.abs(residual2)))
            a1 += beta * residual1
            a2 += beta * (f - q)
            beta = min(self.rho * beta, 1e5)

            support_by_component = np.count_nonzero(q, axis=0)
            objective = float(
                np.trace(q.T @ sw_minus_sb @ q)
                + self.lambda1 * np.linalg.norm(support_by_component)
                + self.lambda2 * np.count_nonzero(f)
                + self.lambda3 * np.count_nonzero(e)
            )
            self.objective_.append(objective)
            self.n_iter_ = iteration
            objective_change = (
                np.inf if previous_objective is None else abs(objective - previous_objective)
            )
            previous_objective = objective
            if (
                iteration > 2
                and self.primal_residual1_inf_ < self.tol
                and self.primal_residual2_inf_ < self.tol
                and objective_change < 1e-5
            ):
                self.converged_ = True
                break

        self.components_ = q
        self.n_nonzero_features_ = int(np.count_nonzero(np.linalg.norm(q, axis=1)))
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if not hasattr(self, "components_"):
            raise RuntimeError("fit must be called before transform")
        return (np.asarray(x, dtype=float) - self.mean_) @ self.components_

