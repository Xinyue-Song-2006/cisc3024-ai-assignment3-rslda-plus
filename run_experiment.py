"""Reproduce the RSLDA+ handwritten-digit comparison using NumPy only."""

from __future__ import annotations

import csv
import io
import json
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from src.rslda_plus import RSLDAPlus


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
UCI_ZIP = (
    "https://archive.ics.uci.edu/static/public/80/"
    "optical+recognition+of+handwritten+digits.zip"
)
SEED = 3024
N_COMPONENTS = 9
K_NEIGHBORS = 5
PARAMETER_GRID = [
    {"lambda1": 1e-3, "lambda2": 1e-5, "lambda3": 1e-5},
    {"lambda1": 1e-2, "lambda2": 1e-5, "lambda3": 1e-5},
    {"lambda1": 1e-2, "lambda2": 1e-4, "lambda3": 1e-4},
    {"lambda1": 1e-2, "lambda2": 1e-3, "lambda3": 1e-3},
]


def ensure_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    DATA.mkdir(exist_ok=True)
    train_path = DATA / "optdigits.tra"
    test_path = DATA / "optdigits.tes"
    if not (train_path.exists() and test_path.exists()):
        request = urllib.request.Request(UCI_ZIP, headers={"User-Agent": "AI-assignment3/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            archive_bytes = response.read()
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            for name, target in (("optdigits.tra", train_path), ("optdigits.tes", test_path)):
                target.write_bytes(archive.read(name))

    def read_table(path: Path) -> tuple[np.ndarray, np.ndarray]:
        data = np.loadtxt(path, delimiter=",")
        return data[:, :-1].astype(float) / 16.0, data[:, -1].astype(int)

    x_train, y_train = read_table(train_path)
    x_test, y_test = read_table(test_path)
    return x_train, y_train, x_test, y_test


def stratified_split(y: np.ndarray, validation_fraction: float, seed: int):
    rng = np.random.default_rng(seed)
    train_idx, validation_idx = [], []
    for label in np.unique(y):
        indices = np.flatnonzero(y == label)
        rng.shuffle(indices)
        n_validation = max(1, int(round(len(indices) * validation_fraction)))
        validation_idx.extend(indices[:n_validation])
        train_idx.extend(indices[n_validation:])
    return np.array(sorted(train_idx)), np.array(sorted(validation_idx))


def standardize_fit(x_train: np.ndarray):
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale < 1e-12] = 1.0
    return mean, scale


def standardize(x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (x - mean) / scale


def scatter_matrices(x: np.ndarray, y: np.ndarray):
    overall = x.mean(axis=0)
    p = x.shape[1]
    sw = np.zeros((p, p))
    sb = np.zeros((p, p))
    for label in np.unique(y):
        xi = x[y == label]
        mu = xi.mean(axis=0)
        centered = xi - mu
        sw += centered.T @ centered / len(xi)
        delta = mu - overall
        sb += len(xi) * np.outer(delta, delta)
    return sw, sb


def pca_projection(x: np.ndarray, n_components: int):
    mean = x.mean(axis=0)
    centered = x - mean
    cov = centered.T @ centered
    values, vectors = np.linalg.eigh((cov + cov.T) / 2)
    return mean, vectors[:, np.argsort(values)[::-1][:n_components]]


def lda_projection(x: np.ndarray, y: np.ndarray, n_components: int):
    sw, sb = scatter_matrices(x, y)
    ridge = 1e-4 * max(float(np.trace(sw) / sw.shape[0]), 1e-8)
    chol = np.linalg.cholesky(sw + ridge * np.eye(sw.shape[0]))
    inv_chol = np.linalg.solve(chol, np.eye(sw.shape[0]))
    whitened_between = inv_chol @ sb @ inv_chol.T
    values, vectors = np.linalg.eigh((whitened_between + whitened_between.T) / 2)
    directions = np.linalg.solve(chol.T, vectors[:, np.argsort(values)[::-1][:n_components]])
    mean = x.mean(axis=0)
    return mean, directions


def knn_predict(x_train: np.ndarray, y_train: np.ndarray, x_query: np.ndarray, k: int):
    predictions = np.empty(len(x_query), dtype=int)
    train_norms = np.sum(x_train * x_train, axis=1)
    for start in range(0, len(x_query), 256):
        query = x_query[start : start + 256]
        distances = (
            np.sum(query * query, axis=1)[:, None]
            + train_norms[None, :]
            - 2 * query @ x_train.T
        )
        nearest = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
        for row, neighbors in enumerate(nearest):
            predictions[start + row] = np.bincount(y_train[neighbors], minlength=10).argmax()
    return predictions


def macro_scores(y_true: np.ndarray, y_pred: np.ndarray):
    classes = np.unique(y_true)
    precisions, recalls, f1s = [], [], []
    for label in classes:
        tp = int(np.sum((y_true == label) & (y_pred == label)))
        fp = int(np.sum((y_true != label) & (y_pred == label)))
        fn = int(np.sum((y_true == label) & (y_pred != label)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
    }


def salt_and_pepper(x: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    """Replace a fixed share of normalized image features with black or white."""
    rng = np.random.default_rng(seed)
    corrupted = x.copy()
    mask = rng.random(corrupted.shape) < fraction
    values = (rng.random(corrupted.shape) >= 0.5).astype(float)
    corrupted[mask] = values[mask]
    return corrupted


def fit_rsl(model_x: np.ndarray, model_y: np.ndarray, params: dict[str, float]):
    return RSLDAPlus(
        n_components=N_COMPONENTS,
        beta=0.1,
        rho=1.1,
        max_iter=200,
        tol=1e-7,
        **params,
    ).fit(model_x, model_y)


def main():
    RESULTS.mkdir(exist_ok=True)
    x_train_raw, y_train, x_test_raw, y_test = ensure_data()
    x_test_noisy_raw = salt_and_pepper(x_test_raw, fraction=0.10, seed=SEED + 1)
    train_idx, validation_idx = stratified_split(y_train, 0.20, SEED)
    x_fit_raw = x_train_raw[train_idx]
    y_fit = y_train[train_idx]
    x_validation_raw = x_train_raw[validation_idx]
    y_validation = y_train[validation_idx]

    val_mean, val_scale = standardize_fit(x_fit_raw)
    x_fit = standardize(x_fit_raw, val_mean, val_scale)
    x_validation = standardize(x_validation_raw, val_mean, val_scale)
    validation_runs = []
    best_params = None
    best_score = -1.0
    for params in PARAMETER_GRID:
        started = time.perf_counter()
        model = fit_rsl(x_fit, y_fit, params)
        prediction = knn_predict(
            model.transform(x_fit), y_fit, model.transform(x_validation), K_NEIGHBORS
        )
        score = float(np.mean(prediction == y_validation))
        validation_runs.append(
            {
                **params,
                "validation_accuracy": score,
                "iterations": model.n_iter_,
                "nonzero_features": model.n_nonzero_features_,
                "seconds": round(time.perf_counter() - started, 3),
            }
        )
        if score > best_score:
            best_score = score
            best_params = params

    # Refit normalization and models on the complete official training set.
    mean, scale = standardize_fit(x_train_raw)
    x_train = standardize(x_train_raw, mean, scale)
    x_test = standardize(x_test_raw, mean, scale)
    x_test_noisy = standardize(x_test_noisy_raw, mean, scale)

    results = []

    def evaluate(method: str, dimensions: int, train_features: np.ndarray,
                 clean_features: np.ndarray, noisy_features: np.ndarray) -> None:
        for condition, query in (("Clean test", clean_features),
                                 ("10% salt-and-pepper test", noisy_features)):
            prediction = knn_predict(train_features, y_train, query, K_NEIGHBORS)
            results.append({
                "method": method,
                "condition": condition,
                "dimensions": dimensions,
                **macro_scores(y_test, prediction),
            })

    evaluate("Raw pixels + 5-NN", 64, x_train, x_test, x_test_noisy)

    pca_mean, pca_components = pca_projection(x_train, N_COMPONENTS)
    pca_train = (x_train - pca_mean) @ pca_components
    pca_test = (x_test - pca_mean) @ pca_components
    pca_test_noisy = (x_test_noisy - pca_mean) @ pca_components
    evaluate("PCA + 5-NN", N_COMPONENTS, pca_train, pca_test, pca_test_noisy)

    lda_mean, lda_components = lda_projection(x_train, y_train, N_COMPONENTS)
    lda_train = (x_train - lda_mean) @ lda_components
    lda_test = (x_test - lda_mean) @ lda_components
    lda_test_noisy = (x_test_noisy - lda_mean) @ lda_components
    evaluate("Fisher LDA + 5-NN", N_COMPONENTS, lda_train, lda_test, lda_test_noisy)

    rsl = fit_rsl(x_train, y_train, best_params)
    rsl_train = rsl.transform(x_train)
    rsl_test = rsl.transform(x_test)
    rsl_test_noisy = rsl.transform(x_test_noisy)
    evaluate("RSLDA+ + 5-NN", N_COMPONENTS, rsl_train, rsl_test, rsl_test_noisy)

    with (RESULTS / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    summary = {
        "dataset": "UCI Optical Recognition of Handwritten Digits",
        "dataset_doi": "10.24432/C50P49",
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "features": int(x_train.shape[1]),
        "classes": int(len(np.unique(y_train))),
        "seed": SEED,
        "n_components": N_COMPONENTS,
        "classifier": f"Euclidean {K_NEIGHBORS}-nearest neighbors",
        "noise_test": "10% of normalized test pixel features replaced by random black or white values",
        "validation_split": "stratified 80/20 from official training split",
        "validation_parameter_search": validation_runs,
        "selected_rsl_parameters": best_params,
        "rsl_final_fit_iterations": rsl.n_iter_,
        "rsl_converged": rsl.converged_,
        "rsl_final_primal_residual1_inf": rsl.primal_residual1_inf_,
        "rsl_final_primal_residual2_inf": rsl.primal_residual2_inf_,
        "rsl_nonzero_features": rsl.n_nonzero_features_,
        "results": results,
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
