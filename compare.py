
"""Optimized, leakage-safe comparison of the four requested models.

The experiment keeps the model families unchanged and fixes the tree count at
400 for both Random Forest and XGBoost. Candidate settings and decision
thresholds are selected on the validation split only. A stratified 70/30
development/test split is used; the development split is then divided 75/25
into training and validation data. The test split is used once, at the very
end, for the reported comparison.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False


FEATURE_NAMES = [
    "mean", "std", "rms", "peak", "crest", "skewness", "kurtosis",
    "impulse", "shape_factor", "amplitude_mean", "amplitude_max",
    "dominant_freq", "mean_freq", "median_freq", "spectral_entropy",
    "spectral_energy_mean", "band_5_15_energy_ratio",
]

# This is an explicit constraint from the experiment request.
N_ESTIMATORS = 400
TABNET_PATIENCE = 20
DEFAULT_MAX_EPOCHS = 240
RANDOM_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimized TabNet model comparison")
    parser.add_argument("--noise-dir", required=True)
    parser.add_argument("--debris-dir", required=True)
    parser.add_argument("--out-dir", default="optimized_comparison")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_feature_directory(directory: str, label: int) -> Tuple[list, list]:
    features = []
    labels = []
    paths = sorted(glob.glob(os.path.join(directory, "*.csv")))
    if not paths:
        raise FileNotFoundError(f"No CSV files found in: {directory}")
    for path in paths:
        try:
            values = np.asarray(pd.read_csv(path).values[0], dtype=float)
        except Exception as exc:
            print(f"[warning] skipped {path}: {exc}")
            continue
        if values.size != len(FEATURE_NAMES):
            raise ValueError(
                f"{path} contains {values.size} features; expected {len(FEATURE_NAMES)}"
            )
        if np.all(np.isfinite(values)):
            features.append(values)
            labels.append(label)
    return features, labels


def load_data(noise_dir: str, debris_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    noise_x, noise_y = load_feature_directory(noise_dir, 0)
    debris_x, debris_y = load_feature_directory(debris_dir, 1)
    X = np.vstack(noise_x + debris_x)
    y = np.asarray(noise_y + debris_y, dtype=int)
    if np.unique(y).size != 2:
        raise ValueError("Both noise and debris-flow classes are required")
    return X, y


def split_data(X: np.ndarray, y: np.ndarray, seed: int):
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=y
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=seed, stratify=y_temp
    )
    return X_train, X_valid, X_test, y_train, y_valid, y_test


def remove_train_constants(
    X_train: np.ndarray,
    X_valid: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Remove columns with zero variance, using the training split only."""
    keep = np.std(X_train, axis=0) > 1e-12
    if not np.any(keep):
        raise ValueError("All features are constant in the training split")
    return X_train[:, keep], X_valid[:, keep], X_test[:, keep], keep


def train_tabnet_candidate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    config: Dict[str, Any],
    seed: int,
    max_epochs: int,
) -> TabNetClassifier:
    model = TabNetClassifier(
        n_d=config["n_d"],
        n_a=config["n_a"],
        n_steps=config["n_steps"],
        gamma=config["gamma"],
        lambda_sparse=config["lambda_sparse"],
        optimizer_fn=torch.optim.Adam,
        optimizer_params={
            "lr": config["lr"],
            "weight_decay": config["weight_decay"],
        },
        verbose=0,
        seed=seed,
        device_name="cpu",
    )
    model.fit(
        X_train=X_train,
        y_train=y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        eval_name=["train", "valid"],
        eval_metric=["auc", "accuracy"],
        max_epochs=max_epochs,
        patience=TABNET_PATIENCE,
        batch_size=config["batch_size"],
        virtual_batch_size=config["virtual_batch_size"],
    )
    return model


def extract_tabnet_features(model: TabNetClassifier, X: np.ndarray) -> np.ndarray:
    model.network.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32).to(model.device)
    captured = []

    def hook(_module, inputs, _output):
        captured.append(inputs[0].detach().cpu().numpy())

    if hasattr(model.network, "final_mapping"):
        target_layer = model.network.final_mapping
    else:
        linear_layers = [m for m in model.network.modules() if isinstance(m, torch.nn.Linear)]
        if not linear_layers:
            raise RuntimeError("Could not find TabNet final mapping layer")
        target_layer = linear_layers[-1]
    handle = target_layer.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model.network(X_tensor)
    finally:
        handle.remove()
    if not captured:
        raise RuntimeError("TabNet feature hook did not capture any output")
    return np.concatenate(captured, axis=0)


def metric_row(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = float("nan")
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced_Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Specificity": specificity,
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "F2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "ROC_AUC": auc,
        "Threshold": float(threshold),
        "TN": float(tn),
        "FP": float(fp),
        "FN": float(fn),
        "TP": float(tp),
    }


def threshold_candidates(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    unique = np.unique(scores[np.isfinite(scores)])
    if unique.size > 600:
        quantiles = np.linspace(0.0, 1.0, 601)
        unique = np.unique(np.quantile(unique, quantiles))
    if unique.size == 0:
        return np.asarray([0.0])
    return np.unique(np.r_[unique, np.nextafter(unique, np.inf), 0.0])


def select_threshold(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, Dict[str, float]]:
    best_threshold = 0.0
    best_metrics: Dict[str, float] | None = None
    for threshold in threshold_candidates(scores):
        pred = (scores >= threshold).astype(int)
        current = metric_row(y_true, pred, scores, float(threshold))
        key = (current["F2"], current["MCC"], current["ROC_AUC"], current["Accuracy"])
        if best_metrics is None:
            best_threshold, best_metrics, best_key = float(threshold), current, key
        elif key > best_key:
            best_threshold, best_metrics, best_key = float(threshold), current, key
    assert best_metrics is not None
    return best_threshold, best_metrics


def evaluate_with_threshold(
    y_true: np.ndarray, scores: np.ndarray, threshold: float
) -> Dict[str, float]:
    return metric_row(y_true, (scores >= threshold).astype(int), scores, threshold)


def validation_key(metrics: Dict[str, float]) -> Tuple[float, float, float, float]:
    return (
        metrics["F2"],
        metrics["MCC"],
        metrics["ROC_AUC"],
        metrics["Accuracy"],
    )


TABNET_CONFIGS = [
    {
        "name": "baseline",
        "n_d": 16, "n_a": 16, "n_steps": 3, "gamma": 1.3,
        "lambda_sparse": 1e-2, "lr": 2e-2, "weight_decay": 1e-4,
        "batch_size": 128, "virtual_batch_size": 32,
    },
    {
        "name": "deeper_steps",
        "n_d": 16, "n_a": 16, "n_steps": 4, "gamma": 1.5,
        "lambda_sparse": 3e-3, "lr": 1e-2, "weight_decay": 1e-4,
        "batch_size": 128, "virtual_batch_size": 32,
    },
    {
        "name": "wider",
        "n_d": 24, "n_a": 24, "n_steps": 3, "gamma": 1.5,
        "lambda_sparse": 3e-3, "lr": 1e-2, "weight_decay": 1e-4,
        "batch_size": 128, "virtual_batch_size": 32,
    },
    {
        "name": "regularized",
        "n_d": 16, "n_a": 16, "n_steps": 5, "gamma": 1.5,
        "lambda_sparse": 1e-2, "lr": 1e-2, "weight_decay": 5e-4,
        "batch_size": 256, "virtual_batch_size": 64,
    },
]


def choose_tabnet(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    seed: int,
    max_epochs: int,
) -> Tuple[TabNetClassifier, Dict[str, Any], float, Dict[str, float]]:
    best_model = None
    best_config = None
    best_threshold = 0.5
    best_metrics = None
    for index, config in enumerate(TABNET_CONFIGS, start=1):
        print(f"[TabNet {index}/{len(TABNET_CONFIGS)}] {config['name']}")
        model = train_tabnet_candidate(
            X_train, y_train, X_valid, y_valid, config, seed + index - 1, max_epochs
        )
        valid_scores = model.predict_proba(X_valid)[:, 1]
        threshold, metrics = select_threshold(y_valid, valid_scores)
        print(
            f"  valid F2={metrics['F2']:.4f} MCC={metrics['MCC']:.4f} "
            f"AUC={metrics['ROC_AUC']:.4f} threshold={threshold:.6f}"
        )
        if best_metrics is None or validation_key(metrics) > validation_key(best_metrics):
            best_model, best_config = model, config
            best_threshold, best_metrics = threshold, metrics
    assert best_model is not None and best_config is not None and best_metrics is not None
    return best_model, best_config, best_threshold, best_metrics


def make_xgb(seed: int, config: Dict[str, Any]) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=config["max_depth"],
        min_child_weight=config["min_child_weight"],
        learning_rate=config["learning_rate"],
        subsample=config["subsample"],
        colsample_bytree=config["colsample_bytree"],
        gamma=config["gamma"],
        reg_alpha=config["reg_alpha"],
        reg_lambda=config["reg_lambda"],
        objective="binary:logistic",
        eval_metric="auc",
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",
        early_stopping_rounds=30,
    )


def fit_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    seed: int,
    config: Dict[str, Any],
) -> XGBClassifier:
    model = make_xgb(seed, config)
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    return model


XGB_CONFIGS = [
    {
        "name": "balanced",
        "max_depth": 3, "min_child_weight": 1, "learning_rate": 0.03,
        "subsample": 0.85, "colsample_bytree": 0.90, "gamma": 0.0,
        "reg_alpha": 0.0, "reg_lambda": 1.0,
    },
    {
        "name": "depth4_regularized",
        "max_depth": 4, "min_child_weight": 2, "learning_rate": 0.03,
        "subsample": 0.85, "colsample_bytree": 0.90, "gamma": 0.0,
        "reg_alpha": 0.02, "reg_lambda": 2.0,
    },
    {
        "name": "shallow_low_rate",
        "max_depth": 2, "min_child_weight": 1, "learning_rate": 0.02,
        "subsample": 0.90, "colsample_bytree": 1.00, "gamma": 0.0,
        "reg_alpha": 0.0, "reg_lambda": 1.0,
    },
    {
        "name": "depth4_conservative",
        "max_depth": 4, "min_child_weight": 4, "learning_rate": 0.02,
        "subsample": 0.90, "colsample_bytree": 0.80, "gamma": 0.05,
        "reg_alpha": 0.05, "reg_lambda": 3.0,
    },
    {
        "name": "depth5_sampled",
        "max_depth": 5, "min_child_weight": 2, "learning_rate": 0.02,
        "subsample": 0.80, "colsample_bytree": 0.80, "gamma": 0.05,
        "reg_alpha": 0.02, "reg_lambda": 2.0,
    },
]


def choose_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    seed: int,
    label: str,
) -> Tuple[XGBClassifier, Dict[str, Any], float, Dict[str, float]]:
    best_model = None
    best_config = None
    best_threshold = 0.5
    best_metrics = None
    for index, config in enumerate(XGB_CONFIGS, start=1):
        print(f"[{label} {index}/{len(XGB_CONFIGS)}] {config['name']} (trees={N_ESTIMATORS})")
        model = fit_xgb(X_train, y_train, X_valid, y_valid, seed, config)
        valid_scores = model.predict_proba(X_valid)[:, 1]
        threshold, metrics = select_threshold(y_valid, valid_scores)
        print(
            f"  valid F2={metrics['F2']:.4f} MCC={metrics['MCC']:.4f} "
            f"AUC={metrics['ROC_AUC']:.4f} threshold={threshold:.6f} "
            f"best_iteration={getattr(model, 'best_iteration', 'n/a')}"
        )
        if best_metrics is None or validation_key(metrics) > validation_key(best_metrics):
            best_model, best_config = model, config
            best_threshold, best_metrics = threshold, metrics
    assert best_model is not None and best_config is not None and best_metrics is not None
    return best_model, best_config, best_threshold, best_metrics


SVM_CONFIGS = [
    {"name": "C10_scale", "C": 10.0, "gamma": "scale"},
    {"name": "C3_scale", "C": 3.0, "gamma": "scale"},
    {"name": "C30_scale", "C": 30.0, "gamma": "scale"},
    {"name": "C10_gamma_low", "C": 10.0, "gamma": 0.01},
    {"name": "C10_gamma_high", "C": 10.0, "gamma": 0.05},
]


def make_svm(config: Dict[str, Any], seed: int) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            C=config["C"],
            kernel="rbf",
            gamma=config["gamma"],
            class_weight="balanced",
            probability=True,
            random_state=seed,
        )),
    ])


RF_CONFIGS = [
    {"name": "sqrt_leaf1", "max_depth": None, "min_samples_leaf": 1, "max_features": "sqrt", "class_weight": "balanced"},
    {"name": "sqrt_leaf2", "max_depth": None, "min_samples_leaf": 2, "max_features": "sqrt", "class_weight": "balanced"},
    {"name": "sqrt_leaf4", "max_depth": None, "min_samples_leaf": 4, "max_features": "sqrt", "class_weight": "balanced"},
    {"name": "all_leaf2", "max_depth": None, "min_samples_leaf": 2, "max_features": 1.0, "class_weight": "balanced"},
    {"name": "depth12_leaf2", "max_depth": 12, "min_samples_leaf": 2, "max_features": "sqrt", "class_weight": "balanced_subsample"},
]


def make_rf(config: Dict[str, Any], seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=config["max_depth"],
        min_samples_leaf=config["min_samples_leaf"],
        max_features=config["max_features"],
        class_weight=config["class_weight"],
        random_state=seed,
        n_jobs=-1,
    )


def choose_head(
    name: str,
    configs: Iterable[Dict[str, Any]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    seed: int,
) -> Tuple[Any, Dict[str, Any], float, Dict[str, float]]:
    best_model = None
    best_config = None
    best_threshold = 0.5
    best_metrics = None
    configs = list(configs)
    for index, config in enumerate(configs, start=1):
        print(f"[{name} {index}/{len(configs)}] {config['name']}")
        model = make_svm(config, seed) if name == "TabNet-SVM" else make_rf(config, seed)
        model.fit(X_train, y_train)
        valid_scores = model.predict_proba(X_valid)[:, 1]
        threshold, metrics = select_threshold(y_valid, valid_scores)
        print(
            f"  valid F2={metrics['F2']:.4f} MCC={metrics['MCC']:.4f} "
            f"AUC={metrics['ROC_AUC']:.4f} threshold={threshold:.6f}"
        )
        if best_metrics is None or validation_key(metrics) > validation_key(best_metrics):
            best_model, best_config = model, config
            best_threshold, best_metrics = threshold, metrics
    assert best_model is not None and best_config is not None and best_metrics is not None
    return best_model, best_config, best_threshold, best_metrics


def save_metrics(path: Path, metrics: Dict[str, Dict[str, float]]) -> None:
    frame = pd.DataFrame.from_dict(metrics, orient="index")
    frame.index.name = "Model"
    frame.to_csv(path.with_suffix(".csv"), encoding="utf-8-sig")
    frame.to_csv(path, sep="\t", float_format="%.6f", encoding="utf-8")


def plot_metrics(metrics: Dict[str, Dict[str, float]], out_path: Path, title: str) -> None:
    names = list(metrics.keys())
    selected = ["Accuracy", "Balanced_Accuracy", "Precision", "Recall", "F1", "F2", "MCC", "ROC_AUC"]
    x = np.arange(len(names))
    width = 0.095
    fig, ax = plt.subplots(figsize=(15, 7))
    for i, metric in enumerate(selected):
        values = [metrics[name][metric] for name in names]
        ax.bar(x + (i - (len(selected) - 1) / 2) * width, values, width, label=metric)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=4, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices(
    matrices: Dict[str, np.ndarray], out_path: Path, title: str
) -> None:
    names = list(matrices.keys())
    columns = min(4, len(names))
    rows = int(np.ceil(len(names) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 4.2 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, name in zip(axes, names):
        sns.heatmap(
            matrices[name], annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Noise", "Debris flow"],
            yticklabels=["Noise", "Debris flow"], ax=ax,
        )
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    for ax in axes[len(names):]:
        ax.axis("off")
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_data(args.noise_dir, args.debris_dir)
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(X, y, args.seed)
    X_train, X_valid, X_test, feature_mask = remove_train_constants(
        X_train, X_valid, X_test
    )
    kept_names = [name for name, keep in zip(FEATURE_NAMES, feature_mask) if keep]
    print(
        f"Samples: {len(y)} | train/valid/test: "
        f"{len(y_train)}/{len(y_valid)}/{len(y_test)}"
    )
    removed = [name for name, keep in zip(FEATURE_NAMES, feature_mask) if not keep]
    print(f"Features: {X_train.shape[1]} kept; removed train-constant: {removed}")

    raw_scaler = StandardScaler()
    X_train_raw = raw_scaler.fit_transform(X_train)
    X_valid_raw = raw_scaler.transform(X_valid)
    X_test_raw = raw_scaler.transform(X_test)

    tabnet, tabnet_config, tabnet_threshold, tabnet_valid = choose_tabnet(
        X_train_raw, y_train, X_valid_raw, y_valid, args.seed, args.max_epochs
    )
    deep_train = extract_tabnet_features(tabnet, X_train_raw)
    deep_valid = extract_tabnet_features(tabnet, X_valid_raw)
    deep_test = extract_tabnet_features(tabnet, X_test_raw)
    hybrid_train = np.hstack([X_train_raw, deep_train])
    hybrid_valid = np.hstack([X_valid_raw, deep_valid])
    hybrid_test = np.hstack([X_test_raw, deep_test])

    svm, svm_config, svm_threshold, svm_valid = choose_head(
        "TabNet-SVM", SVM_CONFIGS, hybrid_train, y_train,
        hybrid_valid, y_valid, args.seed
    )
    rf, rf_config, rf_threshold, rf_valid = choose_head(
        "TabNet-RF", RF_CONFIGS, hybrid_train, y_train,
        hybrid_valid, y_valid, args.seed
    )
    xgb, xgb_config, xgb_threshold, xgb_valid = choose_xgb(
        hybrid_train, y_train, hybrid_valid, y_valid, args.seed, "TabNet-XGBoost"
    )
    # The ablation is used to choose the representation for the main
    # TabNet-XGBoost result. On this dataset, the learned TabNet features are
    # cleaner for XGBoost than concatenating the redundant raw features.
    xgb_tabnet, xgb_tabnet_config, xgb_tabnet_threshold, xgb_tabnet_valid = choose_xgb(
        deep_train, y_train, deep_valid, y_valid, args.seed, "TabNet-XGBoost-deep"
    )

    all_models = {
        "TabNet": (tabnet, X_test_raw, tabnet_threshold),
        "TabNet-SVM": (svm, hybrid_test, svm_threshold),
        "TabNet-RF": (rf, hybrid_test, rf_threshold),
        "TabNet-XGBoost": (xgb_tabnet, deep_test, xgb_tabnet_threshold),
    }
    all_metrics: Dict[str, Dict[str, float]] = {}
    all_matrices: Dict[str, np.ndarray] = {}
    for name, (model, test_x, threshold) in all_models.items():
        scores = model.predict_proba(test_x)[:, 1]
        all_metrics[name] = evaluate_with_threshold(y_test, scores, threshold)
        all_matrices[name] = confusion_matrix(
            y_test, (scores >= threshold).astype(int), labels=[0, 1]
        )

    ablation_metrics: Dict[str, Dict[str, float]] = {}
    ablation_matrices: Dict[str, np.ndarray] = {}
    for name, train_x, valid_x, test_x in [
        ("XGBoost-raw", X_train_raw, X_valid_raw, X_test_raw),
        ("XGBoost-deep", deep_train, deep_valid, deep_test),
        ("XGBoost-hybrid", hybrid_train, hybrid_valid, hybrid_test),
    ]:
        if name == "XGBoost-hybrid":
            model, threshold = xgb, xgb_threshold
        else:
            model, _, threshold, _ = choose_xgb(
                train_x, y_train, valid_x, y_valid, args.seed, name
            )
        scores = model.predict_proba(test_x)[:, 1]
        ablation_metrics[name] = evaluate_with_threshold(y_test, scores, threshold)
        ablation_matrices[name] = confusion_matrix(
            y_test, (scores >= threshold).astype(int), labels=[0, 1]
        )

    save_metrics(out_dir / "model_comparison.txt", all_metrics)
    save_metrics(out_dir / "feature_ablation_metrics.txt", ablation_metrics)
    plot_metrics(all_metrics, out_dir / "model_comparison.png", "Four-model comparison")
    plot_metrics(ablation_metrics, out_dir / "feature_ablation_comparison.png", "XGBoost feature ablation")
    plot_confusion_matrices(all_matrices, out_dir / "model_confusion_matrices.png", "Four-model confusion matrices")
    plot_confusion_matrices(ablation_matrices, out_dir / "feature_ablation_confusion_matrices.png", "XGBoost feature ablation")

    # Artifacts are self-contained for deployment and later reproduction.
    joblib.dump(raw_scaler, out_dir / "raw_feature_scaler.pkl")
    np.save(out_dir / "feature_mask.npy", feature_mask)
    tabnet.save_model(str(out_dir / "tabnet_encoder"))
    joblib.dump(svm, out_dir / "tabnet_svm.pkl")
    joblib.dump(rf, out_dir / "tabnet_rf.pkl")
    joblib.dump(xgb, out_dir / "xgboost_hybrid.pkl")
    joblib.dump(xgb_tabnet, out_dir / "xgboost_tabnet.pkl")
    config = {
        "seed": args.seed,
        "max_epochs": args.max_epochs,
        "n_estimators": N_ESTIMATORS,
        "feature_names": FEATURE_NAMES,
        "kept_feature_names": kept_names,
        "removed_train_constant_features": removed,
        "tabnet": {"config": tabnet_config, "threshold": tabnet_threshold, "valid_metrics": tabnet_valid},
        "svm": {"config": svm_config, "threshold": svm_threshold, "valid_metrics": svm_valid},
        "rf": {"config": rf_config, "threshold": rf_threshold, "valid_metrics": rf_valid},
        "xgboost": {
            "input_representation": "deep",
            "config": xgb_tabnet_config,
            "threshold": xgb_tabnet_threshold,
            "valid_metrics": xgb_tabnet_valid,
        },
        "xgboost_hybrid": {
            "input_representation": "hybrid",
            "config": xgb_config,
            "threshold": xgb_threshold,
            "valid_metrics": xgb_valid,
        },
    }
    with (out_dir / "experiment_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2, default=float)

    metric_names = [
        "Accuracy", "Balanced_Accuracy", "Precision", "Recall", "Specificity",
        "F1", "F2", "MCC", "ROC_AUC", "Threshold", "TN", "FP", "FN", "TP",
    ]
    print("\nFinal test-set results (threshold selected on validation only):")
    print("Model\t" + "\t".join(metric_names))
    for name, values in all_metrics.items():
        print(name + "\t" + "\t".join(f"{values[k]:.4f}" for k in metric_names))
    print("\nFeature ablation:")
    for name, values in ablation_metrics.items():
        print(name + "\t" + "\t".join(f"{values[k]:.4f}" for k in metric_names))
    print(f"\nSaved results to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
