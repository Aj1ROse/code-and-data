"""Train and export the 33-dimensional fused TabNet-XGBoost classifier.

The XGBoost classifier receives 17 standardized engineered features followed
by the 16-dimensional representation captured from TabNet's final mapping.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from xgboost import XGBClassifier

import compare
import xgboost_tabnet_config as config


ROOT_DIR = Path(__file__).resolve().parent
RAW_FEATURE_COUNT = len(config.FEATURE_NAMES)
TABNET_FEATURE_COUNT = config.TABNET_PARAMS["n_d"]
FUSION_FEATURE_COUNT = RAW_FEATURE_COUNT + TABNET_FEATURE_COUNT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a 33-dimensional fused TabNet-XGBoost model"
    )
    parser.add_argument("--noise-dir", type=Path, default=ROOT_DIR / "data" / "噪声特征值")
    parser.add_argument("--debris-dir", type=Path, default=ROOT_DIR / "data" / "泥石流特征值")
    parser.add_argument("--out-dir", type=Path, default=ROOT_DIR / "model")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_tabnet(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
) -> TabNetClassifier:
    params = config.TABNET_PARAMS
    model = TabNetClassifier(
        n_d=params["n_d"],
        n_a=params["n_a"],
        n_steps=params["n_steps"],
        gamma=params["gamma"],
        lambda_sparse=params["lambda_sparse"],
        optimizer_fn=torch.optim.Adam,
        optimizer_params={
            "lr": params["learning_rate"],
            "weight_decay": params["weight_decay"],
        },
        verbose=0,
        seed=config.TABNET_SEED,
        device_name=params["device_name"],
    )
    model.fit(
        X_train=X_train,
        y_train=y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        eval_name=["train", "valid"],
        eval_metric=["auc", "accuracy"],
        max_epochs=config.MAX_EPOCHS,
        patience=config.TABNET_PATIENCE,
        batch_size=params["batch_size"],
        virtual_batch_size=params["virtual_batch_size"],
    )
    return model


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
) -> XGBClassifier:
    params = dict(config.XGBOOST_PARAMS)
    early_stopping_rounds = params.pop("early_stopping_rounds")
    model = XGBClassifier(
        random_state=config.RANDOM_SEED,
        early_stopping_rounds=early_stopping_rounds,
        **params,
    )
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    return model


def main() -> None:
    args = parse_args()
    set_seed(config.RANDOM_SEED)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    X, y = compare.load_data(str(args.noise_dir), str(args.debris_dir))
    if X.shape[1] != RAW_FEATURE_COUNT:
        raise ValueError(f"Expected {RAW_FEATURE_COUNT} raw features, got {X.shape[1]}")

    X_train, X_valid, X_test, y_train, y_valid, y_test = compare.split_data(
        X, y, config.RANDOM_SEED
    )

    # Do not remove median_freq: the requested fusion representation is 17 + 16.
    feature_mask = np.ones(RAW_FEATURE_COUNT, dtype=bool)
    scaler = compare.StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    X_test_scaled = scaler.transform(X_test)

    print("Training TabNet encoder on all 17 raw features...")
    tabnet = train_tabnet(X_train_scaled, y_train, X_valid_scaled, y_valid)
    deep_train = compare.extract_tabnet_features(tabnet, X_train_scaled)
    deep_valid = compare.extract_tabnet_features(tabnet, X_valid_scaled)
    deep_test = compare.extract_tabnet_features(tabnet, X_test_scaled)
    if deep_train.shape[1] != TABNET_FEATURE_COUNT:
        raise RuntimeError(
            f"Expected {TABNET_FEATURE_COUNT} TabNet features, got {deep_train.shape[1]}"
        )

    fusion_train = np.hstack([X_train_scaled, deep_train])
    fusion_valid = np.hstack([X_valid_scaled, deep_valid])
    fusion_test = np.hstack([X_test_scaled, deep_test])
    if fusion_train.shape[1] != FUSION_FEATURE_COUNT:
        raise RuntimeError(
            f"Expected {FUSION_FEATURE_COUNT} fusion features, got {fusion_train.shape[1]}"
        )

    print("Selecting XGBoost parameters for 33-dimensional fused features...")
    xgb_model, selected_xgb_config, threshold, validation_metrics = compare.choose_xgb(
        fusion_train,
        y_train,
        fusion_valid,
        y_valid,
        config.RANDOM_SEED,
        "TabNet-XGBoost-fusion",
    )
    test_score = xgb_model.predict_proba(fusion_test)[:, 1]
    test_metrics = compare.evaluate_with_threshold(y_test, test_score, threshold)

    joblib.dump(scaler, args.out_dir / "raw_feature_scaler.pkl")
    np.save(args.out_dir / "feature_mask.npy", feature_mask)
    tabnet.save_model(str(args.out_dir / "tabnet_encoder"))
    joblib.dump(xgb_model, args.out_dir / "xgboost_tabnet.pkl")

    saved_config = {
        "seed": config.RANDOM_SEED,
        "tabnet_seed": config.TABNET_SEED,
        "max_epochs": config.MAX_EPOCHS,
        "tabnet_params": config.TABNET_PARAMS,
        "tabnet_patience": config.TABNET_PATIENCE,
        "xgboost_params": {
            "n_estimators": compare.N_ESTIMATORS,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "early_stopping_rounds": 30,
            "n_jobs": -1,
            **selected_xgb_config,
        },
        "input_representation": "hybrid",
        "raw_feature_names": config.FEATURE_NAMES,
        "kept_feature_names": config.FEATURE_NAMES,
        "removed_feature_names": [],
        "raw_feature_count": RAW_FEATURE_COUNT,
        "tabnet_feature_count": TABNET_FEATURE_COUNT,
        "classifier_input_dim": FUSION_FEATURE_COUNT,
        "decision_threshold": float(threshold),
        "xgboost": {
            "input_representation": "hybrid",
            "threshold": float(threshold),
            "classifier_input_dim": FUSION_FEATURE_COUNT,
        },
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
    }
    for filename in ("experiment_config.json", "xgboost_tabnet_config.json"):
        with (args.out_dir / filename).open("w", encoding="utf-8") as file:
            json.dump(saved_config, file, ensure_ascii=False, indent=2, default=float)

    print("\nTest metrics:")
    for name, value in test_metrics.items():
        print(f"{name}: {value:.6f}")
    print("Classifier input: 33 dimensions (17 raw + 16 TabNet)")
    print(f"Saved model to: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
