"""Configuration for the 33-dimensional fused TabNet-XGBoost model."""

from __future__ import annotations


RANDOM_SEED = 42

TABNET_SEED = 45
MAX_EPOCHS = 500
TABNET_PATIENCE = 200

FEATURE_NAMES = [
    "mean", "std", "rms", "peak", "crest", "skewness", "kurtosis",
    "impulse", "shape_factor", "amplitude_mean", "amplitude_max",
    "dominant_freq", "mean_freq", "median_freq", "spectral_entropy",
    "spectral_energy_mean", "band_5_15_energy_ratio",
]

# Keep all 17 raw features; XGBoost receives 17 raw + 16 TabNet features.
REMOVED_FEATURE_NAMES = []

TABNET_PARAMS = {
    "n_d": 16,
    "n_a": 16,
    "n_steps": 5,
    "gamma": 1.5,
    "lambda_sparse": 1e-2,
    "optimizer": "Adam",
    "learning_rate": 1e-2,
    "weight_decay": 5e-4,
    "batch_size": 256,
    "virtual_batch_size": 64,
    "device_name": "cpu",
}

XGBOOST_PARAMS = {
    "n_estimators": 400,
    "max_depth": 5,
    "min_child_weight": 2,
    "learning_rate": 0.02,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "gamma": 0.05,
    "reg_alpha": 0.02,
    "reg_lambda": 2.0,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "early_stopping_rounds": 30,
    "n_jobs": -1,
}

# Selected on the validation set and used by deploy_system.py.
DECISION_THRESHOLD = 0.39780643582344055
