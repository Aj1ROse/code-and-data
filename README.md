# TabNet-XGBoost Mudflow Signal Classification

This directory contains the complete workflow for mudflow-signal processing, feature extraction, model training, model comparison, VMD parameter optimization, and deployment inference.

## Directory layout

```text
code and data/
├── data/
│   ├── 噪声特征值/                 # Feature CSV files for the noise class
│   └── 泥石流特征值/               # Feature CSV files for the mudflow class
├── main.py                         # Raw-signal preprocessing and 17-feature extraction
├── VMD.py                          # Variational Mode Decomposition implementation
├── deploy_system.py                # Deployment inference entry point
├── compare.py                      # Four-model training and comparison program
├── train_tabnet_xgboost_fusion.py  # 33-dimensional fused TabNet-XGBoost training program
├── xgboost_tabnet_config.py        # 33-dimensional model training parameters
├── model/                          # Deployed 33-dimensional model artifacts
├── SSA_Simple_Improved.py          # Levy-flight Sparrow Search Algorithm (LF-SSA)
├── fitness1.py                     # LF-SSA/SSA objective function for VMD parameters
└── sampleEntropy.py                # Sample-entropy calculation used by fitness1.py
```

## Code files

| File | Purpose | Direct dependencies |
| --- | --- | --- |
| `main.py` | Reads a raw TXT/CSV waveform and performs scaling, centering, 0.5-20 Hz band-pass filtering, wavelet denoising, VMD decomposition, IMF selection, and 17-feature extraction. It produces feature data; it does not classify signals. | `VMD.py`, `numpy`, `scipy`, `PyWavelets` |
| `VMD.py` | Implements Variational Mode Decomposition (VMD), used to decompose the denoised waveform into modal components. | `numpy` |
| `deploy_system.py` | Deployment/inference script. It calls `main.py` to extract features, then loads the scaler, feature mask, TabNet encoder, XGBoost classifier, and threshold to output class probabilities and the final label. | `main.py`, model artifacts, `pytorch-tabnet`, `xgboost`, `joblib`, `numpy`, `pandas` |
| `compare.py` | Trains and compares Pure TabNet, TabNet-SVM, TabNet-RF, and TabNet-XGBoost using a stratified 70/30 development/test split. The development split is further divided into 52.5% training and 17.5% validation data for candidate and threshold selection. It saves metrics, figures, configurations, and trained models. The RF and XGBoost tree counts are fixed at 400. | `numpy`, `pandas`, `scikit-learn`, `torch`, `pytorch-tabnet`, `xgboost`, `joblib`, `matplotlib` |
| `train_tabnet_xgboost_fusion.py` | Focused training script for the deployed TabNet-XGBoost pipeline. It retains all 17 raw features, extracts 16 TabNet features, concatenates them into 33-dimensional fusion features, selects XGBoost parameters on the validation set, and exports deployment artifacts. | `compare.py`, `xgboost_tabnet_config.py` and the packages above |
| `xgboost_tabnet_config.py` | Python configuration for the 33-dimensional TabNet-XGBoost setup: seeds, TabNet parameters, and XGBoost search settings. | None |
| `SSA_Simple_Improved.py` | Implements the LF-SSA optimizer used to search VMD penalty factor alpha and mode number K. The function is `SSA_Simple_Improved(pop, Max_iter, lb, ub, dim, fobj, seed=None)`. | `numpy` |
| `fitness1.py` | Defines the decomposition-quality objective function used when SSA/LF-SSA optimizes VMD parameters. | `VMD.py`, `sampleEntropy.py`, `numpy` |
| `sampleEntropy.py` | Provides sample-entropy calculation for the VMD optimization fitness function. | `numpy` |

## Model files

The following files in `model/` form the deployed 33-dimensional TabNet-XGBoost inference model and must remain together:

| File | Purpose |
| --- | --- |
| `raw_feature_scaler.pkl` | Standardizes the raw engineered features before feature selection and TabNet encoding. |
| `feature_mask.npy` | Identifies which engineered features are retained by the trained pipeline. |
| `tabnet_encoder.zip` | The trained TabNet feature encoder. |
| `xgboost_tabnet.pkl` | The XGBoost classifier trained on 17 raw + 16 TabNet features. |
| `experiment_config.json` | Deployment configuration, including the hybrid input representation and selected decision threshold. |
| `xgboost_tabnet_config.json` | Duplicate human-readable model configuration exported with the model. |

