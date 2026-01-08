"""
Training XGBoost - GRADIENT BOOSTING OTTIMIZZATO per NIDS

PERCHÉ XGBOOST:
- Gradient boosting ottimizzato (veloce + accurato)
- Gestisce bene imbalance (scale_pos_weight)
- Feature importance interpretabile
- Ottimo per dati tabulari come network traffic
- Regularization incorporata (L1/L2)

VANTAGGI per NIDS:
- Veloce su dataset grandi
- Robusto a outlier
- Eccellente accuracy/speed trade-off

Usage (from ANY directory):
    python src/train_xgboost.py --dataset-type original
    python src/train_xgboost.py --dataset-type smote --n-estimators 200
"""

import pandas as pd
import numpy as np
import joblib
import time
import json
import os
import sys
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import pyarrow.parquet as pq

# XGBoost import with error handling
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("❌ XGBoost not installed!")
    print("Install with: pip install xgboost")

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

SCRIPT_PATH = Path(__file__).resolve()
BASE_DIR = SCRIPT_PATH.parent.parent

DATA_DIR = BASE_DIR / "data" / "processed"
CICIOT_DIR = DATA_DIR / "CICIOT23"
SMOTE_DIR = DATA_DIR / "SMOTE"
BORDERLINE_DIR = DATA_DIR / "BorderlineSMOTE"
ADASYN_DIR = DATA_DIR / "ADASYN"
CTGAN_DIR = DATA_DIR / "CTGAN"

MODELS_DIR = BASE_DIR / "models" / "XGBoost"
DOCS_DIR = BASE_DIR / "docs" / "XGBoost"

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def print_header(text):
    print("\n" + "="*80)
    print(text.center(80))
    print("="*80 + "\n")

def print_section(text):
    print("\n" + "-"*80)
    print(text)
    print("-"*80)

def validate_dataset_type(dataset_type):
    """Valida dataset type."""
    valid_types = ['original', 'smote', 'borderline', 'adasyn', 'ctgan']
    if dataset_type not in valid_types:
        raise ValueError(f"Invalid dataset_type: {dataset_type}. Valid: {valid_types}")
    return dataset_type

def get_paths(dataset_type):
    """Ottieni path basati su dataset type."""
    paths = {
        'test': CICIOT_DIR / 'test_processed.parquet',
        'validation': CICIOT_DIR / 'validation_processed.parquet',
        'artifacts': CICIOT_DIR
    }
    
    train_files = {
        'original': CICIOT_DIR / 'train_processed.parquet',
        'smote': SMOTE_DIR / 'train_smote.parquet',
        'borderline': BORDERLINE_DIR / 'train_borderline_smote.parquet',
        'adasyn': ADASYN_DIR / 'train_adasyn.parquet',
        'ctgan': CTGAN_DIR / 'train_ctgan.parquet'
    }
    
    paths['train'] = train_files[dataset_type]
    
    for key, path in paths.items():
        if key != 'artifacts' and not path.exists():
            raise FileNotFoundError(f"{key.upper()} not found: {path}")
    
    return paths

def get_output_paths(dataset_type):
    """Ottieni path output."""
    model_dir = MODELS_DIR
    plots_dir = DOCS_DIR / dataset_type
    
    model_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    return {
        'model': model_dir / f'xgb_model_{dataset_type}.pkl',
        'plots': plots_dir
    }

# =============================================================================
# DATA LOADING
# =============================================================================

def load_parquet_dataset(parquet_path, description="dataset"):
    """Carica dataset parquet."""
    print(f"📂 Loading {description}: {parquet_path}")
    
    if not parquet_path.exists():
        raise FileNotFoundError(f"Not found: {parquet_path}")
    
    df = pq.read_table(parquet_path).to_pandas()
    
    print(f"   ✅ Loaded: {df.shape}")
    print(f"   Memory: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    return df

def load_processed_data(paths):
    """Carica train/test/val."""
    print_header("LOADING DATASETS (Parquet)")
    
    df_train = load_parquet_dataset(paths['train'], "TRAIN")
    df_test = load_parquet_dataset(paths['test'], "TEST")
    df_val = load_parquet_dataset(paths['validation'], "VALIDATION")
    
    feature_cols = [col for col in df_train.columns 
                   if col not in ['y_macro_encoded', 'y_specific']]
    
    X_train = df_train[feature_cols].values
    y_train = df_train['y_macro_encoded'].values
    
    X_test = df_test[feature_cols].values
    y_test = df_test['y_macro_encoded'].values
    
    X_val = df_val[feature_cols].values
    y_val = df_val['y_macro_encoded'].values
    
    print(f"\n📊 Dataset Summary:")
    print(f"   Features: {len(feature_cols)}")
    print(f"   Classes: {len(np.unique(y_train))}")
    print(f"   Train: {len(X_train):,} samples")
    print(f"   Test: {len(X_test):,} samples")
    print(f"   Val: {len(X_val):,} samples")
    
    print(f"\n📊 Train Class Distribution:")
    unique, counts = np.unique(y_train, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"   Class {cls}: {count:>8,} ({count/len(y_train)*100:>5.2f}%)")
    
    return X_train, X_test, X_val, y_train, y_test, y_val, feature_cols

def load_label_encoder(artifacts_path):
    """Carica label encoder."""
    encoder_path = artifacts_path / 'label_encoder.pkl'
    
    if not encoder_path.exists():
        raise FileNotFoundError(f"Encoder not found: {encoder_path}")
    
    label_encoder = joblib.load(encoder_path)
    print(f"\n✅ Label encoder: {label_encoder.classes_}")
    
    return label_encoder

# =============================================================================
# TRAINING
# =============================================================================

def train_xgboost(X_train, y_train, n_estimators=100, max_depth=6, learning_rate=0.1):
    """
    Train XGBoost classifier.
    
    XGBoost HYPERPARAMETERS:
    - n_estimators: numero alberi (100-300)
    - max_depth: profondità alberi (3-10, default 6)
    - learning_rate: step size (0.01-0.3, default 0.1)
    - subsample: frazione samples per albero (0.8)
    - colsample_bytree: frazione features per albero (0.8)
    - gamma: min loss reduction (regularization)
    - reg_alpha: L1 regularization
    - reg_lambda: L2 regularization
    
    Args:
        X_train: Training features
        y_train: Training labels
        n_estimators: Number of boosting rounds
        max_depth: Max tree depth
        learning_rate: Learning rate (eta)
    
    Returns:
        Trained XGBoost model
    """
    print_header("TRAINING XGBOOST")
    
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost not installed!")
    
    print(f"Hyperparameters:")
    print(f"  n_estimators: {n_estimators}")
    print(f"  max_depth: {max_depth}")
    print(f"  learning_rate: {learning_rate}")
    print(f"  subsample: 0.8")
    print(f"  colsample_bytree: 0.8")
    print(f"  objective: multi:softmax")
    print(f"  eval_metric: mlogloss")
    print(f"  n_jobs: -1 (all cores)")
    
    start_time = time.time()
    
    # Calcola class weights per imbalance
    class_counts = np.bincount(y_train)
    n_samples = len(y_train)
    n_classes = len(class_counts)
    
    # Compute scale_pos_weight (per multi-class usa max_delta_step)
    print(f"\nClass balance analysis:")
    for cls_idx, count in enumerate(class_counts):
        pct = count / n_samples * 100
        print(f"  Class {cls_idx}: {count:>8,} ({pct:>5.2f}%)")
    
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softmax',  # Multi-class classification
        eval_metric='mlogloss',
        num_class=n_classes,
        max_delta_step=1,  # Aiuta con imbalance
        gamma=0.1,  # Min loss reduction
        reg_alpha=0.1,  # L1 regularization
        reg_lambda=1.0,  # L2 regularization
        n_jobs=-1,
        random_state=42,
        verbosity=1
    )
    
    print("\nTraining XGBoost (gradient boosting)...")
    model.fit(X_train, y_train)
    
    elapsed = time.time() - start_time
    print(f"\n✅ Complete in {elapsed:.1f}s ({elapsed/60:.1f} min)!")
    
    return model

# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_model(model, X_train, X_test, X_val, y_train, y_test, y_val,
                   label_encoder, output_paths, dataset_type):
    """Valuta modello."""
    print_header("MODEL EVALUATION")
    
    print("Making predictions...")
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_val_pred = model.predict(X_val)
    
    class_names = label_encoder.classes_
    
    metrics = {}
    for set_name, y_true, y_pred in [
        ('Train', y_train, y_train_pred),
        ('Test', y_test, y_test_pred),
        ('Val', y_val, y_val_pred)
    ]:
        metrics[set_name] = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
    
    print("\n" + "-"*80)
    print(f"RESULTS - Dataset: {dataset_type.upper()}")
    print("-"*80)
    print(f"{'Metric':<15} {'Train':>12} {'Test':>12} {'Val':>12} {'Status':>8}")
    print("-"*80)
    
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        train_val = metrics['Train'][metric]
        test_val = metrics['Test'][metric]
        val_val = metrics['Val'][metric]
        
        status = ""
        if metric == 'accuracy' and test_val >= 0.95:
            status = "✅"
        elif metric == 'precision' and test_val >= 0.90:
            status = "✅"
        elif metric == 'recall' and test_val >= 0.95:
            status = "✅"
        elif metric in ['accuracy', 'recall']:
            status = "❌"
        
        print(f"{metric.capitalize():<15} {train_val:>12.4f} {test_val:>12.4f} "
              f"{val_val:>12.4f} {status:>8}")
    
    print("-"*80)
    
    # Overfitting check
    acc_diff = metrics['Train']['accuracy'] - metrics['Test']['accuracy']
    if acc_diff > 0.05:
        print(f"\n⚠️ Possible overfitting (Train-Test diff: {acc_diff:.4f})")
    else:
        print(f"\n✅ Good generalization (Train-Test diff: {acc_diff:.4f})")
    
    print("\n" + "-"*80)
    print("CLASSIFICATION REPORT (Test Set)")
    print("-"*80)
    print(classification_report(y_test, y_test_pred,
                                target_names=class_names,
                                digits=4, zero_division=0))
    
    cm = confusion_matrix(y_test, y_test_pred)
    
    generate_plots(cm, metrics, class_names, output_paths, dataset_type)
    
    return metrics, cm

def generate_plots(cm, metrics, class_names, output_paths, dataset_type):
    """Genera plots."""
    plots_dir = output_paths['plots']
    
    # 1. Confusion Matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - XGBoost\nDataset: {dataset_type.upper()}',
              fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    cm_path = plots_dir / 'confusion_matrix.png'
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Confusion matrix: {cm_path}")
    plt.close()
    
    # 2. Metrics Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    train_values = [metrics['Train'][k] for k in ['accuracy', 'precision', 'recall', 'f1']]
    test_values = [metrics['Test'][k] for k in ['accuracy', 'precision', 'recall', 'f1']]
    val_values = [metrics['Val'][k] for k in ['accuracy', 'precision', 'recall', 'f1']]
    
    x = np.arange(len(metric_names))
    width = 0.25
    
    ax.bar(x - width, train_values, width, label='Train', alpha=0.8, color='darkorange')
    ax.bar(x, test_values, width, label='Test', alpha=0.8, color='darkred')
    ax.bar(x + width, val_values, width, label='Val', alpha=0.8, color='gold')
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Metrics Comparison - XGBoost\nDataset: {dataset_type.upper()}',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0.95, color='red', linestyle='--', alpha=0.5)
    ax.axhline(y=0.90, color='orange', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    metrics_path = plots_dir / 'metrics_comparison.png'
    plt.savefig(metrics_path, dpi=150, bbox_inches='tight')
    print(f"📊 Metrics comparison: {metrics_path}")
    plt.close()

def plot_feature_importance(model, feature_names, output_paths, dataset_type, n_features=20):
    """Visualizza feature importance."""
    if not feature_names or len(feature_names) == 0:
        return
    
    print_header("FEATURE IMPORTANCE")
    
    # XGBoost feature importance (gain)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:n_features]
    
    print(f"\nTop {min(15, n_features)} Features (by gain):")
    for i in range(min(15, n_features)):
        idx = indices[i]
        feat_name = feature_names[idx] if idx < len(feature_names) else f"Feature_{idx}"
        print(f"{i+1:2d}. {feat_name:45s} {importances[idx]:.6f}")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_features = []
    plot_importances = []
    
    for idx in indices:
        if idx < len(feature_names):
            plot_features.append(feature_names[idx])
            plot_importances.append(importances[idx])
    
    colors = plt.cm.Oranges(np.linspace(0.4, 0.8, len(plot_importances)))
    ax.barh(range(len(plot_importances)), plot_importances[::-1], alpha=0.9, color=colors[::-1])
    ax.set_yticks(range(len(plot_features)))
    ax.set_yticklabels(plot_features[::-1], fontsize=9)
    ax.set_xlabel('Importance (Gain)', fontsize=12)
    ax.set_title(f'Top {n_features} Features - XGBoost\nDataset: {dataset_type.upper()}',
                 fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    feat_path = output_paths['plots'] / 'feature_importance.png'
    plt.savefig(feat_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Feature importance: {feat_path}")
    plt.close()

def save_model(model, output_paths, dataset_type):
    """Salva modello."""
    print_header("SAVING MODEL")
    
    model_path = output_paths['model']
    joblib.dump(model, model_path)
    
    size_mb = model_path.stat().st_size / 1024**2
    print(f"💾 Model: {model_path}")
    print(f"   Size: {size_mb:.2f} MB")

# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Train XGBoost (Gradient Boosting)'
    )
    parser.add_argument('--dataset-type', type=str, required=True,
                        choices=['original', 'smote', 'borderline', 'adasyn', 'ctgan'],
                        help='Dataset type')
    parser.add_argument('--n-estimators', type=int, default=100,
                        help='Number of boosting rounds (default: 100)')
    parser.add_argument('--max-depth', type=int, default=6,
                        help='Max tree depth (default: 6)')
    parser.add_argument('--learning-rate', type=float, default=0.1,
                        help='Learning rate (default: 0.1)')
    
    args = parser.parse_args()
    
    if not XGBOOST_AVAILABLE:
        print("\n❌ XGBoost not available!")
        print("Install with: pip install xgboost")
        sys.exit(1)
    
    print("\n" + "🚀"*40)
    print(f"XGBOOST TRAINING - {args.dataset_type.upper()}".center(80))
    print("🚀"*40)
    
    dataset_type = validate_dataset_type(args.dataset_type)
    data_paths = get_paths(dataset_type)
    output_paths = get_output_paths(dataset_type)
    
    print(f"\n📂 Configuration:")
    print(f"   Dataset: {dataset_type}")
    print(f"   Train: {data_paths['train']}")
    print(f"   Output: {output_paths['model']}")
    
    X_train, X_test, X_val, y_train, y_test, y_val, feature_cols = load_processed_data(data_paths)
    label_encoder = load_label_encoder(data_paths['artifacts'])
    
    model = train_xgboost(X_train, y_train,
                         n_estimators=args.n_estimators,
                         max_depth=args.max_depth,
                         learning_rate=args.learning_rate)
    
    metrics, cm = evaluate_model(
        model, X_train, X_test, X_val,
        y_train, y_test, y_val,
        label_encoder, output_paths, dataset_type
    )
    
    plot_feature_importance(model, feature_cols, output_paths, dataset_type)
    save_model(model, output_paths, dataset_type)
    
    print_header("✅ TRAINING COMPLETE!")
    
    test_metrics = metrics['Test']
    meets_requirements = (
        test_metrics['accuracy'] >= 0.95 and
        test_metrics['precision'] >= 0.90 and
        test_metrics['recall'] >= 0.95
    )
    
    if meets_requirements:
        print("✅ MODEL MEETS ALL REQUIREMENTS!")
    else:
        print("⚠️ Model does not meet all requirements.")
    
    print(f"\n📁 Outputs:")
    print(f"   Model: {output_paths['model']}")
    print(f"   Plots: {output_paths['plots']}/")

if __name__ == '__main__':
    main()