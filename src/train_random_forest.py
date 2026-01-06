"""
Training Random Forest per CICIoT2023 - Multi-Class Classification.
Rifattorizzato per supportare dataset original e SMOTE con path management robusto.

Usage:
    python src/train_random_forest.py --dataset-type original
    python src/train_random_forest.py --dataset-type smote --n-estimators 200
"""

import pandas as pd
import numpy as np
import joblib
import time
import json
import os
import sys
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
CICIOT_DIR = DATA_DIR / "CICIOT23"
SMOTE_DIR = DATA_DIR / "SMOTE"
MODELS_DIR = BASE_DIR / "models" / "RandomForest"
DOCS_DIR = BASE_DIR / "docs" / "RandomForest"


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def print_header(text):
    print("\n" + "="*80)
    print(text.center(80))
    print("="*80 + "\n")


def validate_dataset_type(dataset_type):
    valid_types = ['original', 'smote']
    if dataset_type not in valid_types:
        raise ValueError(f"Invalid dataset_type: {dataset_type}. Must be one of {valid_types}")
    return dataset_type


def get_paths(dataset_type):
    paths = {
        'test': CICIOT_DIR / 'test_processed.pkl',
        'validation': CICIOT_DIR / 'validation_processed.pkl',
        'artifacts': CICIOT_DIR
    }
    
    if dataset_type == 'original':
        paths['train'] = CICIOT_DIR / 'train_processed.pkl'
    elif dataset_type == 'smote':
        paths['train'] = SMOTE_DIR / 'train_smote.pkl'
    
    for key, path in paths.items():
        if key != 'artifacts' and not path.exists():
            raise FileNotFoundError(f"{key.capitalize()} file not found: {path}")
    
    return paths


def get_output_paths(dataset_type):
    model_dir = MODELS_DIR
    plots_dir = DOCS_DIR / dataset_type
    
    model_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    return {
        'model': model_dir / f'rf_model_{dataset_type}.pkl',
        'plots': plots_dir
    }


# =============================================================================
# DATA LOADING
# =============================================================================

def load_processed_data(paths):
    print_header("LOADING PROCESSED DATA")
    
    print(f"📂 Loading Train: {paths['train']}")
    df_train = pd.read_pickle(paths['train'])
    print(f"   ✅ Shape: {df_train.shape}")
    
    print(f"📂 Loading Test: {paths['test']}")
    df_test = pd.read_pickle(paths['test'])
    print(f"   ✅ Shape: {df_test.shape}")
    
    print(f"📂 Loading Validation: {paths['validation']}")
    df_val = pd.read_pickle(paths['validation'])
    print(f"   ✅ Shape: {df_val.shape}")
    
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
    print(f"   Train samples: {len(X_train):,}")
    print(f"   Test samples: {len(X_test):,}")
    print(f"   Val samples: {len(X_val):,}")
    
    print(f"\n📊 Train Class Distribution:")
    unique, counts = np.unique(y_train, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"   Class {cls}: {count:>6,} ({count/len(y_train)*100:>5.2f}%)")
    
    return X_train, X_test, X_val, y_train, y_test, y_val, feature_cols


def load_label_encoder(artifacts_path):
    encoder_path = artifacts_path / 'label_encoder.pkl'
    label_encoder = joblib.load(encoder_path)
    print(f"\n✅ Label encoder loaded from: {encoder_path}")
    print(f"   Classes: {label_encoder.classes_}")
    return label_encoder


# =============================================================================
# TRAINING
# =============================================================================

def train_random_forest(X_train, y_train, n_estimators=100, max_depth=25):
    print_header("TRAINING RANDOM FOREST")
    
    print(f"Hyperparameters:")
    print(f"  n_estimators: {n_estimators}")
    print(f"  max_depth: {max_depth}")
    print(f"  min_samples_split: 10")
    print(f"  min_samples_leaf: 5")
    print(f"  max_features: sqrt")
    print(f"  class_weight: balanced")
    print(f"  n_jobs: -1")
    
    start_time = time.time()
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        class_weight='balanced',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    print("\nTraining (this may take a few minutes)...")
    model.fit(X_train, y_train)
    
    elapsed = time.time() - start_time
    print(f"\n✅ Training complete in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)!")
    
    return model


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_model(model, X_train, X_test, X_val, y_train, y_test, y_val,
                   label_encoder, output_paths, dataset_type):
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
    print(f"RESULTS SUMMARY - Dataset: {dataset_type.upper()}")
    print("-"*80)
    print(f"{'Metric':<15} {'Train':>12} {'Test':>12} {'Val':>12} {'Status':>8}")
    print("-"*80)
    
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        train_val = metrics['Train'][metric]
        test_val = metrics['Test'][metric]
        val_val = metrics['Val'][metric]
        
        status = ""
        if metric == 'accuracy':
            status = "✅" if test_val >= 0.95 else "❌"
        elif metric == 'precision':
            status = "✅" if test_val >= 0.90 else "❌"
        elif metric == 'recall':
            status = "✅" if test_val >= 0.95 else "❌"
        
        print(f"{metric.capitalize():<15} {train_val:>12.4f} {test_val:>12.4f} {val_val:>12.4f} {status:>8}")
    
    print("-"*80)
    
    # Overfitting check
    acc_diff = metrics['Train']['accuracy'] - metrics['Test']['accuracy']
    if acc_diff > 0.05:
        print(f"\n⚠️ Possibile overfitting (Train-Test diff: {acc_diff:.4f})")
    else:
        print(f"\n✅ Buona generalizzazione (Train-Test diff: {acc_diff:.4f})")
    
    print("\n" + "-"*80)
    print("CLASSIFICATION REPORT (Test Set)")
    print("-"*80)
    print(classification_report(y_test, y_test_pred,
                                target_names=class_names,
                                digits=4,
                                zero_division=0))
    
    cm = confusion_matrix(y_test, y_test_pred)
    
    # Plots
    plots_dir = output_paths['plots']
    
    # 1. Confusion Matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - Random Forest\nDataset: {dataset_type.upper()}',
              fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    cm_path = plots_dir / 'confusion_matrix.png'
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Confusion matrix saved: {cm_path}")
    plt.close()
    
    # 2. Metrics Comparison
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    train_values = [metrics['Train'][k] for k in ['accuracy', 'precision', 'recall', 'f1']]
    test_values = [metrics['Test'][k] for k in ['accuracy', 'precision', 'recall', 'f1']]
    val_values = [metrics['Val'][k] for k in ['accuracy', 'precision', 'recall', 'f1']]
    
    x = np.arange(len(metric_names))
    width = 0.25
    
    bars1 = ax.bar(x - width, train_values, width, label='Train', alpha=0.8, color='steelblue')
    bars2 = ax.bar(x, test_values, width, label='Test', alpha=0.8, color='darkgreen')
    bars3 = ax.bar(x + width, val_values, width, label='Val', alpha=0.8, color='orange')
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Metrics Comparison - Random Forest\nDataset: {dataset_type.upper()}',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    
    ax.axhline(y=0.95, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.axhline(y=0.90, color='orange', linestyle='--', alpha=0.5, linewidth=1.5)
    
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    metrics_path = plots_dir / 'metrics_comparison.png'
    plt.savefig(metrics_path, dpi=150, bbox_inches='tight')
    print(f"📊 Metrics comparison saved: {metrics_path}")
    plt.close()
    
    # 3. Per-Class Performance
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    per_class_precision = precision_score(y_test, y_test_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_test, y_test_pred, average=None, zero_division=0)
    per_class_f1 = f1_score(y_test, y_test_pred, average=None, zero_division=0)
    
    x = np.arange(len(class_names))
    width = 0.25
    
    ax.bar(x - width, per_class_precision, width, label='Precision', alpha=0.8, color='steelblue')
    ax.bar(x, per_class_recall, width, label='Recall', alpha=0.8, color='darkgreen')
    ax.bar(x + width, per_class_f1, width, label='F1-Score', alpha=0.8, color='orange')
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Per-Class Performance - Random Forest\nDataset: {dataset_type.upper()}',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    perclass_path = plots_dir / 'per_class_performance.png'
    plt.savefig(perclass_path, dpi=150, bbox_inches='tight')
    print(f"📊 Per-class performance saved: {perclass_path}")
    plt.close()
    
    return metrics, cm


def plot_feature_importance(model, feature_names, output_paths, dataset_type, n_features=20):
    if feature_names is None or len(feature_names) == 0:
        print("\n⚠️ Feature names not available")
        return
    
    print_header("FEATURE IMPORTANCE")
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:n_features]
    
    print(f"\nTop {min(15, n_features)} Most Important Features:")
    for i in range(min(15, n_features)):
        idx = indices[i]
        feat_name = feature_names[idx] if idx < len(feature_names) else f"Feature_{idx}"
        print(f"{i+1:2d}. {feat_name:45s} {importances[idx]:.6f}")
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    plot_features = []
    plot_importances = []
    
    for idx in indices:
        if idx < len(feature_names):
            plot_features.append(feature_names[idx])
            plot_importances.append(importances[idx])
    
    colors = plt.cm.Greens(np.linspace(0.4, 0.8, len(plot_importances)))
    ax.barh(range(len(plot_importances)), plot_importances[::-1], alpha=0.9, color=colors[::-1])
    ax.set_yticks(range(len(plot_features)))
    ax.set_yticklabels(plot_features[::-1], fontsize=9)
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(f'Top {n_features} Feature Importances - Random Forest\nDataset: {dataset_type.upper()}',
                 fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    feat_path = output_paths['plots'] / 'feature_importance.png'
    plt.savefig(feat_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Feature importance saved: {feat_path}")
    plt.close()


def save_model(model, output_paths, dataset_type):
    print_header("SAVING MODEL")
    
    model_path = output_paths['model']
    joblib.dump(model, model_path)
    size_mb = model_path.stat().st_size / 1024**2
    print(f"💾 Model saved: {model_path}")
    print(f"   Size: {size_mb:.2f} MB")


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Train Random Forest on CICIoT2023 (Original or SMOTE)'
    )
    parser.add_argument('--dataset-type', type=str, required=True,
                        choices=['original', 'smote'],
                        help='Dataset type: original (unbalanced) or smote (balanced)')
    parser.add_argument('--n-estimators', type=int, default=100,
                        help='Number of trees in forest (default: 100)')
    parser.add_argument('--max-depth', type=int, default=25,
                        help='Max depth of each tree (default: 25)')
    
    args = parser.parse_args()
    
    print("\n" + "🌲"*40)
    print(f"RANDOM FOREST TRAINING - {args.dataset_type.upper()}".center(80))
    print("🌲"*40)
    
    dataset_type = validate_dataset_type(args.dataset_type)
    data_paths = get_paths(dataset_type)
    output_paths = get_output_paths(dataset_type)
    
    print(f"\n📂 Configuration:")
    print(f"   Dataset type: {dataset_type}")
    print(f"   Train: {data_paths['train']}")
    print(f"   Test: {data_paths['test']}")
    print(f"   Validation: {data_paths['validation']}")
    print(f"   Model output: {output_paths['model']}")
    print(f"   Plots output: {output_paths['plots']}")
    
    X_train, X_test, X_val, y_train, y_test, y_val, feature_cols = load_processed_data(data_paths)
    label_encoder = load_label_encoder(data_paths['artifacts'])
    
    model = train_random_forest(X_train, y_train, 
                                n_estimators=args.n_estimators,
                                max_depth=args.max_depth)
    
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
        print("⚠️ Model does not meet all requirements yet.")
    
    print(f"\n📁 Outputs saved to:")
    print(f"   Model: {output_paths['model']}")
    print(f"   Plots: {output_paths['plots']}/")


if __name__ == '__main__':
    main()