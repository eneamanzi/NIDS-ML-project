"""
Training Decision Tree per CICIoT2023 - Multi-Class Classification.
Rifattorizzato per supportare dataset original e SMOTE con path management robusto.

Usage:
    python src/train_decision_tree.py --dataset-type original
    python src/train_decision_tree.py --dataset-type smote
"""

import pandas as pd
import numpy as np
import joblib
import time
import json
import os
import sys
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# PATH CONFIGURATION - Robusto indipendentemente da dove si lancia lo script
# =============================================================================

# Base directory del progetto (parent della cartella src/)
BASE_DIR = Path(__file__).resolve().parent.parent

# Cartelle dati (FISSE)
DATA_DIR = BASE_DIR / "data" / "processed"
CICIOT_DIR = DATA_DIR / "CICIOT23"  # Test/Val sempre qui
SMOTE_DIR = DATA_DIR / "SMOTE"      # Train alternativo

# Cartelle output (DINAMICHE in base a dataset_type)
MODELS_DIR = BASE_DIR / "models" / "DecisionTree"
DOCS_DIR = BASE_DIR / "docs" / "DecisionTree"


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def print_header(text):
    """Stampa header formattato."""
    print("\n" + "="*80)
    print(text.center(80))
    print("="*80 + "\n")


def validate_dataset_type(dataset_type):
    """Valida il parametro dataset_type."""
    valid_types = ['original', 'smote']
    if dataset_type not in valid_types:
        raise ValueError(f"Invalid dataset_type: {dataset_type}. Must be one of {valid_types}")
    return dataset_type


def get_paths(dataset_type):
    """
    Restituisce i path corretti in base al dataset_type.
    
    REGOLA:
    - Train: CICIOT23 (original) o SMOTE (smote)
    - Test: SEMPRE CICIOT23 (valutazione realistica)
    - Validation: SEMPRE CICIOT23 (valutazione realistica)
    
    Returns:
        dict con train_path, test_path, val_path, artifacts_path
    """
    paths = {
        'test': CICIOT_DIR / 'test_processed.pkl',
        'validation': CICIOT_DIR / 'validation_processed.pkl',
        'artifacts': CICIOT_DIR  # scaler, encoder sempre da original
    }
    
    if dataset_type == 'original':
        paths['train'] = CICIOT_DIR / 'train_processed.pkl'
    elif dataset_type == 'smote':
        paths['train'] = SMOTE_DIR / 'train_smote.pkl'
    
    # Verifica che tutti i path esistano
    for key, path in paths.items():
        if key != 'artifacts' and not path.exists():
            raise FileNotFoundError(f"{key.capitalize()} file not found: {path}")
    
    return paths


def get_output_paths(dataset_type):
    """
    Restituisce i path di output per modelli e plots.
    
    Returns:
        dict con model_path, plots_dir
    """
    # Crea directories se non esistono
    model_dir = MODELS_DIR
    plots_dir = DOCS_DIR / dataset_type
    
    model_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    return {
        'model': model_dir / f'dt_model_{dataset_type}.pkl',
        'plots': plots_dir
    }


# =============================================================================
# DATA LOADING
# =============================================================================

def load_processed_data(paths):
    """
    Carica dataset processati in formato PKL.
    
    Args:
        paths: Dict con train_path, test_path, validation_path
    
    Returns:
        X_train, X_test, X_val, y_train, y_test, y_val, feature_cols
    """
    print_header("LOADING PROCESSED DATA")
    
    # Load train
    print(f"📂 Loading Train: {paths['train']}")
    df_train = pd.read_pickle(paths['train'])
    print(f"   ✅ Shape: {df_train.shape}")
    
    # Load test
    print(f"📂 Loading Test: {paths['test']}")
    df_test = pd.read_pickle(paths['test'])
    print(f"   ✅ Shape: {df_test.shape}")
    
    # Load validation
    print(f"📂 Loading Validation: {paths['validation']}")
    df_val = pd.read_pickle(paths['validation'])
    print(f"   ✅ Shape: {df_val.shape}")
    
    # Separa features e labels
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
    
    # Class distribution
    print(f"\n📊 Train Class Distribution:")
    unique, counts = np.unique(y_train, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"   Class {cls}: {count:>6,} ({count/len(y_train)*100:>5.2f}%)")
    
    return X_train, X_test, X_val, y_train, y_test, y_val, feature_cols


def load_label_encoder(artifacts_path):
    """Carica label encoder per interpretare le classi."""
    encoder_path = artifacts_path / 'label_encoder.pkl'
    label_encoder = joblib.load(encoder_path)
    print(f"\n✅ Label encoder loaded from: {encoder_path}")
    print(f"   Classes: {label_encoder.classes_}")
    return label_encoder


# =============================================================================
# TRAINING
# =============================================================================

def train_decision_tree(X_train, y_train, max_depth=20):
    """
    Train Decision Tree classifier per multi-class.
    
    Args:
        X_train: Training features
        y_train: Training labels (macro-categories encoded)
        max_depth: Profondità massima albero
    
    Returns:
        Trained model
    """
    print_header("TRAINING DECISION TREE")
    
    print(f"Hyperparameters:")
    print(f"  max_depth: {max_depth}")
    print(f"  min_samples_split: 20")
    print(f"  min_samples_leaf: 10")
    print(f"  criterion: gini")
    print(f"  class_weight: balanced")
    
    start_time = time.time()
    
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=20,
        min_samples_leaf=10,
        criterion='gini',
        class_weight='balanced',
        random_state=42
    )
    
    print("\nTraining...")
    model.fit(X_train, y_train)
    
    elapsed = time.time() - start_time
    print(f"✅ Training complete in {elapsed:.2f} seconds!")
    print(f"   Tree depth: {model.get_depth()}")
    print(f"   Number of leaves: {model.get_n_leaves()}")
    
    return model


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_model(model, X_train, X_test, X_val, y_train, y_test, y_val,
                   label_encoder, output_paths, dataset_type):
    """
    Valuta modello su train, test e validation set.
    
    Returns:
        Dict con metriche
    """
    print_header("MODEL EVALUATION")
    
    # Predizioni
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_val_pred = model.predict(X_val)
    
    # Nomi classi
    class_names = label_encoder.classes_
    
    # Metriche
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
    
    # Stampa risultati
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
    
    # Classification report (test)
    print("\n" + "-"*80)
    print("CLASSIFICATION REPORT (Test Set)")
    print("-"*80)
    print(classification_report(y_test, y_test_pred,
                                target_names=class_names,
                                digits=4,
                                zero_division=0))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    
    # Plots
    plots_dir = output_paths['plots']
    
    # 1. Confusion Matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - Decision Tree\nDataset: {dataset_type.upper()}',
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
    
    ax.bar(x - width, train_values, width, label='Train', alpha=0.8)
    ax.bar(x, test_values, width, label='Test', alpha=0.8)
    ax.bar(x + width, val_values, width, label='Val', alpha=0.8)
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Metrics Comparison - Decision Tree\nDataset: {dataset_type.upper()}',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.legend(fontsize=10)
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    
    # Target lines
    ax.axhline(y=0.95, color='r', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(y=0.90, color='orange', linestyle='--', alpha=0.5, linewidth=1)
    
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
    
    ax.bar(x - width, per_class_precision, width, label='Precision', alpha=0.8)
    ax.bar(x, per_class_recall, width, label='Recall', alpha=0.8)
    ax.bar(x + width, per_class_f1, width, label='F1-Score', alpha=0.8)
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Per-Class Performance - Decision Tree\nDataset: {dataset_type.upper()}',
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
    """Visualizza feature importance."""
    if feature_names is None or len(feature_names) == 0:
        print("\n⚠️ Feature names not available")
        return
    
    print_header("FEATURE IMPORTANCE")
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:n_features]
    
    print(f"\nTop {min(10, n_features)} Most Important Features:")
    for i in range(min(10, n_features)):
        idx = indices[i]
        feat_name = feature_names[idx] if idx < len(feature_names) else f"Feature_{idx}"
        print(f"{i+1:2d}. {feat_name:40s} {importances[idx]:.6f}")
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    plot_features = []
    plot_importances = []
    
    for idx in indices:
        if idx < len(feature_names):
            plot_features.append(feature_names[idx])
            plot_importances.append(importances[idx])
    
    ax.barh(range(len(plot_importances)), plot_importances[::-1], alpha=0.8)
    ax.set_yticks(range(len(plot_features)))
    ax.set_yticklabels(plot_features[::-1], fontsize=9)
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(f'Top {n_features} Feature Importances - Decision Tree\nDataset: {dataset_type.upper()}',
                 fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    feat_path = output_paths['plots'] / 'feature_importance.png'
    plt.savefig(feat_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Feature importance saved: {feat_path}")
    plt.close()


def save_model(model, output_paths, dataset_type):
    """Salva modello addestrato."""
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
        description='Train Decision Tree on CICIoT2023 (Original or SMOTE)'
    )
    parser.add_argument('--dataset-type', type=str, required=True,
                        choices=['original', 'smote'],
                        help='Dataset type: original (unbalanced) or smote (balanced)')
    parser.add_argument('--max-depth', type=int, default=20,
                        help='Max depth of decision tree (default: 20)')
    
    args = parser.parse_args()
    
    print("\n" + "🌳"*40)
    print(f"DECISION TREE TRAINING - {args.dataset_type.upper()}".center(80))
    print("🌳"*40)
    
    # Validate
    dataset_type = validate_dataset_type(args.dataset_type)
    
    # Get paths
    data_paths = get_paths(dataset_type)
    output_paths = get_output_paths(dataset_type)
    
    print(f"\n📂 Configuration:")
    print(f"   Dataset type: {dataset_type}")
    print(f"   Train: {data_paths['train']}")
    print(f"   Test: {data_paths['test']}")
    print(f"   Validation: {data_paths['validation']}")
    print(f"   Model output: {output_paths['model']}")
    print(f"   Plots output: {output_paths['plots']}")
    
    # Load data
    X_train, X_test, X_val, y_train, y_test, y_val, feature_cols = load_processed_data(data_paths)
    label_encoder = load_label_encoder(data_paths['artifacts'])
    
    # Train
    model = train_decision_tree(X_train, y_train, max_depth=args.max_depth)
    
    # Evaluate
    metrics, cm = evaluate_model(
        model, X_train, X_test, X_val,
        y_train, y_test, y_val,
        label_encoder, output_paths, dataset_type
    )
    
    # Feature importance
    plot_feature_importance(model, feature_cols, output_paths, dataset_type)
    
    # Save
    save_model(model, output_paths, dataset_type)
    
    # Summary
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