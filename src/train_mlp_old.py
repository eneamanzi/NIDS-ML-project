"""
Training MLP (Multi-Layer Perceptron) - NEURAL NETWORK per NIDS

PERCHÉ MLP:
- Approccio completamente diverso (neural network vs tree-based)
- Cattura pattern non-lineari complessi
- Ottimo complemento a XGBoost/RF per ensemble
- Scalabile e adattabile

ARCHITETTURA MLP:
- Input layer: N features
- Hidden layers: 2-3 layers (128-256-128 neurons)
- Output layer: N classes (softmax)
- Activation: ReLU
- Optimizer: Adam
- Loss: categorical_crossentropy

VANTAGGI per NIDS:
- Pattern discovery automatico
- Robusto a feature correlate
- Ottimo per feature continue (network traffic)

Usage (from ANY directory):
    python src/train_mlp.py --dataset-type original
    python src/train_mlp.py --dataset-type smote --epochs 50 --batch-size 256
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

# Scikit-learn MLP import
from sklearn.neural_network import MLPClassifier

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

MODELS_DIR = BASE_DIR / "models" / "MLP"
DOCS_DIR = BASE_DIR / "docs" / "MLP"

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
        'model': model_dir / f'mlp_model_{dataset_type}.pkl',
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

def train_mlp(X_train, y_train, hidden_layers=(256, 128, 64), 
              max_iter=100, learning_rate_init=0.001):
    """
    Train Multi-Layer Perceptron (Neural Network).
    
    MLP ARCHITECTURE:
    - Input layer: N features (auto)
    - Hidden layers: Customizable (default: 256-128-64)
    - Output layer: N classes (auto)
    - Activation: ReLU (hidden), softmax (output)
    - Optimizer: Adam
    - Learning rate: 0.001 (default)
    - Batch size: auto (min(200, n_samples))
    - Early stopping: Enabled (validation-based)
    
    BEST PRACTICES per NIDS:
    - Hidden layers: 2-3 layers sufficiente
    - Neurons: Decrescente (256→128→64)
    - ReLU activation: Migliore per network traffic
    - Adam optimizer: Adaptive learning rate
    - Early stopping: Previene overfitting
    
    Args:
        X_train: Training features (MUST be scaled!)
        y_train: Training labels
        hidden_layers: Tuple of hidden layer sizes
        max_iter: Max epochs
        learning_rate_init: Initial learning rate
    
    Returns:
        Trained MLP model
    """
    print_header("TRAINING MLP (Neural Network)")
    
    n_features = X_train.shape[1]
    n_classes = len(np.unique(y_train))
    
    print(f"Network Architecture:")
    print(f"  Input layer: {n_features} features")
    for i, size in enumerate(hidden_layers):
        print(f"  Hidden layer {i+1}: {size} neurons (ReLU)")
    print(f"  Output layer: {n_classes} classes (softmax)")
    
    print(f"\nHyperparameters:")
    print(f"  hidden_layer_sizes: {hidden_layers}")
    print(f"  activation: relu")
    print(f"  solver: adam")
    print(f"  learning_rate_init: {learning_rate_init}")
    print(f"  max_iter: {max_iter}")
    print(f"  early_stopping: True")
    print(f"  validation_fraction: 0.1")
    print(f"  batch_size: auto")
    print(f"  alpha: 0.0001 (L2 regularization)")
    
    # Calcola class weights per imbalance
    class_counts = np.bincount(y_train)
    n_samples = len(y_train)
    
    print(f"\nClass balance analysis:")
    for cls_idx, count in enumerate(class_counts):
        pct = count / n_samples * 100
        print(f"  Class {cls_idx}: {count:>8,} ({pct:>5.2f}%)")
    
    # Check if data is scaled
    print(f"\nData scaling check:")
    x_min, x_max = X_train.min(), X_train.max()
    x_mean, x_std = X_train.mean(), X_train.std()
    print(f"  Range: [{x_min:.4f}, {x_max:.4f}]")
    print(f"  Mean: {x_mean:.4f}, Std: {x_std:.4f}")
    
    if abs(x_mean) > 1.0 or x_std > 10.0:
        print(f"  ⚠️ Data may not be properly scaled!")
        print(f"  ⚠️ MLP works best with standardized features (mean=0, std=1)")
    else:
        print(f"  ✅ Data appears to be scaled properly")
    
    start_time = time.time()
    
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation='relu',
        solver='adam',
        alpha=0.0001,  # L2 regularization
        batch_size='auto',  # min(200, n_samples)
        learning_rate='adaptive',  # Adjusts when loss plateaus
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        shuffle=True,
        random_state=42,
        early_stopping=True,  # Stop when validation score stops improving
        validation_fraction=0.1,
        n_iter_no_change=10,  # Patience for early stopping
        verbose=True
    )
    
    print("\nTraining MLP (with early stopping)...")
    print("This may take several minutes depending on dataset size...")
    
    model.fit(X_train, y_train)
    
    elapsed = time.time() - start_time
    
    print(f"\n✅ Training complete!")
    print(f"   Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"   Iterations: {model.n_iter_}")
    print(f"   Final loss: {model.loss_:.6f}")
    
    if model.n_iter_ < max_iter:
        print(f"   Early stopping triggered at iteration {model.n_iter_}")
    
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
        print(f"   Consider: reducing max_iter, increasing alpha (regularization)")
    else:
        print(f"\n✅ Good generalization (Train-Test diff: {acc_diff:.4f})")
    
    print("\n" + "-"*80)
    print("CLASSIFICATION REPORT (Test Set)")
    print("-"*80)
    print(classification_report(y_test, y_test_pred,
                                target_names=class_names,
                                digits=4, zero_division=0))
    
    cm = confusion_matrix(y_test, y_test_pred)
    
    generate_plots(cm, metrics, class_names, model, output_paths, dataset_type)
    
    return metrics, cm

def generate_plots(cm, metrics, class_names, model, output_paths, dataset_type):
    """Genera plots."""
    plots_dir = output_paths['plots']
    
    # 1. Confusion Matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - MLP\nDataset: {dataset_type.upper()}',
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
    
    ax.bar(x - width, train_values, width, label='Train', alpha=0.8, color='mediumpurple')
    ax.bar(x, test_values, width, label='Test', alpha=0.8, color='darkviolet')
    ax.bar(x + width, val_values, width, label='Val', alpha=0.8, color='plum')
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Metrics Comparison - MLP\nDataset: {dataset_type.upper()}',
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
    
    # 3. Training Loss Curve
    if hasattr(model, 'loss_curve_'):
        plt.figure(figsize=(10, 6))
        plt.plot(model.loss_curve_, color='darkviolet', linewidth=2)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title(f'Training Loss Curve - MLP\nDataset: {dataset_type.upper()}',
                 fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        loss_path = plots_dir / 'loss_curve.png'
        plt.savefig(loss_path, dpi=150, bbox_inches='tight')
        print(f"📊 Loss curve: {loss_path}")
        plt.close()

def save_model(model, output_paths, dataset_type):
    """Salva modello."""
    print_header("SAVING MODEL")
    
    model_path = output_paths['model']
    joblib.dump(model, model_path)
    
    size_mb = model_path.stat().st_size / 1024**2
    print(f"💾 Model: {model_path}")
    print(f"   Size: {size_mb:.2f} MB")
    
    # Save network architecture info
    arch_info = {
        'input_size': model.n_features_in_,
        'hidden_layers': model.hidden_layer_sizes,
        'output_size': model.n_outputs_,
        'n_layers': model.n_layers_,
        'n_iter': model.n_iter_,
        'final_loss': float(model.loss_)
    }
    
    arch_path = output_paths['plots'] / 'architecture.json'
    with open(arch_path, 'w') as f:
        json.dump(arch_info, f, indent=2)
    
    print(f"   Architecture info: {arch_path}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Train MLP (Multi-Layer Perceptron Neural Network)'
    )
    parser.add_argument('--dataset-type', type=str, required=True,
                        choices=['original', 'smote', 'borderline', 'adasyn', 'ctgan'],
                        help='Dataset type')
    parser.add_argument('--hidden-layers', type=str, default='256,128,64',
                        help='Hidden layer sizes (comma-separated, default: 256,128,64)')
    parser.add_argument('--max-iter', type=int, default=100,
                        help='Max epochs (default: 100)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Initial learning rate (default: 0.001)')
    
    args = parser.parse_args()
    
    # Parse hidden layers
    hidden_layers = tuple(map(int, args.hidden_layers.split(',')))
    
    print("\n" + "🧠"*40)
    print(f"MLP TRAINING - {args.dataset_type.upper()}".center(80))
    print("🧠"*40)
    
    dataset_type = validate_dataset_type(args.dataset_type)
    data_paths = get_paths(dataset_type)
    output_paths = get_output_paths(dataset_type)
    
    print(f"\n📂 Configuration:")
    print(f"   Dataset: {dataset_type}")
    print(f"   Train: {data_paths['train']}")
    print(f"   Hidden layers: {hidden_layers}")
    print(f"   Max iterations: {args.max_iter}")
    print(f"   Learning rate: {args.learning_rate}")
    print(f"   Output: {output_paths['model']}")
    
    X_train, X_test, X_val, y_train, y_test, y_val, feature_cols = load_processed_data(data_paths)
    label_encoder = load_label_encoder(data_paths['artifacts'])
    
    model = train_mlp(X_train, y_train,
                     hidden_layers=hidden_layers,
                     max_iter=args.max_iter,
                     learning_rate_init=args.learning_rate)
    
    metrics, cm = evaluate_model(
        model, X_train, X_test, X_val,
        y_train, y_test, y_val,
        label_encoder, output_paths, dataset_type
    )
    
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
    
    print(f"\n💡 Tips for improving MLP:")
    if test_metrics['accuracy'] < 0.95:
        print("   - Try more hidden layers or neurons")
        print("   - Increase max_iter (more epochs)")
        print("   - Try different learning rates (0.01, 0.0001)")
    
    acc_diff = metrics['Train']['accuracy'] - test_metrics['accuracy']
    if acc_diff > 0.05:
        print("   - Model is overfitting:")
        print("     * Increase alpha (regularization)")
        print("     * Reduce network size")
        print("     * Use more training data")
    
    print(f"\n📁 Outputs:")
    print(f"   Model: {output_paths['model']}")
    print(f"   Plots: {output_paths['plots']}/")

if __name__ == '__main__':
    main()