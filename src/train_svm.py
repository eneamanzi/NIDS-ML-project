"""
Training SVM (Support Vector Machine) per CICIoT2023 - Multi-Class Classification.

Note Tecniche:
- SVM scala in modo quadratico O(n^2) con il numero di campioni.
- È il modello più lento da addestrare tra quelli scelti.
- probability=True è necessario per ottenere le probabilità di confidenza, ma rallenta ulteriormente.

Target: Accuracy >95%, Precision >90%, Recall >95%
"""

import pandas as pd
import numpy as np
import joblib
import time
import json
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import os

def print_header(text):
    """Stampa header formattato."""
    print("\n" + "="*80)
    print(text.center(80))
    print("="*80 + "\n")


def load_processed_data(data_dir='../data/processed'):
    """
    Carica dataset processati in formato PKL.
    """
    print_header("LOADING PROCESSED DATA")
    
    # Load train
    train_path = f"{data_dir}/train_processed.pkl"
    df_train = pd.read_pickle(train_path)
    print(f"✅ Train loaded: {train_path}")
    print(f"   Shape: {df_train.shape}")
    
    # Load test
    test_path = f"{data_dir}/test_processed.pkl"
    df_test = pd.read_pickle(test_path)
    print(f"✅ Test loaded: {test_path}")
    print(f"   Shape: {df_test.shape}")
    
    # Load val
    val_path = f"{data_dir}/validation_processed.pkl"
    df_val = pd.read_pickle(val_path)
    print(f"✅ Val loaded: {val_path}")
    print(f"   Shape: {df_val.shape}")
    
    # Separa features e labels
    feature_cols = [col for col in df_train.columns if col not in ['y_macro_encoded', 'y_specific']]
    
    X_train = df_train[feature_cols].values
    y_train = df_train['y_macro_encoded'].values
    
    X_test = df_test[feature_cols].values
    y_test = df_test['y_macro_encoded'].values
    
    X_val = df_val[feature_cols].values
    y_val = df_val['y_macro_encoded'].values
    
    print(f"\nFeatures: {len(feature_cols)}")
    print(f"Classes (macro): {len(np.unique(y_train))}")
    
    return X_train, X_test, X_val, y_train, y_test, y_val, feature_cols


def load_mapping_info(data_dir='../data/processed'):
    """Carica info mapping per interpretare le classi."""
    mapping_path = f"{data_dir}/mapping_info.json"
    with open(mapping_path, 'r') as f:
        mapping_info = json.load(f)
    
    # Carica label encoder per ottenere nomi classi
    encoder_path = f"{data_dir}/label_encoder.pkl"
    label_encoder = joblib.load(encoder_path)
    
    return mapping_info, label_encoder


def train_svm(X_train, y_train):
    """
    Train SVM classifier per multi-class.
    """
    print_header("TRAINING SVM (Multi-Class)")
    
    print("⚠️  WARNING: SVM training allows NO parallel processing (single core usually).")
    print("    Ensure your dataset size is manageable (e.g. < 50k rows for reasonable time).")
    
    print(f"\nHyperparameters:")
    print(f"  kernel: rbf (Radial Basis Function)")
    print(f"  C: 1.0")
    print(f"  gamma: scale")
    print(f"  class_weight: balanced")
    print(f"  probability: True (slow!)")
    print(f"  cache_size: 2000 MB")
    
    start_time = time.time()
    
    model = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        class_weight='balanced', # Cruciale per classi sbilanciate
        probability=True,        # Necessario per la Dashboard se vogliamo le % di confidenza
        cache_size=2000,         # Usa 2GB di RAM come cache per velocizzare
        random_state=42,
        verbose=True             # Stampa log di avanzamento libsvm
    )
    
    print("\nTraining (this might take a while)...")
    model.fit(X_train, y_train)
    
    elapsed = time.time() - start_time
    print(f"\n✅ Training complete in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)!")
    
    return model


def evaluate_model(model, X_train, X_test, X_val, y_train, y_test, y_val, 
                   label_encoder, save_dir='../docs'):
    """
    Valuta modello su train, test e validation set.
    """
    print_header("MODEL EVALUATION (Multi-Class)")
    
    # Predizioni
    print("Making predictions on Validation Set...")
    y_val_pred = model.predict(X_val)
    
    print("Making predictions on Test Set...")
    y_test_pred = model.predict(X_test)
    
    # ATTENZIONE: Predire sul train set con SVM è lento. 
    # Se il train set è > 50k, facciamo subsample solo per le metriche.
    print("Making predictions on Train Set...")
    if len(X_train) > 50000:
        print("   (Using subset of 50k samples for Train metrics evaluation speed...)")
        idx = np.random.choice(len(X_train), 50000, replace=False)
        y_train_sample = y_train[idx]
        y_train_pred = model.predict(X_train[idx])
    else:
        y_train_sample = y_train
        y_train_pred = model.predict(X_train)
    
    # Nomi classi
    class_names = label_encoder.classes_
    
    # Metriche
    metrics = {}
    
    def get_metrics(y_true, y_p):
        return {
            'accuracy': accuracy_score(y_true, y_p),
            'precision': precision_score(y_true, y_p, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_p, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_p, average='weighted', zero_division=0)
        }

    metrics['Train'] = get_metrics(y_train_sample, y_train_pred)
    metrics['Test'] = get_metrics(y_test, y_test_pred)
    metrics['Val'] = get_metrics(y_val, y_val_pred)
    
    # Stampa risultati
    print("\n" + "-"*80)
    print("RESULTS SUMMARY")
    print("-"*80)
    print(f"{'Metric':<15} {'Train':>12} {'Test':>12} {'Val':>12} {'Status':>8}")
    print("-"*80)
    
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        train_val = metrics['Train'][metric]
        test_val = metrics['Test'][metric]
        val_val = metrics['Val'][metric]
        
        status = "✅" if (metric == 'accuracy' and test_val >= 0.95) else "❌" if metric == 'accuracy' else ""
        
        print(f"{metric.capitalize():<15} {train_val:>12.4f} {test_val:>12.4f} {val_val:>12.4f} {status:>8}")
    
    print("-"*80)
    
    # Classification report
    print("\n" + "-"*80)
    print("CLASSIFICATION REPORT (Test Set)")
    print("-"*80)
    print(classification_report(y_test, y_test_pred, 
                                target_names=class_names,
                                digits=4,
                                zero_division=0))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    
    # Visualizzazioni
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Confusion Matrix Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', # Rosso per SVM
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - SVM (RBF Kernel)\nCICIoT2023 Multi-Class', 
              fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/confusion_matrix_svm_multiclass.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Metrics Comparison
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    train_values = [metrics['Train'][k] for k in ['accuracy', 'precision', 'recall', 'f1']]
    test_values = [metrics['Test'][k] for k in ['accuracy', 'precision', 'recall', 'f1']]
    val_values = [metrics['Val'][k] for k in ['accuracy', 'precision', 'recall', 'f1']]
    
    x = np.arange(len(metric_names))
    width = 0.25
    
    ax.bar(x - width, train_values, width, label='Train', alpha=0.8, color='firebrick')
    ax.bar(x, test_values, width, label='Test', alpha=0.8, color='red')
    ax.bar(x + width, val_values, width, label='Val', alpha=0.8, color='lightcoral')
    
    ax.set_title('Metrics Comparison - SVM Multi-Class', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/metrics_comparison_svm_multiclass.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return metrics


def save_model(model, filepath='../models/SVM_CICIoT2023_multiclass.pkl'):
    """Salva modello addestrato."""
    print_header("SAVING MODEL")
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    print("⚠️  Saving SVM model... (This can be large due to support vectors)")
    joblib.dump(model, filepath, compress=3)
    
    size_mb = os.path.getsize(filepath) / 1024**2
    print(f"💾 Model saved to {filepath}")
    print(f"   Size: {size_mb:.2f} MB")


# Main
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train SVM on CICIoT2023 Multi-Class')
    parser.add_argument('--data-dir', type=str, default='../data/processed/CICIOT23',
                        help='Directory with preprocessed data')
    parser.add_argument('--output-dir', type=str, default='../docs/svm',
                        help='Output directory for plots')
    parser.add_argument('--model-path', type=str, default='../models/SVM_CICIoT2023_multiclass.pkl',
                        help='Path to save trained model')
    
    args = parser.parse_args()
    
    print("\n" + "🛡️"*40)
    print("SVM TRAINING (RBF Kernel) - MULTI-CLASS".center(80))
    print("CICIoT2023 Dataset".center(80))
    print("🛡️"*40)
    
    # Load data
    X_train, X_test, X_val, y_train, y_test, y_val, feature_cols = load_processed_data(args.data_dir)
    
    # Load mapping info
    mapping_info, label_encoder = load_mapping_info(args.data_dir)
    
    # Train
    model = train_svm(X_train, y_train)
    
    # Evaluate
    metrics = evaluate_model(model, X_train, X_test, X_val, 
                             y_train, y_test, y_val,
                             label_encoder, save_dir=args.output_dir)
    
    # Save
    save_model(model, filepath=args.model_path)
    
    print_header("✅ SVM TRAINING COMPLETE!")