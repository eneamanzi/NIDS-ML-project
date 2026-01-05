"""
Training k-NN (k-Nearest Neighbors) per CICIoT2023 - Multi-Class Classification.

Note Tecniche:
- k-NN è un algoritmo "Lazy Learner": il training è istantaneo (memorizza i dati),
  ma la predizione può essere lenta su dataset grandi.
- Non produce Feature Importance nativa.
- Ideale per rilevare pattern locali di attacco non lineari.

Target: Accuracy >95%, Precision >90%, Recall >95%
"""

import pandas as pd
import numpy as np
import joblib
import time
import json
from sklearn.neighbors import KNeighborsClassifier
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
    y_specific_train = df_train['y_specific'].values
    
    X_test = df_test[feature_cols].values
    y_test = df_test['y_macro_encoded'].values
    y_specific_test = df_test['y_specific'].values
    
    X_val = df_val[feature_cols].values
    y_val = df_val['y_macro_encoded'].values
    y_specific_val = df_val['y_specific'].values
    
    print(f"\nFeatures: {len(feature_cols)}")
    print(f"Classes (macro): {len(np.unique(y_train))}")
    
    return X_train, X_test, X_val, y_train, y_test, y_val, feature_cols, df_train


def load_mapping_info(data_dir='../data/processed'):
    """Carica info mapping per interpretare le classi."""
    mapping_path = f"{data_dir}/mapping_info.json"
    with open(mapping_path, 'r') as f:
        mapping_info = json.load(f)
    
    # Carica label encoder per ottenere nomi classi
    encoder_path = f"{data_dir}/label_encoder.pkl"
    label_encoder = joblib.load(encoder_path)
    
    return mapping_info, label_encoder


def train_knn(X_train, y_train, n_neighbors=5):
    """
    Train k-NN classifier per multi-class.
    """
    print_header("TRAINING k-NN (Multi-Class)")
    
    print(f"Hyperparameters:")
    print(f"  n_neighbors (k): {n_neighbors}")
    print(f"  weights: distance (i vicini più prossimi contano di più)")
    print(f"  metric: euclidean")
    print(f"  algorithm: auto")
    print(f"  n_jobs: -1 (parallel processing)")
    
    start_time = time.time()
    
    # Configurazione ottimizzata per Cyber Security
    # 'distance' è spesso meglio di 'uniform' per attacchi rari ma densi
    model = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights='distance', 
        metric='euclidean',
        n_jobs=-1
    )
    
    print("\nFitting model (Memorizing training data)...")
    model.fit(X_train, y_train)
    
    elapsed = time.time() - start_time
    print(f"✅ Training complete in {elapsed:.2f} seconds!")
    
    return model


def evaluate_model(model, X_train, X_test, X_val, y_train, y_test, y_val, 
                   label_encoder, save_dir='../docs'):
    """
    Valuta modello su train, test e validation set.
    """
    print_header("MODEL EVALUATION (Multi-Class)")
    print("⚠️  WARNING: k-NN prediction implies calculating distances to ALL training points.")
    print("    This step might be slower than Random Forest prediction.")
    
    # Predizioni
    print("\nPredicting on Validation Set...")
    y_val_pred = model.predict(X_val)
    
    print("Predicting on Test Set...")
    y_test_pred = model.predict(X_test)
    
    # NOTA: Saltiamo la predizione sul TRAIN set completo se è troppo grande,
    # altrimenti il k-NN impiega ore. Facciamo un sample se necessario.
    print("Predicting on Train Set (might take time)...")
    if len(X_train) > 100000:
        print(f"    Dataset too large ({len(X_train)}), using a subset of 50k for Train Metric Estimation...")
        # Subset solo per la metrica di "autovalutazione", non cambia il modello
        idx = np.random.choice(len(X_train), 50000, replace=False)
        y_train_pred = model.predict(X_train[idx])
        y_train_ref = y_train[idx]
    else:
        y_train_pred = model.predict(X_train)
        y_train_ref = y_train
    
    # Nomi classi
    class_names = label_encoder.classes_
    
    # Metriche
    metrics = {}
    
    # Helper per calcolo
    def get_metrics(y_true, y_p):
        return {
            'accuracy': accuracy_score(y_true, y_p),
            'precision': precision_score(y_true, y_p, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_p, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_p, average='weighted', zero_division=0)
        }

    metrics['Train'] = get_metrics(y_train_ref, y_train_pred)
    metrics['Test'] = get_metrics(y_test, y_test_pred)
    metrics['Val'] = get_metrics(y_val, y_val_pred)
    
    # Stampa risultati
    print("\n" + "-"*80)
    print("RESULTS SUMMARY")
    print("-"*80)
    print(f"{'Metric':<15} {'Train (Est.)':>12} {'Test':>12} {'Val':>12} {'Status':>8}")
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
    print(classification_report(y_test, y_test_pred, target_names=class_names, digits=4, zero_division=0))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    
    # Visualizzazioni
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Confusion Matrix Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', # Colore diverso per distinguere k-NN
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - k-NN (k={model.n_neighbors})\nCICIoT2023 Multi-Class', 
              fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/confusion_matrix_knn_multiclass.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Metrics Comparison
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    train_values = [metrics['Train'][k] for k in ['accuracy', 'precision', 'recall', 'f1']]
    test_values = [metrics['Test'][k] for k in ['accuracy', 'precision', 'recall', 'f1']]
    val_values = [metrics['Val'][k] for k in ['accuracy', 'precision', 'recall', 'f1']]
    
    x = np.arange(len(metric_names))
    width = 0.25
    
    ax.bar(x - width, train_values, width, label='Train (Subset)', alpha=0.8, color='purple')
    ax.bar(x, test_values, width, label='Test', alpha=0.8, color='mediumorchid')
    ax.bar(x + width, val_values, width, label='Val', alpha=0.8, color='plum')
    
    ax.set_title(f'Metrics Comparison - k-NN (k={model.n_neighbors})', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/metrics_comparison_knn_multiclass.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return metrics


def save_model(model, filepath='../models/kNN_CICIoT2023_multiclass.pkl'):
    """Salva modello addestrato."""
    print_header("SAVING MODEL")
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Nota: Il file k-NN sarà GRANDE perché contiene tutto il dataset di training!
    print("⚠️  Saving k-NN model... (This file will be large as it stores training data)")
    joblib.dump(model, filepath, compress=3) # Alta compressione per risparmiare spazio
    
    size_mb = os.path.getsize(filepath) / 1024**2
    print(f"💾 Model saved to {filepath}")
    print(f"   Size: {size_mb:.2f} MB")


# Main
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train k-NN on CICIoT2023 Multi-Class')
    parser.add_argument('--data-dir', type=str, default='../data/processed/CICIOT23',
                        help='Directory with preprocessed data')
    parser.add_argument('--output-dir', type=str, default='../docs/knn',
                        help='Output directory for plots')
    parser.add_argument('--neighbors', type=int, default=5,
                        help='Number of neighbors (k)')
    parser.add_argument('--model-path', type=str, default='../models/kNN_CICIoT2023_multiclass.pkl',
                        help='Path to save trained model')
    
    args = parser.parse_args()
    
    print("\n" + "🔗"*40)
    print("k-NEAREST NEIGHBORS TRAINING - MULTI-CLASS".center(80))
    print("CICIoT2023 Dataset".center(80))
    print("🔗"*40)
    
    # Load data
    X_train, X_test, X_val, y_train, y_test, y_val, feature_cols, _ = load_processed_data(args.data_dir)
    
    # Load mapping info
    mapping_info, label_encoder = load_mapping_info(args.data_dir)
    
    # Train
    model = train_knn(X_train, y_train, n_neighbors=args.neighbors)
    
    # Evaluate (Feature importance rimossa perché k-NN non la supporta)
    metrics = evaluate_model(model, X_train, X_test, X_val, 
                             y_train, y_test, y_val,
                             label_encoder, save_dir=args.output_dir)
    
    # Save
    save_model(model, filepath=args.model_path)
    
    print_header("✅ k-NN TRAINING COMPLETE!")
    
    print("\n💡 Note for Production:")
    print("   k-NN models are CPU-intensive during inference.")
    print("   Ensure your hardware can handle the latency (check Recall/Latency trade-off).")