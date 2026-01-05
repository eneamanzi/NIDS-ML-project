"""
Training Random Forest per CICIoT2023 - Multi-Class Classification.

Modello principale del progetto NIDS:
- Classificazione in 8 macro-categorie
- Ensemble di Decision Trees
- Class weighting per gestire sbilanciamento
- Target: Accuracy >95%, Precision >90%, Recall >95%
"""

import pandas as pd
import numpy as np
import joblib
import time
import json
from sklearn.ensemble import RandomForestClassifier
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
    
    Returns:
        X_train, X_test, X_val, y_train, y_test, y_val, y_specific_train, y_specific_test, y_specific_val
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
    print(f"\nClass distribution (train):")
    unique, counts = np.unique(y_train, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"  Class {cls}: {count:>6,} ({count/len(y_train)*100:>5.2f}%)")
    
    return X_train, X_test, X_val, y_train, y_test, y_val, y_specific_train, y_specific_test, y_specific_val, feature_cols


def load_mapping_info(data_dir='../data/processed'):
    """Carica info mapping per interpretare le classi."""
    mapping_path = f"{data_dir}/mapping_info.json"
    with open(mapping_path, 'r') as f:
        mapping_info = json.load(f)
    
    # Carica label encoder per ottenere nomi classi
    encoder_path = f"{data_dir}/label_encoder.pkl"
    label_encoder = joblib.load(encoder_path)
    
    return mapping_info, label_encoder


def train_random_forest(X_train, y_train, n_estimators=100, max_depth=25):
    """
    Train Random Forest classifier per multi-class.
    
    Args:
        X_train: Training features
        y_train: Training labels (macro-categories encoded)
        n_estimators: Numero alberi nella foresta
        max_depth: Profondità massima di ogni albero
    
    Returns:
        Trained model
    """
    print_header("TRAINING RANDOM FOREST (Multi-Class)")
    
    print(f"Hyperparameters:")
    print(f"  n_estimators: {n_estimators}")
    print(f"  max_depth: {max_depth}")
    print(f"  min_samples_split: 10")
    print(f"  min_samples_leaf: 5")
    print(f"  max_features: sqrt")
    print(f"  class_weight: balanced (per gestire sbilanciamento)")
    print(f"  n_jobs: -1 (use all CPU cores)")
    
    start_time = time.time()
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        class_weight='balanced',  # IMPORTANTE: gestisce sbilanciamento
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    print("\nTraining...")
    print("This may take a few minutes depending on data size...")
    model.fit(X_train, y_train)
    
    elapsed = time.time() - start_time
    print(f"\n✅ Training complete in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)!")
    
    return model


def evaluate_model(model, X_train, X_test, X_val, y_train, y_test, y_val, 
                   label_encoder, save_dir='../docs'):
    """
    Valuta modello su train, test e validation set.
    
    Returns:
        Dict con metriche
    """
    print_header("MODEL EVALUATION (Multi-Class)")
    
    # Predizioni
    print("Making predictions...")
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_val_pred = model.predict(X_val)
    
    # Nomi classi
    class_names = label_encoder.classes_
    
    # Metriche (weighted average per multi-class)
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
    print("RESULTS SUMMARY (⭐ MAIN EVALUATION)")
    print("-"*80)
    print(f"{'Metric':<15} {'Train':>12} {'Test':>12} {'Val':>12} {'Status':>8}")
    print("-"*80)
    
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        train_val = metrics['Train'][metric]
        test_val = metrics['Test'][metric]
        val_val = metrics['Val'][metric]
        
        # Check target
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
        print(f"\n⚠️ Possibile overfitting (Train-Test accuracy diff: {acc_diff:.4f})")
    else:
        print(f"\n✅ Buona generalizzazione (Train-Test accuracy diff: {acc_diff:.4f})")
    
    # Classification report (test set)
    print("\n" + "-"*80)
    print("CLASSIFICATION REPORT (Test Set)")
    print("-"*80)
    print(classification_report(y_test, y_test_pred, 
                                target_names=class_names,
                                digits=4,
                                zero_division=0))
    
    # Confusion Matrix (test set)
    cm = confusion_matrix(y_test, y_test_pred)
    
    print("\n" + "-"*80)
    print("CONFUSION MATRIX (Test Set)")
    print("-"*80)
    print("Rows: True labels | Columns: Predicted labels")
    print()
    
    # Header
    print(f"{'':>15}", end="")
    for name in class_names:
        print(f"{name[:10]:>12}", end="")
    print()
    
    # Matrice
    for i, true_name in enumerate(class_names):
        print(f"{true_name[:15]:>15}", end="")
        for j in range(len(class_names)):
            print(f"{cm[i, j]:>12,}", end="")
        print()
    
    # Visualizzazioni
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Confusion Matrix Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - Random Forest\nCICIoT2023 Multi-Class', 
              fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/confusion_matrix_rf_multiclass.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Confusion matrix saved to {save_dir}/confusion_matrix_rf_multiclass.png")
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
    ax.set_title('Metrics Comparison - Random Forest Multi-Class\nCICIoT2023', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    
    # Target lines
    ax.axhline(y=0.95, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.axhline(y=0.90, color='orange', linestyle='--', alpha=0.5, linewidth=1.5)
    
    # Aggiungi valori sopra le barre
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/metrics_comparison_rf_multiclass.png', dpi=150, bbox_inches='tight')
    print(f"📊 Metrics comparison saved to {save_dir}/metrics_comparison_rf_multiclass.png")
    plt.close()
    
    # 3. Per-Class Performance
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Calcola metriche per classe (test set)
    per_class_precision = precision_score(y_test, y_test_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_test, y_test_pred, average=None, zero_division=0)
    per_class_f1 = f1_score(y_test, y_test_pred, average=None, zero_division=0)
    
    x = np.arange(len(class_names))
    width = 0.25
    
    ax.bar(x - width, per_class_precision, width, label='Precision', alpha=0.8, color='steelblue')
    ax.bar(x, per_class_recall, width, label='Recall', alpha=0.8, color='darkgreen')
    ax.bar(x + width, per_class_f1, width, label='F1-Score', alpha=0.8, color='orange')
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Performance - Random Forest\nTest Set', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/per_class_performance_rf_multiclass.png', dpi=150, bbox_inches='tight')
    print(f"📊 Per-class performance saved to {save_dir}/per_class_performance_rf_multiclass.png")
    plt.close()
    
    return metrics, cm


def plot_feature_importance(model, feature_names, n_features=20, save_dir='../docs'):
    """Visualizza feature importance."""
    if feature_names is None or len(feature_names) == 0:
        print("\n⚠️ Feature names not available")
        return
    
    print_header("FEATURE IMPORTANCE ANALYSIS")
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:n_features]
    
    print(f"\nTop {min(15, n_features)} Most Important Features:")
    for i in range(min(15, n_features)):
        idx = indices[i]
        feat_name = feature_names[idx] if idx < len(feature_names) else f"Feature_{idx}"
        print(f"{i+1:2d}. {feat_name:45s} {importances[idx]:.6f}")
    
    # Plot
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
    ax.set_title(f'Top {n_features} Feature Importances - Random Forest\nCICIoT2023 Multi-Class',
                 fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/feature_importance_rf_multiclass.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Feature importance saved to {save_dir}/feature_importance_rf_multiclass.png")
    plt.close()


def save_model(model, filepath='../models/RandomForest_CICIoT2023_multiclass.pkl'):
    """Salva modello addestrato."""
    print_header("SAVING MODEL")
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    joblib.dump(model, filepath)
    size_mb = os.path.getsize(filepath) / 1024**2
    print(f"💾 Model saved to {filepath}")
    print(f"   Size: {size_mb:.2f} MB")

# Main
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Random Forest on CICIoT2023 Multi-Class')
    parser.add_argument('--data-dir', type=str, default='../data/processed/CICIOT23',
                        help='Directory with preprocessed data')
    parser.add_argument('--output-dir', type=str, default='../docs/random_forest',
                        help='Output directory for plots')
    parser.add_argument('--n-estimators', type=int, default=100,
                        help='Number of trees in the forest')
    parser.add_argument('--max-depth', type=int, default=25,
                        help='Max depth of each tree')
    parser.add_argument('--model-path', type=str, default='../models/RandomForest_CICIoT2023_multiclass.pkl',
                        help='Path to save trained model')
    
    args = parser.parse_args()
    
    print("\n" + "🌲"*40)
    print("RANDOM FOREST TRAINING - MULTI-CLASS".center(80))
    print("CICIoT2023 Dataset".center(80))
    print("🌲"*40)
    
    # Load data
    X_train, X_test, X_val, y_train, y_test, y_val, \
        y_specific_train, y_specific_test, y_specific_val, feature_cols = load_processed_data(args.data_dir)
    
    # Load mapping info
    mapping_info, label_encoder = load_mapping_info(args.data_dir)
    
    # Train
    model = train_random_forest(X_train, y_train, 
                                n_estimators=args.n_estimators,
                                max_depth=args.max_depth)
    
    # Evaluate
    metrics, cm = evaluate_model(model, X_train, X_test, X_val, 
                                 y_train, y_test, y_val,
                                 label_encoder, save_dir=args.output_dir)
    
    # Feature importance
    plot_feature_importance(model, feature_cols, n_features=20, save_dir=args.output_dir)
    
    # Save
    save_model(model, filepath=args.model_path)
    
    # Final summary
    print_header("🎉 RANDOM FOREST TRAINING COMPLETE!")
    
    meets_requirements = (
        metrics['Test']['accuracy'] >= 0.95 and 
        metrics['Test']['precision'] >= 0.90 and 
        metrics['Test']['recall'] >= 0.95
    )
    
    if meets_requirements:
        print("✅ MODEL MEETS ALL PROJECT REQUIREMENTS!")
        print("   Accuracy:  ≥0.95 ✓")
        print("   Precision: ≥0.90 ✓")
        print("   Recall:    ≥0.95 ✓")
    else:
        print("⚠️ Model does not meet all requirements yet.")
        print("   Consider:")
        print("   - Increasing n_estimators (es. 200)")
        print("   - Tuning max_depth")
        print("   - Using SMOTE for class balancing")
        print("   - Feature engineering")
    
    print("\n💡 Next steps:")
    print(f"   1. Review plots in {args.output_dir}")
    print("   2. Compare with Decision Tree results")
    print("   3. Implement real-time detection system")