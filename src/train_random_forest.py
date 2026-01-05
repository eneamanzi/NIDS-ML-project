# """
# Training Random Forest - Modello principale del progetto.
# """

# import joblib
# import numpy as np
# import time
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import (
#     accuracy_score, precision_score, recall_score, f1_score,
#     confusion_matrix, classification_report, roc_auc_score, roc_curve
# )
# import matplotlib.pyplot as plt
# import seaborn as sns


# def load_preprocessed_data(data_dir='../data/processed'):
#     """Carica dati preprocessati."""
#     print("Loading preprocessed data...")
#     X_train = joblib.load(f'{data_dir}/X_train.pkl')
#     X_test = joblib.load(f'{data_dir}/X_test.pkl')
#     y_train = joblib.load(f'{data_dir}/y_train.pkl')
#     y_test = joblib.load(f'{data_dir}/y_test.pkl')
    
#     print(f"X_train: {X_train.shape}")
#     print(f"X_test: {X_test.shape}\n")
    
#     return X_train, X_test, y_train, y_test


# def train_random_forest(X_train, y_train, n_estimators=100):
#     """
#     Train Random Forest classifier.
    
#     Args:
#         X_train: Training features
#         y_train: Training labels
#         n_estimators: Numero alberi (più = meglio ma più lento)
    
#     Returns:
#         Trained model
#     """
#     print(f"Training Random Forest ({n_estimators} trees)...")
#     print("This may take 1-2 minutes...")
    
#     start_time = time.time()
    
#     model = RandomForestClassifier(
#         n_estimators=n_estimators,
#         max_depth=20,              # Più profondo di Decision Tree singolo
#         min_samples_split=10,
#         min_samples_leaf=5,
#         max_features='sqrt',        # Ogni albero usa sqrt(41) ≈ 6 features
#         n_jobs=-1,                  # Usa tutti i core CPU
#         random_state=42,
#         verbose=1                   # Mostra progress
#     )
    
#     model.fit(X_train, y_train)
    
#     elapsed = time.time() - start_time
#     print(f"✅ Training complete in {elapsed:.1f} seconds!\n")
    
#     return model


# def evaluate_model(model, X_test, y_test):
#     """Valuta modello su test set."""
#     print("Evaluating model on test set...")
    
#     y_pred = model.predict(X_test)
#     y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilità classe Anomaly
    
#     metrics = {
#         'accuracy': accuracy_score(y_test, y_pred),
#         'precision': precision_score(y_test, y_pred),
#         'recall': recall_score(y_test, y_pred),
#         'f1': f1_score(y_test, y_pred),
#         'auc_roc': roc_auc_score(y_test, y_pred_proba)
#     }
    
#     print("\n" + "="*60)
#     print("RESULTS - Random Forest")
#     print("="*60)
#     print(f"Accuracy:  {metrics['accuracy']:.4f}")
#     print(f"Precision: {metrics['precision']:.4f}")
#     print(f"Recall:    {metrics['recall']:.4f}")
#     print(f"F1-Score:  {metrics['f1']:.4f}")
#     print(f"AUC-ROC:   {metrics['auc_roc']:.4f}")
#     print("="*60)
    
#     # Classification report
#     print("\nClassification Report:")
#     print(classification_report(y_test, y_pred, target_names=['Normal', 'Anomaly']))
    
#     # Confusion Matrix
#     cm = confusion_matrix(y_test, y_pred)
#     print("\nConfusion Matrix:")
#     print(cm)
#     print(f"\nTrue Negatives:  {cm[0,0]:>6}")
#     print(f"False Positives: {cm[0,1]:>6}  ⚠️")
#     print(f"False Negatives: {cm[1,0]:>6}  ❌ CRITICO")
#     print(f"True Positives:  {cm[1,1]:>6}")
    
#     # Visualizza confusion matrix
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
#                 xticklabels=['Normal', 'Anomaly'],
#                 yticklabels=['Normal', 'Anomaly'])
#     plt.title('Confusion Matrix - Random Forest')
#     plt.ylabel('True Label')
#     plt.xlabel('Predicted Label')
#     plt.tight_layout()
#     plt.savefig('../docs/confusion_matrix_rf.png', dpi=150)
#     print(f"\n📊 Confusion matrix saved to ../docs/confusion_matrix_rf.png")
    
#     # ROC Curve
#     fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
#     plt.figure(figsize=(8, 6))
#     plt.plot(fpr, tpr, linewidth=2, label=f'Random Forest (AUC = {metrics["auc_roc"]:.4f})')
#     plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate (Recall)')
#     plt.title('ROC Curve - Random Forest')
#     plt.legend()
#     plt.grid(alpha=0.3)
#     plt.tight_layout()
#     plt.savefig('../docs/roc_curve_rf.png', dpi=150)
#     print(f"📊 ROC curve saved to ../docs/roc_curve_rf.png")
    
#     return metrics, cm


# def plot_feature_importance(model, n_features=20):
#     """
#     Visualizza le features più importanti.
    
#     Spiega quali caratteristiche del traffico sono più predittive.
#     """
#     from src.preprocessing import COLUMN_NAMES
    
#     # Escludi label e difficulty
#     feature_names = [col for col in COLUMN_NAMES if col not in ['label', 'difficulty']]
    
#     importances = model.feature_importances_
#     indices = np.argsort(importances)[::-1][:n_features]
    
#     plt.figure(figsize=(12, 8))
#     plt.title(f'Top {n_features} Feature Importances - Random Forest')
#     plt.bar(range(n_features), importances[indices])
#     plt.xticks(range(n_features), [feature_names[i] for i in indices], rotation=90)
#     plt.ylabel('Importance')
#     plt.tight_layout()
#     plt.savefig('../docs/feature_importance_rf.png', dpi=150)
#     print(f"📊 Feature importance saved to ../docs/feature_importance_rf.png")
    
#     print("\nTop 10 Most Important Features:")
#     for i in range(min(10, n_features)):
#         idx = indices[i]
#         print(f"{i+1:2d}. {feature_names[idx]:30s} {importances[idx]:.4f}")


# def save_model(model, filepath='../models/RandomForest.pkl'):
#     """Salva modello addestrato."""
#     joblib.dump(model, filepath)
#     print(f"\n💾 Model saved to {filepath}")
    
#     # Salva anche come "best_model" (per .gitignore)
#     joblib.dump(model, '../models/best_model.pkl')
#     print(f"💾 Also saved as ../models/best_model.pkl")


# # Main
# if __name__ == '__main__':
#     # Load data
#     X_train, X_test, y_train, y_test = load_preprocessed_data()
    
#     # Train
#     model = train_random_forest(X_train, y_train, n_estimators=100)
    
#     # Evaluate
#     metrics, cm = evaluate_model(model, X_test, y_test)
    
#     # Feature importance
#     plot_feature_importance(model)
    
#     # Save
#     save_model(model)
    
#     print("\n✅ Random Forest training complete!")
#     print("This is now your main model for the NIDS.")


"""
Training Random Forest per CICIoT2023 dataset.

Modello principale del progetto NIDS:
- Ensemble di Decision Trees
- Classificazione binaria: Normal (0) vs Anomaly (1)
- Target: Accuracy >95%, Precision >90%, Recall >95%
- Migliore di Decision Tree singolo per robustezza e accuracy
"""

import joblib
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns


def load_preprocessed_data(data_dir='../data/processed'):
    """Carica dati preprocessati."""
    print("="*80)
    print("LOADING PREPROCESSED DATA")
    print("="*80)
    
    X_train = joblib.load(f'{data_dir}/X_train.pkl')
    X_test = joblib.load(f'{data_dir}/X_test.pkl')
    y_train = joblib.load(f'{data_dir}/y_train.pkl')
    y_test = joblib.load(f'{data_dir}/y_test.pkl')
    
    try:
        feature_names = joblib.load(f'{data_dir}/feature_names.pkl')
    except:
        feature_names = None
        
    try:
        dataset_info = joblib.load(f'{data_dir}/dataset_info.pkl')
        print("\nDataset Info:")
        for key, value in dataset_info.items():
            print(f"  {key}: {value}")
    except:
        pass
    
    print(f"\nX_train: {X_train.shape}")
    print(f"X_test:  {X_test.shape}")
    print(f"y_train: {y_train.shape} (Anomaly rate: {y_train.mean():.2%})")
    print(f"y_test:  {y_test.shape} (Anomaly rate: {y_test.mean():.2%})")
    
    return X_train, X_test, y_train, y_test, feature_names


def train_random_forest(X_train, y_train, n_estimators=100, max_depth=20):
    """
    Train Random Forest classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        n_estimators: Numero alberi nella foresta
        max_depth: Profondità massima di ogni albero
    
    Returns:
        Trained model
    """
    print("\n" + "="*80)
    print("TRAINING RANDOM FOREST")
    print("="*80)
    
    print(f"Hyperparameters:")
    print(f"  n_estimators: {n_estimators}")
    print(f"  max_depth: {max_depth}")
    print(f"  min_samples_split: 10")
    print(f"  min_samples_leaf: 5")
    print(f"  max_features: sqrt")
    print(f"  n_jobs: -1 (use all CPU cores)")
    
    start_time = time.time()
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        n_jobs=-1,
        random_state=42,
        verbose=1  # Mostra progress
    )
    
    print("\nTraining...")
    print("This may take a few minutes depending on data size...")
    model.fit(X_train, y_train)
    
    elapsed = time.time() - start_time
    print(f"\n✅ Training complete in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)!")
    
    return model


def evaluate_model(model, X_train, X_test, y_train, y_test, save_dir='../docs'):
    """
    Valuta modello su train e test set.
    
    Returns:
        Dict con metriche
    """
    print("\n" + "="*80)
    print("MODEL EVALUATION")
    print("="*80)
    
    # Predizioni
    print("Making predictions on train set...")
    y_train_pred = model.predict(X_train)
    
    print("Making predictions on test set...")
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # Metriche
    train_metrics = {
        'accuracy': accuracy_score(y_train, y_train_pred),
        'precision': precision_score(y_train, y_train_pred),
        'recall': recall_score(y_train, y_train_pred),
        'f1': f1_score(y_train, y_train_pred)
    }
    
    test_metrics = {
        'accuracy': accuracy_score(y_test, y_test_pred),
        'precision': precision_score(y_test, y_test_pred),
        'recall': recall_score(y_test, y_test_pred),
        'f1': f1_score(y_test, y_test_pred),
        'auc_roc': roc_auc_score(y_test, y_test_proba)
    }
    
    # Stampa risultati
    print("\n" + "-"*80)
    print("TRAIN SET RESULTS")
    print("-"*80)
    print(f"Accuracy:  {train_metrics['accuracy']:.4f}")
    print(f"Precision: {train_metrics['precision']:.4f}")
    print(f"Recall:    {train_metrics['recall']:.4f}")
    print(f"F1-Score:  {train_metrics['f1']:.4f}")
    
    print("\n" + "-"*80)
    print("TEST SET RESULTS (⭐ MAIN EVALUATION)")
    print("-"*80)
    print(f"Accuracy:  {test_metrics['accuracy']:.4f} {'✅' if test_metrics['accuracy'] >= 0.95 else '❌'} (Target: >0.95)")
    print(f"Precision: {test_metrics['precision']:.4f} {'✅' if test_metrics['precision'] >= 0.90 else '❌'} (Target: >0.90)")
    print(f"Recall:    {test_metrics['recall']:.4f} {'✅' if test_metrics['recall'] >= 0.95 else '❌'} (Target: >0.95)")
    print(f"F1-Score:  {test_metrics['f1']:.4f}")
    print(f"AUC-ROC:   {test_metrics['auc_roc']:.4f}")
    
    # Overfitting check
    acc_diff = train_metrics['accuracy'] - test_metrics['accuracy']
    if acc_diff > 0.05:
        print(f"\n⚠️ Possibile overfitting (Train-Test accuracy diff: {acc_diff:.4f})")
    else:
        print(f"\n✅ Buona generalizzazione (Train-Test accuracy diff: {acc_diff:.4f})")
    
    # Classification report
    print("\n" + "-"*80)
    print("CLASSIFICATION REPORT (Test Set)")
    print("-"*80)
    print(classification_report(y_test, y_test_pred, 
                                target_names=['Normal', 'Anomaly'],
                                digits=4))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print("\n" + "-"*80)
    print("CONFUSION MATRIX (Test Set)")
    print("-"*80)
    print(f"True Negatives:  {tn:>8,}  (Normal → Normal)")
    print(f"False Positives: {fp:>8,}  (Normal → Anomaly) ⚠️")
    print(f"False Negatives: {fn:>8,}  (Anomaly → Normal) ❌ CRITICO")
    print(f"True Positives:  {tp:>8,}  (Anomaly → Anomaly)")
    print(f"\nFalse Positive Rate: {fp / (fp + tn):.2%}")
    print(f"False Negative Rate: {fn / (fn + tp):.2%} (dovrebbe essere <5%)")
    
    # Visualizzazioni
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Confusion Matrix
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=',d', cmap='Greens',
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'],
                ax=ax, cbar_kws={'label': 'Count'})
    ax.set_title('Confusion Matrix - Random Forest\nCICIoT2023 Dataset', 
                 fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/confusion_matrix_rf_ciciot2023.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Confusion matrix saved to {save_dir}/confusion_matrix_rf_ciciot2023.png")
    plt.close()
    
    # 2. ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(fpr, tpr, linewidth=2, 
            label=f'Random Forest (AUC = {test_metrics["auc_roc"]:.4f})', 
            color='darkgreen')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate (Recall)', fontsize=12)
    ax.set_title('ROC Curve - Random Forest\nCICIoT2023 Dataset', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/roc_curve_rf_ciciot2023.png', dpi=150, bbox_inches='tight')
    print(f"📊 ROC curve saved to {save_dir}/roc_curve_rf_ciciot2023.png")
    plt.close()
    
    # 3. Metrics comparison
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    train_values = [train_metrics[k] for k in ['accuracy', 'precision', 'recall', 'f1']]
    test_values = [test_metrics[k] for k in ['accuracy', 'precision', 'recall', 'f1']]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, train_values, width, label='Train', alpha=0.8, color='steelblue')
    bars2 = ax.bar(x + width/2, test_values, width, label='Test', alpha=0.8, color='darkgreen')
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Metrics Comparison - Random Forest\nCICIoT2023 Dataset', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    
    # Target lines
    ax.axhline(y=0.95, color='red', linestyle='--', alpha=0.5, linewidth=1.5, 
               label='Target (Acc/Recall ≥0.95)')
    ax.axhline(y=0.90, color='orange', linestyle='--', alpha=0.5, linewidth=1.5, 
               label='Target (Precision ≥0.90)')
    ax.legend(fontsize=9, loc='lower right')
    
    # Aggiungi valori sopra le barre
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/metrics_comparison_rf_ciciot2023.png', dpi=150, bbox_inches='tight')
    print(f"📊 Metrics comparison saved to {save_dir}/metrics_comparison_rf_ciciot2023.png")
    plt.close()
    
    return test_metrics, cm


def plot_feature_importance(model, feature_names, n_features=20, save_dir='../docs'):
    """
    Visualizza le features più importanti.
    """
    if feature_names is None:
        print("\n⚠️ Feature names not available, skipping feature importance plot")
        return
    
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*80)
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:n_features]
    
    print(f"\nTop {min(15, n_features)} Most Important Features:")
    for i in range(min(15, n_features)):
        idx = indices[i]
        print(f"{i+1:2d}. {feature_names[idx]:45s} {importances[idx]:.6f}")
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    colors = plt.cm.Greens(np.linspace(0.4, 0.8, n_features))
    ax.barh(range(n_features), importances[indices][::-1], alpha=0.9, color=colors[::-1])
    ax.set_yticks(range(n_features))
    ax.set_yticklabels([feature_names[i] for i in indices[::-1]], fontsize=9)
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(f'Top {n_features} Feature Importances - Random Forest\nCICIoT2023 Dataset',
                 fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/feature_importance_rf_ciciot2023.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Feature importance saved to {save_dir}/feature_importance_rf_ciciot2023.png")
    plt.close()


def save_model(model, filepath='../models/RandomForest_CICIoT2023.pkl'):
    """Salva modello addestrato."""
    print("\n" + "="*80)
    print("SAVING MODEL")
    print("="*80)
    
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    joblib.dump(model, filepath)
    print(f"💾 Model saved to {filepath}")
    
    # Salva anche come best_model (per convenzione)
    best_model_path = '../models/best_model_ciciot2023.pkl'
    joblib.dump(model, best_model_path)
    print(f"💾 Also saved as {best_model_path}")
    
    # Info modello
    model_size = os.path.getsize(filepath) / 1024**2
    print(f"\nModel size: {model_size:.2f} MB")


# Main
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Random Forest on CICIoT2023')
    parser.add_argument('--data-dir', type=str, default='../data/processed',
                        help='Directory with preprocessed data')
    parser.add_argument('--output-dir', type=str, default='../docs',
                        help='Output directory for plots')
    parser.add_argument('--n-estimators', type=int, default=100,
                        help='Number of trees in the forest')
    parser.add_argument('--max-depth', type=int, default=20,
                        help='Max depth of each tree')
    parser.add_argument('--model-path', type=str, default='../models/RandomForest_CICIoT2023.pkl',
                        help='Path to save trained model')
    
    args = parser.parse_args()
    
    print("\n" + "🌲"*40)
    print("RANDOM FOREST TRAINING - CICIoT2023 NIDS")
    print("🌲"*40)
    
    # Load data
    X_train, X_test, y_train, y_test, feature_names = load_preprocessed_data(args.data_dir)
    
    # Train
    model = train_random_forest(X_train, y_train, 
                                n_estimators=args.n_estimators,
                                max_depth=args.max_depth)
    
    # Evaluate
    metrics, cm = evaluate_model(model, X_train, X_test, y_train, y_test, 
                                 save_dir=args.output_dir)
    
    # Feature importance
    plot_feature_importance(model, feature_names, n_features=20, save_dir=args.output_dir)
    
    # Save
    save_model(model, filepath=args.model_path)
    
    # Final summary
    print("\n" + "="*80)
    print("🎉 RANDOM FOREST TRAINING COMPLETE!")
    print("="*80)
    
    meets_requirements = (
        metrics['accuracy'] >= 0.95 and 
        metrics['precision'] >= 0.90 and 
        metrics['recall'] >= 0.95
    )
    
    if meets_requirements:
        print("\n✅ MODEL MEETS ALL PROJECT REQUIREMENTS!")
        print("   Accuracy:  ≥0.95 ✓")
        print("   Precision: ≥0.90 ✓")
        print("   Recall:    ≥0.95 ✓")
    else:
        print("\n⚠️ Model does not meet all requirements yet.")
        print("   Consider:")
        print("   - Increasing n_estimators")
        print("   - Tuning hyperparameters (max_depth, min_samples_split)")
        print("   - Handling class imbalance (SMOTE)")
        print("   - Feature engineering")
    
    print("\n💡 Next steps:")
    print("   1. Review plots in", args.output_dir)
    print("   2. Compare with Decision Tree results")
    print("   3. Train k-NN and SVM models")
    print("   4. Implement real-time detection system")