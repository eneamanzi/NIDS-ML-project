"""
Training Random Forest - Modello principale del progetto.
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
    print("Loading preprocessed data...")
    X_train = joblib.load(f'{data_dir}/X_train.pkl')
    X_test = joblib.load(f'{data_dir}/X_test.pkl')
    y_train = joblib.load(f'{data_dir}/y_train.pkl')
    y_test = joblib.load(f'{data_dir}/y_test.pkl')
    
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}\n")
    
    return X_train, X_test, y_train, y_test


def train_random_forest(X_train, y_train, n_estimators=100):
    """
    Train Random Forest classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        n_estimators: Numero alberi (più = meglio ma più lento)
    
    Returns:
        Trained model
    """
    print(f"Training Random Forest ({n_estimators} trees)...")
    print("This may take 1-2 minutes...")
    
    start_time = time.time()
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=20,              # Più profondo di Decision Tree singolo
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',        # Ogni albero usa sqrt(41) ≈ 6 features
        n_jobs=-1,                  # Usa tutti i core CPU
        random_state=42,
        verbose=1                   # Mostra progress
    )
    
    model.fit(X_train, y_train)
    
    elapsed = time.time() - start_time
    print(f"✅ Training complete in {elapsed:.1f} seconds!\n")
    
    return model


def evaluate_model(model, X_test, y_test):
    """Valuta modello su test set."""
    print("Evaluating model on test set...")
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilità classe Anomaly
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc_roc': roc_auc_score(y_test, y_pred_proba)
    }
    
    print("\n" + "="*60)
    print("RESULTS - Random Forest")
    print("="*60)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}")
    print(f"AUC-ROC:   {metrics['auc_roc']:.4f}")
    print("="*60)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Anomaly']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    print(f"\nTrue Negatives:  {cm[0,0]:>6}")
    print(f"False Positives: {cm[0,1]:>6}  ⚠️")
    print(f"False Negatives: {cm[1,0]:>6}  ❌ CRITICO")
    print(f"True Positives:  {cm[1,1]:>6}")
    
    # Visualizza confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'])
    plt.title('Confusion Matrix - Random Forest')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('../docs/confusion_matrix_rf.png', dpi=150)
    print(f"\n📊 Confusion matrix saved to ../docs/confusion_matrix_rf.png")
    
    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'Random Forest (AUC = {metrics["auc_roc"]:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title('ROC Curve - Random Forest')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('../docs/roc_curve_rf.png', dpi=150)
    print(f"📊 ROC curve saved to ../docs/roc_curve_rf.png")
    
    return metrics, cm


def plot_feature_importance(model, n_features=20):
    """
    Visualizza le features più importanti.
    
    Spiega quali caratteristiche del traffico sono più predittive.
    """
    from src.preprocessing import COLUMN_NAMES
    
    # Escludi label e difficulty
    feature_names = [col for col in COLUMN_NAMES if col not in ['label', 'difficulty']]
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:n_features]
    
    plt.figure(figsize=(12, 8))
    plt.title(f'Top {n_features} Feature Importances - Random Forest')
    plt.bar(range(n_features), importances[indices])
    plt.xticks(range(n_features), [feature_names[i] for i in indices], rotation=90)
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.savefig('../docs/feature_importance_rf.png', dpi=150)
    print(f"📊 Feature importance saved to ../docs/feature_importance_rf.png")
    
    print("\nTop 10 Most Important Features:")
    for i in range(min(10, n_features)):
        idx = indices[i]
        print(f"{i+1:2d}. {feature_names[idx]:30s} {importances[idx]:.4f}")


def save_model(model, filepath='../models/RandomForest.pkl'):
    """Salva modello addestrato."""
    joblib.dump(model, filepath)
    print(f"\n💾 Model saved to {filepath}")
    
    # Salva anche come "best_model" (per .gitignore)
    joblib.dump(model, '../models/best_model.pkl')
    print(f"💾 Also saved as ../models/best_model.pkl")


# Main
if __name__ == '__main__':
    # Load data
    X_train, X_test, y_train, y_test = load_preprocessed_data()
    
    # Train
    model = train_random_forest(X_train, y_train, n_estimators=100)
    
    # Evaluate
    metrics, cm = evaluate_model(model, X_test, y_test)
    
    # Feature importance
    plot_feature_importance(model)
    
    # Save
    save_model(model)
    
    print("\n✅ Random Forest training complete!")
    print("This is now your main model for the NIDS.")
