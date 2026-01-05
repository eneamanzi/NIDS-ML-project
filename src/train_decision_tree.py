"""
Training Decision Tree - Primo modello baseline.
"""

import joblib
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
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


def train_decision_tree(X_train, y_train, max_depth=10):
    """
    Train Decision Tree classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        max_depth: Profondità massima albero (evita overfitting)
    
    Returns:
        Trained model
    """
    print(f"Training Decision Tree (max_depth={max_depth})...")
    
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=20,  # Min campioni per splittare nodo
        min_samples_leaf=10,    # Min campioni per foglia
        random_state=42
    )
    
    model.fit(X_train, y_train)
    print("✅ Training complete!\n")
    
    return model


def evaluate_model(model, X_test, y_test):
    """
    Valuta modello su test set.
    
    Returns:
        Dict con metriche
    """
    print("Evaluating model on test set...")
    
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }
    
    print("\n" + "="*60)
    print("RESULTS - Decision Tree")
    print("="*60)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}")
    print("="*60)
    
    # Classification report dettagliato
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Anomaly']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    print(f"\nTrue Negatives:  {cm[0,0]:>6}  (Normal classificati Normal)")
    print(f"False Positives: {cm[0,1]:>6}  (Normal classificati Anomaly) ⚠️")
    print(f"False Negatives: {cm[1,0]:>6}  (Anomaly classificati Normal) ❌ CRITICO")
    print(f"True Positives:  {cm[1,1]:>6}  (Anomaly classificati Anomaly)")
    
    # Visualizza confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'])
    plt.title('Confusion Matrix - Decision Tree')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('../docs/confusion_matrix_dt.png', dpi=150)
    print(f"\n📊 Confusion matrix saved to ../docs/confusion_matrix_dt.png")
    
    return metrics, cm


def save_model(model, filepath='../models/DecisionTree.pkl'):
    """Salva modello addestrato."""
    joblib.dump(model, filepath)
    print(f"\n💾 Model saved to {filepath}")


# Main
if __name__ == '__main__':
    # Load data
    X_train, X_test, y_train, y_test = load_preprocessed_data()
    
    # Train
    model = train_decision_tree(X_train, y_train, max_depth=10)
    
    # Evaluate
    metrics, cm = evaluate_model(model, X_test, y_test)
    
    # Save
    save_model(model)
    
    print("\n✅ Done! Your first ML model is ready.")
