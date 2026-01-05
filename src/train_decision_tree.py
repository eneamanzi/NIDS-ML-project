# """
# Training Decision Tree - Primo modello baseline.
# """

# import joblib
# import numpy as np
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import (
#     accuracy_score, precision_score, recall_score, f1_score,
#     confusion_matrix, classification_report
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


# def train_decision_tree(X_train, y_train, max_depth=10):
#     """
#     Train Decision Tree classifier.
    
#     Args:
#         X_train: Training features
#         y_train: Training labels
#         max_depth: Profondità massima albero (evita overfitting)
    
#     Returns:
#         Trained model
#     """
#     print(f"Training Decision Tree (max_depth={max_depth})...")
    
#     model = DecisionTreeClassifier(
#         max_depth=max_depth,
#         min_samples_split=20,  # Min campioni per splittare nodo
#         min_samples_leaf=10,    # Min campioni per foglia
#         random_state=42
#     )
    
#     model.fit(X_train, y_train)
#     print("✅ Training complete!\n")
    
#     return model


# def evaluate_model(model, X_test, y_test):
#     """
#     Valuta modello su test set.
    
#     Returns:
#         Dict con metriche
#     """
#     print("Evaluating model on test set...")
    
#     y_pred = model.predict(X_test)
    
#     metrics = {
#         'accuracy': accuracy_score(y_test, y_pred),
#         'precision': precision_score(y_test, y_pred),
#         'recall': recall_score(y_test, y_pred),
#         'f1': f1_score(y_test, y_pred)
#     }
    
#     print("\n" + "="*60)
#     print("RESULTS - Decision Tree")
#     print("="*60)
#     print(f"Accuracy:  {metrics['accuracy']:.4f}")
#     print(f"Precision: {metrics['precision']:.4f}")
#     print(f"Recall:    {metrics['recall']:.4f}")
#     print(f"F1-Score:  {metrics['f1']:.4f}")
#     print("="*60)
    
#     # Classification report dettagliato
#     print("\nClassification Report:")
#     print(classification_report(y_test, y_pred, target_names=['Normal', 'Anomaly']))
    
#     # Confusion Matrix
#     cm = confusion_matrix(y_test, y_pred)
#     print("\nConfusion Matrix:")
#     print(cm)
#     print(f"\nTrue Negatives:  {cm[0,0]:>6}  (Normal classificati Normal)")
#     print(f"False Positives: {cm[0,1]:>6}  (Normal classificati Anomaly) ⚠️")
#     print(f"False Negatives: {cm[1,0]:>6}  (Anomaly classificati Normal) ❌ CRITICO")
#     print(f"True Positives:  {cm[1,1]:>6}  (Anomaly classificati Anomaly)")
    
#     # Visualizza confusion matrix
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=['Normal', 'Anomaly'],
#                 yticklabels=['Normal', 'Anomaly'])
#     plt.title('Confusion Matrix - Decision Tree')
#     plt.ylabel('True Label')
#     plt.xlabel('Predicted Label')
#     plt.tight_layout()
#     plt.savefig('../docs/confusion_matrix_dt.png', dpi=150)
#     print(f"\n📊 Confusion matrix saved to ../docs/confusion_matrix_dt.png")
    
#     return metrics, cm


# def save_model(model, filepath='../models/DecisionTree.pkl'):
#     """Salva modello addestrato."""
#     joblib.dump(model, filepath)
#     print(f"\n💾 Model saved to {filepath}")


# # Main
# if __name__ == '__main__':
#     # Load data
#     X_train, X_test, y_train, y_test = load_preprocessed_data()
    
#     # Train
#     model = train_decision_tree(X_train, y_train, max_depth=10)
    
#     # Evaluate
#     metrics, cm = evaluate_model(model, X_test, y_test)
    
#     # Save
#     save_model(model)
    
#     print("\n✅ Done! Your first ML model is ready.")


"""
Training Decision Tree per CICIoT2023 dataset.

Modello baseline per:
- Network Intrusion Detection System (NIDS)
- Classificazione binaria: Normal (0) vs Anomaly (1)
- Target: Accuracy >95%, Precision >90%, Recall >95%
"""

import joblib
import numpy as np
import time
from sklearn.tree import DecisionTreeClassifier
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


def train_decision_tree(X_train, y_train, max_depth=15):
    """
    Train Decision Tree classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        max_depth: Profondità massima albero
    
    Returns:
        Trained model
    """
    print("\n" + "="*80)
    print("TRAINING DECISION TREE")
    print("="*80)
    
    print(f"Hyperparameters:")
    print(f"  max_depth: {max_depth}")
    print(f"  min_samples_split: 20")
    print(f"  min_samples_leaf: 10")
    print(f"  criterion: gini")
    
    start_time = time.time()
    
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=20,
        min_samples_leaf=10,
        criterion='gini',
        random_state=42
    )
    
    print("\nTraining...")
    model.fit(X_train, y_train)
    
    elapsed = time.time() - start_time
    print(f"✅ Training complete in {elapsed:.2f} seconds!")
    
    # Info albero
    print(f"\nTree depth: {model.get_depth()}")
    print(f"Number of leaves: {model.get_n_leaves()}")
    
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
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Probabilità (per ROC curve)
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
    print("TEST SET RESULTS")
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
    print("\nFalse Positive Rate: {:.2%}".format(fp / (fp + tn)))
    print("False Negative Rate: {:.2%} (dovrebbe essere <5%)".format(fn / (fn + tp)))
    
    # Visualizzazioni
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Confusion Matrix
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=',d', cmap='Blues',
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'],
                ax=ax)
    ax.set_title('Confusion Matrix - Decision Tree\nCICIoT2023 Dataset', 
                 fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/confusion_matrix_dt_ciciot2023.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Confusion matrix saved to {save_dir}/confusion_matrix_dt_ciciot2023.png")
    plt.close()
    
    # 2. ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(fpr, tpr, linewidth=2, label=f'Decision Tree (AUC = {test_metrics["auc_roc"]:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate (Recall)', fontsize=12)
    ax.set_title('ROC Curve - Decision Tree\nCICIoT2023 Dataset', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/roc_curve_dt_ciciot2023.png', dpi=150, bbox_inches='tight')
    print(f"📊 ROC curve saved to {save_dir}/roc_curve_dt_ciciot2023.png")
    plt.close()
    
    # 3. Metrics comparison
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    train_values = [train_metrics[k] for k in ['accuracy', 'precision', 'recall', 'f1']]
    test_values = [test_metrics[k] for k in ['accuracy', 'precision', 'recall', 'f1']]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    ax.bar(x - width/2, train_values, width, label='Train', alpha=0.8)
    ax.bar(x + width/2, test_values, width, label='Test', alpha=0.8)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Metrics Comparison - Decision Tree\nCICIoT2023 Dataset', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend(fontsize=10)
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    
    # Aggiungi target lines
    ax.axhline(y=0.95, color='r', linestyle='--', alpha=0.5, label='Target (Acc/Recall)')
    ax.axhline(y=0.90, color='orange', linestyle='--', alpha=0.5, label='Target (Precision)')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/metrics_comparison_dt_ciciot2023.png', dpi=150, bbox_inches='tight')
    print(f"📊 Metrics comparison saved to {save_dir}/metrics_comparison_dt_ciciot2023.png")
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
    
    print(f"\nTop {min(10, n_features)} Most Important Features:")
    for i in range(min(10, n_features)):
        idx = indices[i]
        print(f"{i+1:2d}. {feature_names[idx]:40s} {importances[idx]:.6f}")
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.barh(range(n_features), importances[indices][::-1], alpha=0.8)
    ax.set_yticks(range(n_features))
    ax.set_yticklabels([feature_names[i] for i in indices[::-1]], fontsize=9)
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(f'Top {n_features} Feature Importances - Decision Tree\nCICIoT2023 Dataset',
                 fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/feature_importance_dt_ciciot2023.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Feature importance saved to {save_dir}/feature_importance_dt_ciciot2023.png")
    plt.close()


def save_model(model, filepath='../models/DecisionTree_CICIoT2023.pkl'):
    """Salva modello addestrato."""
    print("\n" + "="*80)
    print("SAVING MODEL")
    print("="*80)
    
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    joblib.dump(model, filepath)
    print(f"💾 Model saved to {filepath}")


# Main
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Decision Tree on CICIoT2023')
    parser.add_argument('--data-dir', type=str, default='../data/processed',
                        help='Directory with preprocessed data')
    parser.add_argument('--output-dir', type=str, default='../docs',
                        help='Output directory for plots')
    parser.add_argument('--max-depth', type=int, default=15,
                        help='Max depth of decision tree')
    parser.add_argument('--model-path', type=str, default='../models/DecisionTree_CICIoT2023.pkl',
                        help='Path to save trained model')
    
    args = parser.parse_args()
    
    # Load data
    X_train, X_test, y_train, y_test, feature_names = load_preprocessed_data(args.data_dir)
    
    # Train
    model = train_decision_tree(X_train, y_train, max_depth=args.max_depth)
    
    # Evaluate
    metrics, cm = evaluate_model(model, X_train, X_test, y_train, y_test, 
                                 save_dir=args.output_dir)
    
    # Feature importance
    plot_feature_importance(model, feature_names, save_dir=args.output_dir)
    
    # Save
    save_model(model, filepath=args.model_path)
    
    print("\n" + "="*80)
    print("✅ DECISION TREE TRAINING COMPLETE!")
    print("="*80)
    print("\n💡 Next steps:")
    print("   1. Review plots in", args.output_dir)
    print("   2. Train Random Forest: python src/train_random_forest_ciciot2023.py")