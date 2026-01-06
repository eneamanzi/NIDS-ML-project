# """
# Model Comparison & Benchmarking per CICIoT2023.

# Confronta i 4 algoritmi (DT, RF, k-NN, SVM) su:
# 1. Metriche di Classificazione (Accuracy, Precision, Recall, F1)
# 2. Performance Operative (Latenza di inferenza, Dimensione modello)

# Output:
# - Report CSV completo
# - Grafici comparativi (Bar charts + Scatter plot Accuracy/Latency)
# """

# import pandas as pd
# import numpy as np
# import joblib
# import time
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# # =============================================================================
# # CONFIGURAZIONE
# # =============================================================================

# MODEL_PATHS = {
#     'Decision Tree': '../models/DecisionTree_CICIoT2023_multiclass.pkl',
#     'Random Forest': '../models/RandomForest_CICIoT2023_multiclass.pkl',
#     'k-NN': '../models/kNN_CICIoT2023_multiclass.pkl',
#     'SVM': '../models/SVM_CICIoT2023_multiclass.pkl'
# }

# DATA_DIR = '../data/processed/CICIOT23'
# OUTPUT_DIR = '../docs'

# # =============================================================================
# # FUNZIONI UTILITY
# # =============================================================================

# def print_header(text):
#     print("\n" + "="*80)
#     print(text.center(80))
#     print("="*80 + "\n")

# def load_test_data(data_dir):
#     """Carica solo il Test set per il benchmark."""
#     print(">>> Loading Test Data...")
#     test_path = os.path.join(data_dir, 'test_processed.pkl')
    
#     if not os.path.exists(test_path):
#         raise FileNotFoundError(f"Test data not found at {test_path}")
        
#     df_test = pd.read_pickle(test_path)
    
#     # Rimuovi colonne non-feature
#     cols_to_drop = ['y_macro_encoded', 'y_specific']
#     feature_cols = [c for c in df_test.columns if c not in cols_to_drop]
    
#     X_test = df_test[feature_cols].values
#     y_test = df_test['y_macro_encoded'].values
    
#     print(f"✅ Loaded {len(X_test)} test samples.")
#     return X_test, y_test

# def load_models():
#     """Carica i modelli disponibili."""
#     print(">>> Loading Models...")
#     loaded_models = {}
    
#     for name, path in MODEL_PATHS.items():
#         if os.path.exists(path):
#             try:
#                 model = joblib.load(path)
#                 loaded_models[name] = {
#                     'model': model,
#                     'path': path,
#                     'size_mb': os.path.getsize(path) / 1024**2
#                 }
#                 print(f"✅ Loaded: {name}")
#             except Exception as e:
#                 print(f"❌ Error loading {name}: {e}")
#         else:
#             print(f"⚠️  Skipping {name} (File not found: {path})")
            
#     return loaded_models

# # =============================================================================
# # CORE BENCHMARKING
# # =============================================================================

# def benchmark_models(models_dict, X_test, y_test):
#     """Esegue il benchmark completo."""
#     results = []
    
#     print_header("STARTING BENCHMARK")
    
#     for name, info in models_dict.items():
#         print(f"Evaluating {name}...")
#         model = info['model']
        
#         # 1. Metriche di Classificazione
#         start_pred = time.time()
#         y_pred = model.predict(X_test)
#         total_pred_time = time.time() - start_pred
        
#         # Usa 'weighted' per gestire il multi-class correttamente
#         acc = accuracy_score(y_test, y_pred)
#         prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
#         rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
#         f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
#         # 2. Test Latenza (Single Packet Inference)
#         # Simuliamo l'arrivo di pacchetti singoli (scenario reale NIDS)
#         n_warmup = 10
#         n_loops = 1000
#         sample = X_test[0:1] # Prendi 1 campione mantenendo le 2 dimensioni (1, n_features)
        
#         # Warmup
#         for _ in range(n_warmup):
#             model.predict(sample)
            
#         # Timing loop
#         t0 = time.perf_counter()
#         for _ in range(n_loops):
#             model.predict(sample)
#         t1 = time.perf_counter()
        
#         latency_ms = ((t1 - t0) / n_loops) * 1000  # ms per packet
#         throughput = 1000 / latency_ms             # packets per second (pps)
        
#         results.append({
#             'Model': name,
#             'Accuracy': acc,
#             'Precision': prec,
#             'Recall': rec,
#             'F1-Score': f1,
#             'Latency (ms)': latency_ms,
#             'Throughput (pps)': throughput,
#             'Model Size (MB)': info['size_mb']
#         })
        
#         print(f"   -> Accuracy: {acc:.4f} | F1: {f1:.4f}")
#         print(f"   -> Latency:  {latency_ms:.4f} ms/packet")
    
#     return pd.DataFrame(results)

# # =============================================================================
# # VISUALIZATION
# # =============================================================================

# def plot_comparison(df, output_dir):
#     """Genera grafici comparativi."""
#     os.makedirs(output_dir, exist_ok=True)
#     sns.set_style("whitegrid")
    
#     # 1. Bar Chart Metriche Classiche
#     fig, axes = plt.subplots(2, 2, figsize=(14, 10))
#     metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
#     colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    
#     for ax, metric, color in zip(axes.flat, metrics, colors):
#         df_sorted = df.sort_values(metric, ascending=True) # Ascending per barh
#         bars = ax.barh(df_sorted['Model'], df_sorted[metric], color=color, alpha=0.8)
#         ax.set_title(metric, fontweight='bold')
#         ax.set_xlim(0.8, 1.01) # Zoom sui valori alti
        
#         # Valori sulle barre
#         for bar in bars:
#             width = bar.get_width()
#             ax.text(width, bar.get_y() + bar.get_height()/2, 
#                     f'{width:.4f}', va='center', ha='left', fontsize=9, fontweight='bold')

#     plt.suptitle('Model Performance Comparison - CICIoT2023', fontsize=16)
#     plt.tight_layout()
#     plt.savefig(f'{output_dir}/model_metrics_comparison.png', dpi=150)
#     plt.close()
    
#     # 2. Scatter Plot: Accuracy vs Latency (Trade-off)
#     plt.figure(figsize=(10, 8))
    
#     # Plot points
#     sns.scatterplot(data=df, x='Latency (ms)', y='F1-Score', s=200, hue='Model', style='Model', palette='deep')
    
#     # Annotate points
#     for i in range(df.shape[0]):
#         plt.text(
#             df['Latency (ms)'][i]+0.01, 
#             df['F1-Score'][i]+0.001, 
#             df['Model'][i], 
#             fontweight='bold', 
#             fontsize=10
#         )
        
#     plt.title('Trade-off Analysis: Performance vs Speed', fontsize=14, fontweight='bold')
#     plt.xlabel('Inference Latency (ms) [Lower is Better]', fontsize=12)
#     plt.ylabel('F1-Score [Higher is Better]', fontsize=12)
#     plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
#     # Zone ideali
#     plt.axvline(x=0.1, color='green', linestyle=':', alpha=0.5)
#     plt.text(0.05, df['F1-Score'].min(), 'Real-Time Zone\n(<0.1ms)', color='green', fontsize=9)
    
#     plt.tight_layout()
#     plt.savefig(f'{output_dir}/accuracy_vs_latency_tradeoff.png', dpi=150)
#     plt.close()
    
#     print(f"\n📊 Plots saved in {output_dir}")

# # =============================================================================
# # MAIN
# # =============================================================================

# if __name__ == '__main__':
#     print_header("NIDS MODEL BENCHMARK SUITE")
    
#     try:
#         # 1. Load Data
#         X_test, y_test = load_test_data(DATA_DIR)
        
#         # 2. Load Models
#         models = load_models()
        
#         if not models:
#             print("\n❌ No models found! Run training scripts first.")
#             exit()
            
#         # 3. Run Benchmark
#         df_results = benchmark_models(models, X_test, y_test)
        
#         # 4. Display Results
#         print_header("FINAL RESULTS")
#         # Formattazione colonne per leggibilità
#         format_dict = {
#             'Accuracy': '{:.4f}'.format,       # Nota il .format alla fine
#             'Precision': '{:.4f}'.format, 
#             'Recall': '{:.4f}'.format, 
#             'F1-Score': '{:.4f}'.format,
#             'Latency (ms)': '{:.4f}'.format, 
#             'Throughput (pps)': '{:.0f}'.format,
#             'Model Size (MB)': '{:.2f}'.format
#         }
#         print(df_results.to_string(index=False, formatters=format_dict))
        
#         # 5. Save & Plot
#         csv_path = f'{OUTPUT_DIR}/benchmark_results.csv'
#         df_results.to_csv(csv_path, index=False)
#         print(f"\n💾 CSV Report saved: {csv_path}")
        
#         plot_comparison(df_results, OUTPUT_DIR)
        
#         # 6. Winner Declaration
#         best_f1 = df_results.loc[df_results['F1-Score'].idxmax()]
#         best_speed = df_results.loc[df_results['Latency (ms)'].idxmin()]
        
#         print("\n" + "🏆"*20)
#         print(f"BEST MODEL (Quality): {best_f1['Model']} (F1: {best_f1['F1-Score']:.4f})")
#         print(f"FASTEST MODEL (Speed):  {best_speed['Model']} (Latency: {best_speed['Latency (ms)']:.4f} ms)")
#         print("🏆"*20)
        
#     except Exception as e:
#         print(f"\n❌ Critical Error: {e}")
#         import traceback
#         traceback.print_exc()

"""
Compare Models - Confronto Parametrico Multi-Algoritmo Multi-Dataset.

Confronta performance di diversi algoritmi su dataset Original e SMOTE.
Supporta confronti parziali o completi (default: tutti vs tutti).

Usage:
    # Confronta TUTTI i modelli su TUTTI i dataset (default)
    python src/compare_models.py
    
    # Confronta solo alcuni algoritmi
    python src/compare_models.py --algorithms DecisionTree RandomForest
    
    # Confronta su solo un dataset
    python src/compare_models.py --datasets smote
    
    # Confronto specifico
    python src/compare_models.py --algorithms RandomForest kNN --datasets original
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
CICIOT_DIR = DATA_DIR / "CICIOT23"
MODELS_DIR = BASE_DIR / "models"
DOCS_DIR = BASE_DIR / "docs" / "Comparison"

# Algoritmi disponibili
ALGORITHMS = {
    'DecisionTree': {
        'name': 'Decision Tree',
        'dir': 'DecisionTree',
        'file_prefix': 'dt_model',
        'color': '#3498db'
    },
    'RandomForest': {
        'name': 'Random Forest',
        'dir': 'RandomForest',
        'file_prefix': 'rf_model',
        'color': '#2ecc71'
    },
    'kNN': {
        'name': 'k-NN',
        'dir': 'kNN',
        'file_prefix': 'knn_model',
        'color': '#9b59b6'
    },
    'SVM': {
        'name': 'SVM',
        'dir': 'SVM',
        'file_prefix': 'svm_model',
        'color': '#e67e22'
    }
}

# Dataset disponibili
DATASETS = ['original', 'smote']


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


def validate_algorithms(algorithms):
    """Valida lista algoritmi."""
    if not algorithms:
        return list(ALGORITHMS.keys())
    
    invalid = [a for a in algorithms if a not in ALGORITHMS]
    if invalid:
        raise ValueError(f"Invalid algorithms: {invalid}. Valid: {list(ALGORITHMS.keys())}")
    
    return algorithms


def validate_datasets(datasets):
    """Valida lista datasets."""
    if not datasets:
        return DATASETS
    
    invalid = [d for d in datasets if d not in DATASETS]
    if invalid:
        raise ValueError(f"Invalid datasets: {invalid}. Valid: {DATASETS}")
    
    return datasets


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model(algorithm, dataset_type):
    """
    Carica modello specifico.
    
    Returns:
        model, model_path oppure (None, None) se non esiste
    """
    algo_info = ALGORITHMS[algorithm]
    model_dir = MODELS_DIR / algo_info['dir']
    model_file = f"{algo_info['file_prefix']}_{dataset_type}.pkl"
    model_path = model_dir / model_file
    
    if not model_path.exists():
        print(f"⚠️  Model not found: {model_path}")
        return None, None
    
    try:
        model = joblib.load(model_path)
        return model, model_path
    except Exception as e:
        print(f"❌ Error loading {model_path}: {e}")
        return None, None


def load_all_models(algorithms, datasets):
    """
    Carica tutti i modelli richiesti.
    
    Returns:
        dict: {(algorithm, dataset): (model, path)}
    """
    print_header("LOADING MODELS")
    
    models = {}
    total = len(algorithms) * len(datasets)
    loaded = 0
    
    for algo in algorithms:
        for dataset in datasets:
            print(f"Loading {ALGORITHMS[algo]['name']} ({dataset})...", end=" ")
            model, path = load_model(algo, dataset)
            
            if model is not None:
                models[(algo, dataset)] = (model, path)
                loaded += 1
                print(f"✅ {path}")
            else:
                print(f"❌ Not found")
    
    print(f"\n📊 Loaded {loaded}/{total} models")
    
    if loaded == 0:
        raise RuntimeError("No models found. Train models first!")
    
    return models


# =============================================================================
# DATA LOADING
# =============================================================================

def load_test_data():
    """Carica test set (sempre da CICIOT23 per valutazione realistica)."""
    print_header("LOADING TEST DATA")
    
    test_path = CICIOT_DIR / 'test_processed.pkl'
    
    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found: {test_path}")
    
    df_test = pd.read_pickle(test_path)
    
    feature_cols = [col for col in df_test.columns 
                   if col not in ['y_macro_encoded', 'y_specific']]
    
    X_test = df_test[feature_cols].values
    y_test = df_test['y_macro_encoded'].values
    
    print(f"✅ Test data loaded: {test_path}")
    print(f"   Shape: {df_test.shape}")
    print(f"   Features: {len(feature_cols)}")
    print(f"   Samples: {len(X_test):,}")
    
    return X_test, y_test, feature_cols


def load_label_encoder():
    """Carica label encoder."""
    encoder_path = CICIOT_DIR / 'label_encoder.pkl'
    label_encoder = joblib.load(encoder_path)
    print(f"✅ Label encoder loaded: {label_encoder.classes_}")
    return label_encoder


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_model(model, X_test, y_test, algo_name, dataset_type):
    """
    Valuta singolo modello.
    
    Returns:
        dict con metriche
    """
    y_pred = model.predict(X_test)
    
    metrics = {
        'algorithm': algo_name,
        'dataset': dataset_type,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
    }
    
    return metrics, y_pred


def evaluate_all_models(models, X_test, y_test):
    """
    Valuta tutti i modelli caricati.
    
    Returns:
        results: list of dicts
        predictions: dict {(algo, dataset): y_pred}
    """
    print_header("EVALUATING MODELS")
    
    results = []
    predictions = {}
    
    for (algo, dataset), (model, path) in models.items():
        algo_name = ALGORITHMS[algo]['name']
        print(f"Evaluating {algo_name} ({dataset})...", end=" ")
        
        try:
            metrics, y_pred = evaluate_model(model, X_test, y_test, algo_name, dataset)
            results.append(metrics)
            predictions[(algo, dataset)] = y_pred
            print(f"✅ Acc: {metrics['accuracy']:.4f}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    return results, predictions


# =============================================================================
# RESULTS DISPLAY
# =============================================================================

def display_results_table(results):
    """Mostra tabella risultati."""
    print_section("RESULTS SUMMARY")
    
    # Converti a DataFrame
    df = pd.DataFrame(results)
    
    # Sort by accuracy (desc)
    df = df.sort_values('accuracy', ascending=False)
    
    # Formatta
    print(f"{'#':<4} {'Algorithm':<20} {'Dataset':<10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Status':>8}")
    print("-"*90)
    
    for idx, row in df.iterrows():
        # Check target
        status = "✅" if (row['accuracy'] >= 0.95 and 
                        row['precision'] >= 0.90 and 
                        row['recall'] >= 0.95) else "❌"
        
        print(f"{idx+1:<4} {row['algorithm']:<20} {row['dataset']:<10} "
              f"{row['accuracy']:>10.4f} {row['precision']:>10.4f} "
              f"{row['recall']:>10.4f} {row['f1']:>10.4f} {status:>8}")
    
    print("-"*90)
    
    # Best per metric
    print("\n🏆 Best Models:")
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        best = df.loc[df[metric].idxmax()]
        print(f"   {metric.capitalize():<12}: {best['algorithm']} ({best['dataset']}) - {best[metric]:.4f}")
    
    return df


def display_comparison_by_algorithm(df):
    """Confronta Original vs SMOTE per ogni algoritmo."""
    print_section("ORIGINAL vs SMOTE COMPARISON")
    
    algorithms_present = df['algorithm'].unique()
    
    for algo in algorithms_present:
        df_algo = df[df['algorithm'] == algo]
        
        if len(df_algo) == 2:  # Ha sia original che smote
            original = df_algo[df_algo['dataset'] == 'original'].iloc[0]
            smote = df_algo[df_algo['dataset'] == 'smote'].iloc[0]
            
            print(f"\n{algo}:")
            print(f"  Original → SMOTE (Δ)")
            print(f"    Accuracy:  {original['accuracy']:.4f} → {smote['accuracy']:.4f} ({smote['accuracy']-original['accuracy']:+.4f})")
            print(f"    Precision: {original['precision']:.4f} → {smote['precision']:.4f} ({smote['precision']-original['precision']:+.4f})")
            print(f"    Recall:    {original['recall']:.4f} → {smote['recall']:.4f} ({smote['recall']-original['recall']:+.4f})")
            print(f"    F1:        {original['f1']:.4f} → {smote['f1']:.4f} ({smote['f1']-original['f1']:+.4f})")


# =============================================================================
# VISUALIZATIONS
# =============================================================================

def plot_metrics_comparison(df, output_dir):
    """Confronto metriche tra tutti i modelli."""
    print_section("GENERATING PLOTS")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Grouped Bar Chart - Tutte le metriche
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        # Prepara dati
        algorithms = df['algorithm'].unique()
        datasets = df['dataset'].unique()
        
        x = np.arange(len(algorithms))
        width = 0.35
        
        for i, dataset in enumerate(datasets):
            values = []
            for algo in algorithms:
                row = df[(df['algorithm'] == algo) & (df['dataset'] == dataset)]
                if len(row) > 0:
                    values.append(row.iloc[0][metric])
                else:
                    values.append(0)
            
            ax.bar(x + i*width, values, width, 
                   label=dataset.capitalize(), alpha=0.8)
        
        ax.set_ylabel(metric.capitalize(), fontsize=12)
        ax.set_title(f'{metric.capitalize()} Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(algorithms, rotation=15, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0.85, 1.0])
        
        # Target lines
        if metric == 'accuracy':
            ax.axhline(y=0.95, color='r', linestyle='--', alpha=0.5)
        elif metric == 'precision':
            ax.axhline(y=0.90, color='orange', linestyle='--', alpha=0.5)
        elif metric == 'recall':
            ax.axhline(y=0.95, color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    path = output_dir / 'all_metrics_comparison.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {path}")
    plt.close()
    
    # 2. Heatmap - Accuracy per (Algorithm, Dataset)
    fig, ax = plt.subplots(figsize=(10, 8))
    
    pivot = df.pivot_table(values='accuracy', 
                           index='algorithm', 
                           columns='dataset')
    
    sns.heatmap(pivot, annot=True, fmt='.4f', cmap='RdYlGn', 
                vmin=0.85, vmax=1.0, ax=ax,
                cbar_kws={'label': 'Accuracy'})
    ax.set_title('Accuracy Heatmap - All Models', fontsize=14, fontweight='bold')
    ax.set_xlabel('Dataset Type', fontsize=12)
    ax.set_ylabel('Algorithm', fontsize=12)
    
    plt.tight_layout()
    path = output_dir / 'accuracy_heatmap.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {path}")
    plt.close()
    
    # 3. Radar Chart - Per ogni dataset
    for dataset in df['dataset'].unique():
        df_dataset = df[df['dataset'] == dataset]
        
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='polar')
        
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Close the plot
        
        for idx, row in df_dataset.iterrows():
            algo = row['algorithm']
            values = [row[m] for m in metrics]
            values += values[:1]  # Close the plot
            
            # Get color
            algo_key = [k for k, v in ALGORITHMS.items() if v['name'] == algo][0]
            color = ALGORITHMS[algo_key]['color']
            
            ax.plot(angles, values, 'o-', linewidth=2, label=algo, color=color)
            ax.fill(angles, values, alpha=0.15, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.capitalize() for m in metrics], fontsize=11)
        ax.set_ylim(0.85, 1.0)
        ax.set_title(f'Performance Radar - {dataset.capitalize()} Dataset', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        path = output_dir / f'radar_chart_{dataset}.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved: {path}")
        plt.close()
    
    # 4. Algorithm Ranking
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Calcola score medio per ogni (algo, dataset)
    df['mean_score'] = df[['accuracy', 'precision', 'recall', 'f1']].mean(axis=1)
    df_sorted = df.sort_values('mean_score', ascending=True)
    
    y_pos = np.arange(len(df_sorted))
    colors = []
    
    for idx, row in df_sorted.iterrows():
        algo_key = [k for k, v in ALGORITHMS.items() if v['name'] == row['algorithm']][0]
        colors.append(ALGORITHMS[algo_key]['color'])
    
    bars = ax.barh(y_pos, df_sorted['mean_score'], color=colors, alpha=0.8)
    
    # Labels
    labels = [f"{row['algorithm']} ({row['dataset']})" 
              for idx, row in df_sorted.iterrows()]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('Mean Score (Avg of all metrics)', fontsize=12)
    ax.set_title('Model Ranking by Mean Performance', fontsize=14, fontweight='bold')
    ax.set_xlim([0.85, 1.0])
    ax.grid(axis='x', alpha=0.3)
    
    # Add values on bars
    for bar, value in zip(bars, df_sorted['mean_score']):
        width = bar.get_width()
        ax.text(width + 0.003, bar.get_y() + bar.get_height()/2, 
                f'{value:.4f}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    path = output_dir / 'ranking.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {path}")
    plt.close()


def save_results_report(df, output_dir):
    """Salva report in formato CSV e JSON."""
    print_section("SAVING REPORT")
    
    # CSV
    csv_path = output_dir / 'comparison_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"✅ CSV report: {csv_path}")
    
    # JSON
    json_path = output_dir / 'comparison_results.json'
    report = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'total_models': len(df),
        'algorithms': df['algorithm'].unique().tolist(),
        'datasets': df['dataset'].unique().tolist(),
        'results': df.to_dict('records'),
        'best': {
            'accuracy': df.loc[df['accuracy'].idxmax()].to_dict(),
            'precision': df.loc[df['precision'].idxmax()].to_dict(),
            'recall': df.loc[df['recall'].idxmax()].to_dict(),
            'f1': df.loc[df['f1'].idxmax()].to_dict()
        }
    }
    
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✅ JSON report: {json_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Compare ML models across algorithms and datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare all models on all datasets (default)
  python src/compare_models.py
  
  # Compare specific algorithms
  python src/compare_models.py --algorithms DecisionTree RandomForest
  
  # Compare on specific dataset
  python src/compare_models.py --datasets smote
  
  # Specific comparison
  python src/compare_models.py --algorithms RandomForest kNN --datasets original smote
        """
    )
    
    parser.add_argument('--algorithms', type=str, nargs='*',
                        choices=list(ALGORITHMS.keys()),
                        help='Algorithms to compare (default: all)')
    parser.add_argument('--datasets', type=str, nargs='*',
                        choices=DATASETS,
                        help='Datasets to compare (default: all)')
    parser.add_argument('--output-dir', type=Path, 
                        default=DOCS_DIR,
                        help='Output directory for plots and reports')
    
    args = parser.parse_args()
    
    # Valida input
    algorithms = validate_algorithms(args.algorithms)
    datasets = validate_datasets(args.datasets)
    
    print("\n" + "🔬"*40)
    print("MODEL COMPARISON FRAMEWORK".center(80))
    print("🔬"*40)
    
    print(f"\n📊 Configuration:")
    print(f"   Algorithms: {', '.join(algorithms)}")
    print(f"   Datasets: {', '.join(datasets)}")
    print(f"   Total comparisons: {len(algorithms) * len(datasets)}")
    print(f"   Output: {args.output_dir}")
    
    # Load
    models = load_all_models(algorithms, datasets)
    X_test, y_test, feature_cols = load_test_data()
    label_encoder = load_label_encoder()
    
    # Evaluate
    results, predictions = evaluate_all_models(models, X_test, y_test)
    
    # Display
    df = display_results_table(results)
    display_comparison_by_algorithm(df)
    
    # Visualize
    plot_metrics_comparison(df, args.output_dir)
    
    # Save
    save_results_report(df, args.output_dir)
    
    # Summary
    print_header("✅ COMPARISON COMPLETE!")
    
    best = df.loc[df['accuracy'].idxmax()]
    print(f"🏆 Best Model Overall:")
    print(f"   Algorithm: {best['algorithm']}")
    print(f"   Dataset: {best['dataset']}")
    print(f"   Accuracy: {best['accuracy']:.4f}")
    print(f"   Precision: {best['precision']:.4f}")
    print(f"   Recall: {best['recall']:.4f}")
    print(f"   F1: {best['f1']:.4f}")
    
    meets_req = (best['accuracy'] >= 0.95 and 
                 best['precision'] >= 0.90 and 
                 best['recall'] >= 0.95)
    
    if meets_req:
        print("\n✅ Best model MEETS all project requirements!")
    else:
        print("\n⚠️ No model meets all requirements yet.")
    
    print(f"\n📁 Outputs:")
    print(f"   Reports: {args.output_dir}/comparison_results.{{csv,json}}")
    print(f"   Plots: {args.output_dir}/*.png")


if __name__ == '__main__':
    main()