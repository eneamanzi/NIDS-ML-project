"""
Model Comparison & Benchmarking per CICIoT2023.

Confronta i 4 algoritmi (DT, RF, k-NN, SVM) su:
1. Metriche di Classificazione (Accuracy, Precision, Recall, F1)
2. Performance Operative (Latenza di inferenza, Dimensione modello)

Output:
- Report CSV completo
- Grafici comparativi (Bar charts + Scatter plot Accuracy/Latency)
"""

import pandas as pd
import numpy as np
import joblib
import time
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# =============================================================================
# CONFIGURAZIONE
# =============================================================================

MODEL_PATHS = {
    'Decision Tree': '../models/DecisionTree_CICIoT2023_multiclass.pkl',
    'Random Forest': '../models/RandomForest_CICIoT2023_multiclass.pkl',
    'k-NN': '../models/kNN_CICIoT2023_multiclass.pkl',
    'SVM': '../models/SVM_CICIoT2023_multiclass.pkl'
}

DATA_DIR = '../data/processed/CICIOT23'
OUTPUT_DIR = '../docs'

# =============================================================================
# FUNZIONI UTILITY
# =============================================================================

def print_header(text):
    print("\n" + "="*80)
    print(text.center(80))
    print("="*80 + "\n")

def load_test_data(data_dir):
    """Carica solo il Test set per il benchmark."""
    print(">>> Loading Test Data...")
    test_path = os.path.join(data_dir, 'test_processed.pkl')
    
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test data not found at {test_path}")
        
    df_test = pd.read_pickle(test_path)
    
    # Rimuovi colonne non-feature
    cols_to_drop = ['y_macro_encoded', 'y_specific']
    feature_cols = [c for c in df_test.columns if c not in cols_to_drop]
    
    X_test = df_test[feature_cols].values
    y_test = df_test['y_macro_encoded'].values
    
    print(f"✅ Loaded {len(X_test)} test samples.")
    return X_test, y_test

def load_models():
    """Carica i modelli disponibili."""
    print(">>> Loading Models...")
    loaded_models = {}
    
    for name, path in MODEL_PATHS.items():
        if os.path.exists(path):
            try:
                model = joblib.load(path)
                loaded_models[name] = {
                    'model': model,
                    'path': path,
                    'size_mb': os.path.getsize(path) / 1024**2
                }
                print(f"✅ Loaded: {name}")
            except Exception as e:
                print(f"❌ Error loading {name}: {e}")
        else:
            print(f"⚠️  Skipping {name} (File not found: {path})")
            
    return loaded_models

# =============================================================================
# CORE BENCHMARKING
# =============================================================================

def benchmark_models(models_dict, X_test, y_test):
    """Esegue il benchmark completo."""
    results = []
    
    print_header("STARTING BENCHMARK")
    
    for name, info in models_dict.items():
        print(f"Evaluating {name}...")
        model = info['model']
        
        # 1. Metriche di Classificazione
        start_pred = time.time()
        y_pred = model.predict(X_test)
        total_pred_time = time.time() - start_pred
        
        # Usa 'weighted' per gestire il multi-class correttamente
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # 2. Test Latenza (Single Packet Inference)
        # Simuliamo l'arrivo di pacchetti singoli (scenario reale NIDS)
        n_warmup = 10
        n_loops = 1000
        sample = X_test[0:1] # Prendi 1 campione mantenendo le 2 dimensioni (1, n_features)
        
        # Warmup
        for _ in range(n_warmup):
            model.predict(sample)
            
        # Timing loop
        t0 = time.perf_counter()
        for _ in range(n_loops):
            model.predict(sample)
        t1 = time.perf_counter()
        
        latency_ms = ((t1 - t0) / n_loops) * 1000  # ms per packet
        throughput = 1000 / latency_ms             # packets per second (pps)
        
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'Latency (ms)': latency_ms,
            'Throughput (pps)': throughput,
            'Model Size (MB)': info['size_mb']
        })
        
        print(f"   -> Accuracy: {acc:.4f} | F1: {f1:.4f}")
        print(f"   -> Latency:  {latency_ms:.4f} ms/packet")
    
    return pd.DataFrame(results)

# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_comparison(df, output_dir):
    """Genera grafici comparativi."""
    os.makedirs(output_dir, exist_ok=True)
    sns.set_style("whitegrid")
    
    # 1. Bar Chart Metriche Classiche
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    
    for ax, metric, color in zip(axes.flat, metrics, colors):
        df_sorted = df.sort_values(metric, ascending=True) # Ascending per barh
        bars = ax.barh(df_sorted['Model'], df_sorted[metric], color=color, alpha=0.8)
        ax.set_title(metric, fontweight='bold')
        ax.set_xlim(0.8, 1.01) # Zoom sui valori alti
        
        # Valori sulle barre
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                    f'{width:.4f}', va='center', ha='left', fontsize=9, fontweight='bold')

    plt.suptitle('Model Performance Comparison - CICIoT2023', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_metrics_comparison.png', dpi=150)
    plt.close()
    
    # 2. Scatter Plot: Accuracy vs Latency (Trade-off)
    plt.figure(figsize=(10, 8))
    
    # Plot points
    sns.scatterplot(data=df, x='Latency (ms)', y='F1-Score', s=200, hue='Model', style='Model', palette='deep')
    
    # Annotate points
    for i in range(df.shape[0]):
        plt.text(
            df['Latency (ms)'][i]+0.01, 
            df['F1-Score'][i]+0.001, 
            df['Model'][i], 
            fontweight='bold', 
            fontsize=10
        )
        
    plt.title('Trade-off Analysis: Performance vs Speed', fontsize=14, fontweight='bold')
    plt.xlabel('Inference Latency (ms) [Lower is Better]', fontsize=12)
    plt.ylabel('F1-Score [Higher is Better]', fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    # Zone ideali
    plt.axvline(x=0.1, color='green', linestyle=':', alpha=0.5)
    plt.text(0.05, df['F1-Score'].min(), 'Real-Time Zone\n(<0.1ms)', color='green', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/accuracy_vs_latency_tradeoff.png', dpi=150)
    plt.close()
    
    print(f"\n📊 Plots saved in {output_dir}")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print_header("NIDS MODEL BENCHMARK SUITE")
    
    try:
        # 1. Load Data
        X_test, y_test = load_test_data(DATA_DIR)
        
        # 2. Load Models
        models = load_models()
        
        if not models:
            print("\n❌ No models found! Run training scripts first.")
            exit()
            
        # 3. Run Benchmark
        df_results = benchmark_models(models, X_test, y_test)
        
        # 4. Display Results
        print_header("FINAL RESULTS")
        # Formattazione colonne per leggibilità
        format_dict = {
            'Accuracy': '{:.4f}'.format,       # Nota il .format alla fine
            'Precision': '{:.4f}'.format, 
            'Recall': '{:.4f}'.format, 
            'F1-Score': '{:.4f}'.format,
            'Latency (ms)': '{:.4f}'.format, 
            'Throughput (pps)': '{:.0f}'.format,
            'Model Size (MB)': '{:.2f}'.format
        }
        print(df_results.to_string(index=False, formatters=format_dict))
        
        # 5. Save & Plot
        csv_path = f'{OUTPUT_DIR}/benchmark_results.csv'
        df_results.to_csv(csv_path, index=False)
        print(f"\n💾 CSV Report saved: {csv_path}")
        
        plot_comparison(df_results, OUTPUT_DIR)
        
        # 6. Winner Declaration
        best_f1 = df_results.loc[df_results['F1-Score'].idxmax()]
        best_speed = df_results.loc[df_results['Latency (ms)'].idxmin()]
        
        print("\n" + "🏆"*20)
        print(f"BEST MODEL (Quality): {best_f1['Model']} (F1: {best_f1['F1-Score']:.4f})")
        print(f"FASTEST MODEL (Speed):  {best_speed['Model']} (Latency: {best_speed['Latency (ms)']:.4f} ms)")
        print("🏆"*20)
        
    except Exception as e:
        print(f"\n❌ Critical Error: {e}")
        import traceback
        traceback.print_exc()