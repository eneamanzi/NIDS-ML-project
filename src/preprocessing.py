"""
Advanced Data Processing per CICIoT2023 - MERGED VERSION (OPTIMIZED)
Unisce la robustezza statistica del vecchio codice con l'efficienza di memoria del nuovo.

CHANGELOG (v2.0):
- FIX CRITICO: y_specific usa mapping coerente (specific_to_idx) generato nel Pass 1.
- OTTIMIZZAZIONE RAM: Conversione feature a float32 (dimezza l'uso di memoria).
- BEST PRACTICE: Shuffling locale dei chunk per ridurre bias temporale.

FEATURES:
1. Double Pass Strategy:
   - Pass 1: Partial Fit (Scaler) + Statistiche + Costruzione Dizionario Label
   - Pass 2: Transform & Save (Chunking) con mapping coerente
2. Fixed Encoder: LabelEncoder fissato sulle macro-categorie note
3. Memory Efficient: Non satura la RAM

Usage:
    python src/preprocessing.py --train-path data/raw/train.csv
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os
import json
from pathlib import Path
import warnings
import gc

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURAZIONE
# =============================================================================

CHUNK_SIZE = 500_000   # Dimensione chunk ottimizzata
USE_CHUNKING = True    # Sempre True per sicurezza

# =============================================================================
# ATTACK MAPPING
# =============================================================================

ATTACK_MAPPING = {
    # === DDoS (12 varianti) ===
    'DDoS-ICMP_Flood': 'DDoS',
    'DDoS-UDP_Flood': 'DDoS',
    'DDoS-TCP_Flood': 'DDoS',
    'DDoS-PSHACK_Flood': 'DDoS',
    'DDoS-RSTFINFlood': 'DDoS',
    'DDoS-SYN_Flood': 'DDoS',
    'DDoS-SynonymousIP_Flood': 'DDoS',
    'DDoS-ICMP_Fragmentation': 'DDoS',
    'DDoS-UDP_Fragmentation': 'DDoS',
    'DDoS-ACK_Fragmentation': 'DDoS',
    'DDoS-HTTP_Flood': 'DDoS',
    'DDoS-SlowLoris': 'DDoS',
    
    # === DoS (4 varianti) ===
    'DoS-UDP_Flood': 'DoS',
    'DoS-TCP_Flood': 'DoS',
    'DoS-SYN_Flood': 'DoS',
    'DoS-HTTP_Flood': 'DoS',
    
    # === Mirai (3 varianti) ===
    'Mirai-greeth_flood': 'Mirai',
    'Mirai-udpplain': 'Mirai',
    'Mirai-greip_flood': 'Mirai',
    
    # === Spoofing (2 varianti) ===
    'MITM-ArpSpoofing': 'Spoofing',
    'DNS_Spoofing': 'Spoofing',
    
    # === Recon (5 varianti) ===
    'Recon-HostDiscovery': 'Recon',
    'Recon-OSScan': 'Recon',
    'Recon-PortScan': 'Recon',
    'Recon-PingSweep': 'Recon',
    'VulnerabilityScan': 'Recon',
    
    # === Web (6 varianti) ===
    'SqlInjection': 'Web',
    'XSS': 'Web',
    'CommandInjection': 'Web',
    'Uploading_Attack': 'Web',
    'BrowserHijacking': 'Web',
    'Backdoor_Malware': 'Web',
    
    # === BruteForce (1 variante) ===
    'DictionaryBruteForce': 'BruteForce',
    
    # === Benign ===
    'BenignTraffic': 'Benign'
}

# Macro-categorie fisse per l'Encoder
MACRO_CATEGORIES = ['Benign', 'DDoS', 'DoS', 'Mirai', 'Recon', 'Web', 'Spoofing', 'BruteForce']

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

def save_json(data, filepath):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"✅ Saved: {filepath}")

# =============================================================================
# CORE PROCESSING FUNCTIONS
# =============================================================================

def get_feature_cols(filepath, label_col='label'):
    """Legge l'header per identificare le colonne feature."""
    print(f"Reading header from: {filepath}")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
        
    df_head = pd.read_csv(filepath, nrows=1)
    
    if label_col not in df_head.columns:
        raise ValueError(f"Label column '{label_col}' not found!")
        
    feature_cols = [col for col in df_head.columns if col != label_col]
    print(f"  Detected {len(feature_cols)} features")
    return feature_cols


def process_chunk_cleaning(chunk):
    """Pulizia dati robusta su singolo chunk."""
    original_len = len(chunk)
    
    # 1. Missing values
    if chunk.isnull().sum().sum() > 0:
        n_missing = chunk.isnull().sum().sum()
        print(f"  ⚠️  Found {n_missing} missing values, filling with 0")
        chunk = chunk.fillna(0)
    
    # 2. Infinite values
    numeric_cols = chunk.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        inf_mask = np.isinf(chunk[col])
        if inf_mask.any():
            n_inf = inf_mask.sum()
            max_val = chunk[col][~inf_mask].max()
            chunk.loc[inf_mask, col] = max_val
            if n_inf > 10:  # Log solo se significativo
                print(f"  ⚠️  Column '{col}': Replaced {n_inf} infinite values")
    
    # 3. Duplicates (Locali)
    chunk = chunk.drop_duplicates()
    n_removed = original_len - len(chunk)
    if n_removed > 0:
        print(f"  ℹ️  Removed {n_removed} duplicate rows in chunk")
    
    # 4. NUOVO: Check label validity
    if 'label' in chunk.columns:
        unknown_labels = chunk[~chunk['label'].isin(ATTACK_MAPPING.keys())]
        if len(unknown_labels) > 0:
            print(f"  ⚠️  Found {len(unknown_labels)} unknown labels: {unknown_labels['label'].unique()}")
    
    return chunk


def fit_scaler_and_analyze(filepath, feature_cols, label_col='label', nrows=None, chunk_size=CHUNK_SIZE):
    """
    PASSAGGIO 1:
    - Partial Fit Scaler
    - Analisi Classi
    - Costruzione Mappa 'specific_to_idx' (FIX CRITICO)
    """
    print_section("STEP 1: ANALYZING & FITTING (Passaggio 1)")
    
    # Setup Encoder & Scaler
    label_encoder = LabelEncoder()
    label_encoder.fit(MACRO_CATEGORIES)
    scaler = StandardScaler()
    
    # Accumulatori
    specific_label_counts = pd.Series(dtype='int64')
    total_processed_rows = 0
    
    print(f"Reading {filepath} for analysis...")
    chunk_iterator = pd.read_csv(filepath, chunksize=chunk_size, nrows=nrows)
    
    for i, chunk in enumerate(chunk_iterator):
        chunk = process_chunk_cleaning(chunk)
        
        # Accumulo statistiche label
        counts = chunk[label_col].value_counts()
        specific_label_counts = specific_label_counts.add(counts, fill_value=0)
        
        # Partial Fit Scaler
        X_chunk = chunk[feature_cols].values
        scaler.partial_fit(X_chunk)
        
        total_processed_rows += len(chunk)
        if (i+1) % 5 == 0:
            print(f"  Analyzed chunk {i+1}...", end="\r")
        
        del chunk, X_chunk
        gc.collect()
        
    print(f"\n✅ Analysis complete on {total_processed_rows:,} rows.")
    
    # --- REPORT STATISTICO ---
    print("\n" + "-"*60)
    print("DATASET ANALYSIS REPORT")
    print("-" * 60)
    
    # Generazione Mapping Coerente (FIX)
    sorted_labels = sorted(specific_label_counts.index)
    specific_to_idx = {lbl: idx for idx, lbl in enumerate(sorted_labels)}
    print(f"Generated consistent mapping for {len(specific_to_idx)} specific classes.")

    # Top 10 Specific Labels
    print("\nTop 10 Specific Attack Types:")
    top_specific = specific_label_counts.sort_values(ascending=False).head(10)
    print(top_specific)
    
    # Macro Categories Distribution
    print("\nMacro-Categories Distribution:")
    stats_df = pd.DataFrame({'count': specific_label_counts})
    stats_df['macro'] = stats_df.index.map(ATTACK_MAPPING)
    macro_counts = stats_df.groupby('macro')['count'].sum().sort_values(ascending=False)
    
    for cat, count in macro_counts.items():
        perc = (count / total_processed_rows) * 100
        print(f"  {cat:15s}: {int(count):>9,} ({perc:>6.2f}%)")
    print("-" * 60)
    
    mapping_info = {
        'attack_mapping': ATTACK_MAPPING,
        'macro_categories': MACRO_CATEGORIES,
        'label_encoder_classes': label_encoder.classes_.tolist(),
        'n_specific_classes': len(specific_to_idx),
        'total_rows_analyzed': int(total_processed_rows),
        'specific_to_idx': specific_to_idx, 
        'idx_to_specific': {v: k for k, v in specific_to_idx.items()}
    }
    
    return scaler, label_encoder, mapping_info, specific_to_idx


def process_dataset_chunked(filepath, scaler, label_encoder, specific_to_idx, feature_cols,
                            output_path, chunk_size=CHUNK_SIZE, 
                            nrows=None, label_col='label'):
    """
    PASSAGGIO 2: Transform & Save.
    Ora usa specific_to_idx per garantire coerenza in y_specific.
    """
    print_section(f"STEP 2: TRANSFORMING & SAVING: {Path(filepath).name}")
    print(f"Output: {output_path}")
    
    chunk_iterator = pd.read_csv(filepath, chunksize=chunk_size, nrows=nrows)
    
    processed_chunks = []
    total_rows = 0
    chunk_idx = 0
    
    # ### NUOVO: Creiamo la mappa VELOCE una volta sola fuori dal loop
    macro_map = {cls: idx for idx, cls in enumerate(label_encoder.classes_)}

    for chunk in chunk_iterator:
        chunk_idx += 1
        
        # 1. Clean
        chunk = process_chunk_cleaning(chunk)
        
        # 2. Labels (FIX CRITICO)
        chunk['y_specific'] = chunk[label_col].map(specific_to_idx).fillna(-1).astype(int)
        chunk['y_macro'] = chunk[label_col].map(ATTACK_MAPPING)
        
        # 3. Encoding y_macro
        #chunk['y_macro_encoded'] = chunk['y_macro'].apply(
        #    lambda x: label_encoder.transform([x])[0] if x in label_encoder.classes_ else -1
        #)
        chunk['y_macro_encoded'] = chunk['y_macro'].map(macro_map).fillna(-1).astype(int)
        
        # ⚠️ BEST PRACTICE: Shuffling locale per rompere l'ordine temporale
        chunk = chunk.sample(frac=1, random_state=42).reset_index(drop=True)

        # 4. Scaling Features + OTTIMIZZAZIONE RAM (float32)
        # Convertiamo subito a float32 per dimezzare l'occupazione di memoria
        chunk[feature_cols] = scaler.transform(chunk[feature_cols]).astype(np.float32)
        
        # 5. Select & Accumulate
        output_cols = feature_cols + ['y_macro_encoded', 'y_specific']
        processed_chunks.append(chunk[output_cols])
        
        rows_in_chunk = len(chunk)
        total_rows += rows_in_chunk
        
        print(f"  Processed Chunk {chunk_idx}: {rows_in_chunk:,} rows (Total: {total_rows:,})", end="\n")
        
        del chunk
        gc.collect()
    
    print(f"Concatenating {len(processed_chunks)} chunks...")
    df_final = pd.concat(processed_chunks, ignore_index=True)
    
    print(f"Saving pickle to {output_path}...")
    df_final.to_pickle(output_path)
    
    size_mb = Path(output_path).stat().st_size / 1024**2
    print(f"✅ Saved: {output_path} ({size_mb:.2f} MB)")
    
    del processed_chunks, df_final
    gc.collect()
    
    return total_rows


# =============================================================================
# VALIDATION (AGGIUNGI QUESTA SEZIONE DOPO LE ALTRE FUNZIONI)
# =============================================================================

def validate_processed_data(output_dir):
    """Validazione completa post-processing."""
    print_section("VALIDATION CHECKS")
    
    # Carica tutti gli split
    df_train = pd.read_pickle(f"{output_dir}/train_processed.pkl")
    df_test = pd.read_pickle(f"{output_dir}/test_processed.pkl")
    df_val = pd.read_pickle(f"{output_dir}/validation_processed.pkl")
    
    with open(f"{output_dir}/mapping_info.json") as f:
        mapping = json.load(f)
    
    checks = {
        'train': df_train,
        'test': df_test,
        'val': df_val
    }
    
    for name, df in checks.items():
        print(f"\n{name.upper()} Split:")
        
        # 1. Nessun NaN
        assert df.isnull().sum().sum() == 0, f"{name}: Found NaN values!"
        print(f"  ✅ No NaN values")
        
        # 2. Range y_specific
        assert df['y_specific'].min() >= 0, f"{name}: y_specific < 0!"
        assert df['y_specific'].max() < mapping['n_specific_classes'], \
            f"{name}: y_specific out of range!"
        print(f"  ✅ y_specific in range [0, {mapping['n_specific_classes']-1}]")
        
        # 3. Range y_macro_encoded
        assert df['y_macro_encoded'].min() >= 0
        assert df['y_macro_encoded'].max() < len(mapping['macro_categories'])
        print(f"  ✅ y_macro_encoded in range [0, {len(mapping['macro_categories'])-1}]")
        
        # 4. Dtypes corretti
        assert df['y_specific'].dtype in [np.int64, np.int32]
        assert df['y_macro_encoded'].dtype in [np.int64, np.int32]
        print(f"  ✅ Correct dtypes")
        
        # 5. Feature dtype (float32)
        feature_cols = [col for col in df.columns if col not in ['y_macro_encoded', 'y_specific']]
        assert all(df[col].dtype == np.float32 for col in feature_cols), "Features not float32!"
        print(f"  ✅ Features are float32 (memory optimized)")
    
    print("\n✅ ALL VALIDATION CHECKS PASSED!")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def process_pipeline(train_path, test_path, val_path, output_dir, nrows, label_col):
    
    print_header("🚀 ADVANCED DATA PROCESSING (Merged & Optimized)")
    print(f"Chunking Mode: ON (Size: {CHUNK_SIZE:,})")
    print("Optimizations: Float32 Conversion + Local Shuffling")
    
    # 0. Get Info
    feature_cols = get_feature_cols(train_path, label_col)
    
    # 1. Fit & Analyze (Su Train) - Ritorna anche il mapping specifico
    scaler, label_encoder, mapping_info, specific_to_idx = fit_scaler_and_analyze(
        train_path, feature_cols, label_col, nrows=nrows
    )
    
    # 2. Process Datasets (Passando specific_to_idx)
    os.makedirs(output_dir, exist_ok=True)
    
    # Train
    process_dataset_chunked(
        train_path, scaler, label_encoder, specific_to_idx, feature_cols,
        f"{output_dir}/train_processed.pkl", nrows=nrows, label_col=label_col
    )
    
    # Test
    if test_path and os.path.exists(test_path):
        process_dataset_chunked(
            test_path, scaler, label_encoder, specific_to_idx, feature_cols,
            f"{output_dir}/test_processed.pkl", nrows=nrows, label_col=label_col
        )
    
    # Val
    if val_path and os.path.exists(val_path):
        process_dataset_chunked(
            val_path, scaler, label_encoder, specific_to_idx, feature_cols,
            f"{output_dir}/validation_processed.pkl", nrows=nrows, label_col=label_col
        )
        
    # 3. Save Artifacts
    print_section("SAVING ARTIFACTS")
    joblib.dump(scaler, f"{output_dir}/scaler.pkl")
    joblib.dump(label_encoder, f"{output_dir}/label_encoder.pkl")
    save_json(mapping_info, f"{output_dir}/mapping_info.json")

    validate_processed_data(output_dir)
    print_header("✅ ALL DONE")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-path', default='../data/raw/CICIOT23/train/train.csv')
    parser.add_argument('--test-path', default='../data/raw/CICIOT23/test/test.csv')
    parser.add_argument('--val-path', default='../data/raw/CICIOT23/validation/validation.csv')
    parser.add_argument('--output-dir', default='../data/processed/CICIOT23')
    parser.add_argument('--nrows', type=int, default=None)
    parser.add_argument('--label-col', default='label')
    args = parser.parse_args()
    
    process_pipeline(args.train_path, args.test_path, args.val_path, 
                     args.output_dir, args.nrows, args.label_col)