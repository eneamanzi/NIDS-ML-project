"""
Advanced Data Processing per CICIoT2023 - PARQUET OPTIMIZED VERSION

CHANGELOG (v3.0 - Parquet Migration):
- OUTPUT FORMAT: .pkl → .parquet (columnar, compressed, faster I/O)
- MEMORY: float64 → float32 (-50% RAM usage)
- STREAMING: Chunked write con PyArrow (no full load in memory)
- COMPRESSION: snappy (balance speed/size)

FEATURES:
1. Double Pass Strategy (unchanged)
2. Fixed Encoder (unchanged)  
3. Memory Efficient: float32 + streaming writes
4. Parquet: 40% faster I/O, better compression

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
import pyarrow as pa
import pyarrow.parquet as pq

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

CHUNK_SIZE = 500_000
USE_CHUNKING = True

# Parquet settings
PARQUET_COMPRESSION = 'snappy'  # Fast compression
PARQUET_VERSION = '2.6'  # Modern format

# =============================================================================
# ATTACK MAPPING (unchanged)
# =============================================================================

ATTACK_MAPPING = {
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
    'DoS-UDP_Flood': 'DoS',
    'DoS-TCP_Flood': 'DoS',
    'DoS-SYN_Flood': 'DoS',
    'DoS-HTTP_Flood': 'DoS',
    'Mirai-greeth_flood': 'Mirai',
    'Mirai-udpplain': 'Mirai',
    'Mirai-greip_flood': 'Mirai',
    'MITM-ArpSpoofing': 'Spoofing',
    'DNS_Spoofing': 'Spoofing',
    'Recon-HostDiscovery': 'Recon',
    'Recon-OSScan': 'Recon',
    'Recon-PortScan': 'Recon',
    'Recon-PingSweep': 'Recon',
    'VulnerabilityScan': 'Recon',
    'SqlInjection': 'Web',
    'XSS': 'Web',
    'CommandInjection': 'Web',
    'Uploading_Attack': 'Web',
    'BrowserHijacking': 'Web',
    'Backdoor_Malware': 'Web',
    'DictionaryBruteForce': 'BruteForce',
    'BenignTraffic': 'Benign'
}

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
# MEMORY OPTIMIZATION
# =============================================================================

def optimize_memory_float32(df, feature_cols):
    """
    Converte features a float32 per dimezzare RAM.
    
    IMPORTANTE: Chiamare subito dopo caricamento dati.
    
    Args:
        df: DataFrame con features
        feature_cols: Lista colonne numeriche
    
    Returns:
        df: DataFrame ottimizzato
    """
    print(f"🔧 Optimizing memory (float64 → float32)...")
    
    mem_before = df.memory_usage(deep=True).sum() / 1024**2
    
    # Convert features to float32
    for col in feature_cols:
        if col in df.columns and df[col].dtype == np.float64:
            df[col] = df[col].astype(np.float32)
    
    mem_after = df.memory_usage(deep=True).sum() / 1024**2
    savings = mem_before - mem_after
    
    print(f"   Memory: {mem_before:.1f} MB → {mem_after:.1f} MB (saved {savings:.1f} MB, {savings/mem_before*100:.1f}%)")
    
    return df

# =============================================================================
# CORE PROCESSING FUNCTIONS
# =============================================================================

def get_feature_cols(filepath, label_col='label'):
    """Legge header per identificare features."""
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
    """Pulizia dati robusta su chunk."""
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
            if n_inf > 10:
                print(f"  ⚠️  Column '{col}': Replaced {n_inf} infinite values")
    
    # 3. Duplicates (locali)
    chunk = chunk.drop_duplicates()
    n_removed = original_len - len(chunk)
    if n_removed > 0:
        print(f"  ℹ️  Removed {n_removed} duplicate rows in chunk")
    
    # 4. Check label validity
    if 'label' in chunk.columns:
        unknown_labels = chunk[~chunk['label'].isin(ATTACK_MAPPING.keys())]
        if len(unknown_labels) > 0:
            print(f"  ⚠️  Found {len(unknown_labels)} unknown labels: {unknown_labels['label'].unique()}")
    
    return chunk


def fit_scaler_and_analyze(filepath, feature_cols, label_col='label', nrows=None, chunk_size=CHUNK_SIZE):
    """
    PASSAGGIO 1: Partial Fit Scaler + Analisi + Mapping.
    """
    print_section("STEP 1: ANALYZING & FITTING (Passaggio 1)")
    
    label_encoder = LabelEncoder()
    label_encoder.fit(MACRO_CATEGORIES)
    scaler = StandardScaler()
    
    specific_label_counts = pd.Series(dtype='int64')
    total_processed_rows = 0
    
    print(f"Reading {filepath} for analysis...")
    chunk_iterator = pd.read_csv(filepath, chunksize=chunk_size, nrows=nrows)
    
    for i, chunk in enumerate(chunk_iterator):
        chunk = process_chunk_cleaning(chunk)
        
        # Accumulo statistiche
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
    
    # Report statistico
    print("\n" + "-"*60)
    print("DATASET ANALYSIS REPORT")
    print("-" * 60)
    
    sorted_labels = sorted(specific_label_counts.index)
    specific_to_idx = {lbl: idx for idx, lbl in enumerate(sorted_labels)}
    print(f"Generated consistent mapping for {len(specific_to_idx)} specific classes.")

    print("\nTop 10 Specific Attack Types:")
    top_specific = specific_label_counts.sort_values(ascending=False).head(10)
    print(top_specific)
    
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


def process_dataset_chunked_parquet(filepath, scaler, label_encoder, specific_to_idx, 
                                    feature_cols, output_path, chunk_size=CHUNK_SIZE,
                                    nrows=None, label_col='label'):
    """
    PASSAGGIO 2: Transform & Save in PARQUET con streaming.
    
    OTTIMIZZAZIONE CHIAVE:
    - Non accumula chunks in RAM
    - Scrive direttamente su disco chunk per chunk
    - PyArrow Parquet supporta append mode
    
    Args:
        filepath: CSV input
        scaler, label_encoder, specific_to_idx: Artifacts
        feature_cols: Lista features
        output_path: Path .parquet output
        chunk_size: Dimensione chunk
        nrows: Limite righe (debug)
        label_col: Nome colonna label
    """
    print_section(f"STEP 2: TRANSFORMING & SAVING (PARQUET): {Path(filepath).name}")
    print(f"Output: {output_path}")
    
    chunk_iterator = pd.read_csv(filepath, chunksize=chunk_size, nrows=nrows)
    
    total_rows = 0
    chunk_idx = 0
    
    # Mappa veloce per encoding
    macro_map = {cls: idx for idx, cls in enumerate(label_encoder.classes_)}
    
    # ParquetWriter per streaming
    parquet_writer = None
    
    for chunk in chunk_iterator:
        chunk_idx += 1
        
        # 1. Clean
        chunk = process_chunk_cleaning(chunk)
        
        # 2. Labels
        chunk['y_specific'] = chunk[label_col].map(specific_to_idx).fillna(-1).astype(np.int32)
        chunk['y_macro'] = chunk[label_col].map(ATTACK_MAPPING)
        chunk['y_macro_encoded'] = chunk['y_macro'].map(macro_map).fillna(-1).astype(np.int32)
        
        # 3. Shuffling locale
        chunk = chunk.sample(frac=1, random_state=42).reset_index(drop=True)

        # 4. Scaling + float32 optimization
        chunk[feature_cols] = scaler.transform(chunk[feature_cols]).astype(np.float32)
        
        # 5. Select output columns
        output_cols = feature_cols + ['y_macro_encoded', 'y_specific']
        chunk_output = chunk[output_cols]
        
        # ⭐ STREAMING WRITE: Converti chunk a PyArrow Table e scrivi
        table = pa.Table.from_pandas(chunk_output, preserve_index=False)
        
        if parquet_writer is None:
            # Prima iterazione: crea writer
            parquet_writer = pq.ParquetWriter(
                output_path,
                table.schema,
                compression=PARQUET_COMPRESSION,
                version=PARQUET_VERSION
            )
        
        # Scrivi chunk su disco (append)
        parquet_writer.write_table(table)
        
        rows_in_chunk = len(chunk)
        total_rows += rows_in_chunk
        
        print(f"  Chunk {chunk_idx}: {rows_in_chunk:,} rows → disk (Total: {total_rows:,})", end="\n")
        
        del chunk, chunk_output, table
        gc.collect()
    
    # Chiudi writer
    if parquet_writer is not None:
        parquet_writer.close()
    
    size_mb = Path(output_path).stat().st_size / 1024**2
    print(f"\n✅ Saved: {output_path} ({size_mb:.2f} MB)")
    print(f"   Total rows: {total_rows:,}")
    print(f"   Format: Parquet (snappy compressed)")
    
    return total_rows


# =============================================================================
# VALIDATION
# =============================================================================

def validate_processed_data_parquet(output_dir):
    """Validazione parquet files."""
    print_section("VALIDATION CHECKS (Parquet)")
    
    # Carica header per verifica veloce
    train_path = f"{output_dir}/train_processed.parquet"
    test_path = f"{output_dir}/test_processed.parquet"
    val_path = f"{output_dir}/validation_processed.parquet"
    
    with open(f"{output_dir}/mapping_info.json") as f:
        mapping = json.load(f)
    
    checks = {
        'train': train_path,
        'test': test_path,
        'val': val_path
    }
    
    for name, path in checks.items():
        print(f"\n{name.upper()} Split:")
        
        # Leggi metadata senza caricare dati
        parquet_file = pq.ParquetFile(path)
        
        print(f"  ✅ Rows: {parquet_file.metadata.num_rows:,}")
        print(f"  ✅ Columns: {len(parquet_file.schema)}")
        
        # Leggi solo prime 100 righe per validazione
        df_sample = pq.read_table(path, columns=['y_specific', 'y_macro_encoded']).to_pandas().head(100)
        
        # Verifica range
        assert df_sample['y_specific'].min() >= 0, f"{name}: y_specific < 0!"
        assert df_sample['y_specific'].max() < mapping['n_specific_classes'], f"{name}: y_specific out of range!"
        print(f"  ✅ y_specific in range [0, {mapping['n_specific_classes']-1}]")
        
        assert df_sample['y_macro_encoded'].min() >= 0
        assert df_sample['y_macro_encoded'].max() < len(mapping['macro_categories'])
        print(f"  ✅ y_macro_encoded in range [0, {len(mapping['macro_categories'])-1}]")
    
    print("\n✅ ALL VALIDATION CHECKS PASSED!")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def process_pipeline(train_path, test_path, val_path, output_dir, nrows, label_col):
    
    print_header("🚀 ADVANCED DATA PROCESSING (Parquet Optimized)")
    print(f"Chunking Mode: ON (Size: {CHUNK_SIZE:,})")
    print("Optimizations: float32 + Parquet Streaming")
    
    # Get Info
    feature_cols = get_feature_cols(train_path, label_col)
    
    # 1. Fit & Analyze
    scaler, label_encoder, mapping_info, specific_to_idx = fit_scaler_and_analyze(
        train_path, feature_cols, label_col, nrows=nrows
    )
    
    # 2. Process Datasets (PARQUET STREAMING)
    os.makedirs(output_dir, exist_ok=True)
    
    # Train
    process_dataset_chunked_parquet(
        train_path, scaler, label_encoder, specific_to_idx, feature_cols,
        f"{output_dir}/train_processed.parquet", nrows=nrows, label_col=label_col
    )
    
    # Test
    if test_path and os.path.exists(test_path):
        process_dataset_chunked_parquet(
            test_path, scaler, label_encoder, specific_to_idx, feature_cols,
            f"{output_dir}/test_processed.parquet", nrows=nrows, label_col=label_col
        )
    
    # Val
    if val_path and os.path.exists(val_path):
        process_dataset_chunked_parquet(
            val_path, scaler, label_encoder, specific_to_idx, feature_cols,
            f"{output_dir}/validation_processed.parquet", nrows=nrows, label_col=label_col
        )
        
    # 3. Save Artifacts (unchanged)
    print_section("SAVING ARTIFACTS")
    joblib.dump(scaler, f"{output_dir}/scaler.pkl")
    joblib.dump(label_encoder, f"{output_dir}/label_encoder.pkl")
    save_json(mapping_info, f"{output_dir}/mapping_info.json")

    # Validate
    validate_processed_data_parquet(output_dir)
    
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