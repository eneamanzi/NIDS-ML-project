"""
Borderline-SMOTE Data Augmentation - PARQUET OPTIMIZED

OTTIMIZZAZIONI v2.0:
- INPUT/OUTPUT: .pkl → .parquet
- MEMORY: float32 conversion (2 punti: post-load, post-SMOTE)
- STREAMING: Chunked parquet write (no full dataset in RAM)

Usage:
    python src/borderline_smote.py --input-dir data/processed/CICIOT23
                                    --output-dir data/processed/BorderlineSMOTE
                                    --target-sample 800000
"""

import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from imblearn.over_sampling import BorderlineSMOTE
import json
import gc
import warnings
import pyarrow as pa
import pyarrow.parquet as pq

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
CICIOT_DIR = DATA_DIR / "CICIOT23"

DEFAULT_TARGET_SAMPLE = 800_000
DEFAULT_MIN_MINORITY = 50_000

PARQUET_COMPRESSION = 'snappy'
PARQUET_VERSION = '2.6'

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

def optimize_memory_float32(X, description="data"):
    """
    Converte array/DataFrame a float32.
    
    CHIAMARE:
    1. Subito dopo caricamento dati
    2. Subito dopo generazione synthetic
    
    Args:
        X: numpy array o DataFrame
        description: Descrizione per log
    
    Returns:
        X ottimizzato (float32)
    """
    print(f"🔧 Optimizing {description} memory (float64 → float32)...")
    
    if isinstance(X, pd.DataFrame):
        mem_before = X.memory_usage(deep=True).sum() / 1024**2
        X = X.astype({col: np.float32 for col in X.select_dtypes(include=[np.float64]).columns})
        mem_after = X.memory_usage(deep=True).sum() / 1024**2
    else:  # numpy array
        mem_before = X.nbytes / 1024**2
        X = X.astype(np.float32)
        mem_after = X.nbytes / 1024**2
    
    savings = mem_before - mem_after
    print(f"   {description.capitalize()}: {mem_before:.1f} MB → {mem_after:.1f} MB (saved {savings:.1f} MB)")
    
    return X

# =============================================================================
# SMART SAMPLING
# =============================================================================

def extract_smart_stratified_sample(df, y_col, target_total=DEFAULT_TARGET_SAMPLE,
                                   min_minority=DEFAULT_MIN_MINORITY, random_state=42):
    """Smart stratified sampling (unchanged logic)."""
    print_section("SMART STRATIFIED SAMPLING")
    
    total_rows = len(df)
    class_counts = df[y_col].value_counts().sort_values()
    
    print(f"Original dataset: {total_rows:,} rows")
    print(f"Target sample: {target_total:,} rows")
    
    samples_per_class = {}
    budget_used = 0
    minority_classes = []
    majority_classes = []
    
    threshold_minority = min_minority * 2
    
    for cls, count in class_counts.items():
        if count <= threshold_minority:
            n_sample = min(count, min_minority)
            samples_per_class[cls] = n_sample
            minority_classes.append(cls)
            budget_used += n_sample
        else:
            majority_classes.append(cls)
    
    remaining_budget = target_total - budget_used
    majority_total = sum(class_counts[cls] for cls in majority_classes)
    
    for cls in majority_classes:
        proportion = class_counts[cls] / majority_total
        n_sample = int(remaining_budget * proportion)
        n_sample = min(n_sample, class_counts[cls])
        samples_per_class[cls] = n_sample
    
    sampled_dfs = []
    for cls, n_sample in samples_per_class.items():
        df_cls = df[df[y_col] == cls]
        if n_sample >= len(df_cls):
            sampled_dfs.append(df_cls)
        else:
            sampled_dfs.append(df_cls.sample(n=n_sample, random_state=random_state))
    
    df_sample = pd.concat(sampled_dfs, ignore_index=True)
    df_sample = df_sample.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    print(f"\n✅ Sample extracted: {len(df_sample):,} rows")
    
    gc.collect()
    return df_sample

# =============================================================================
# BORDERLINE-SMOTE APPLICATION
# =============================================================================

def apply_borderline_smote(X_train, y_train, sampling_strategy='auto',
                           k_neighbors=5, m_neighbors=10, kind='borderline-1',
                           random_state=42):
    """
    Applica Borderline-SMOTE con ottimizzazione float32.
    """
    print_section("APPLYING BORDERLINE-SMOTE")
    
    print(f"Parameters:")
    print(f"  sampling_strategy: {sampling_strategy}")
    print(f"  k_neighbors: {k_neighbors}")
    print(f"  m_neighbors: {m_neighbors}")
    print(f"  kind: {kind}")
    
    # Adjust neighbors
    class_counts = np.bincount(y_train)
    min_samples = class_counts[class_counts > 0].min()
    
    if min_samples <= k_neighbors:
        k_neighbors = max(1, min_samples - 1)
        print(f"  ⚠️ Adjusted k_neighbors to {k_neighbors}")
    
    if min_samples <= m_neighbors:
        m_neighbors = max(1, min_samples - 1)
        print(f"  ⚠️ Adjusted m_neighbors to {m_neighbors}")
    
    print(f"\nInput:")
    print(f"  Samples: {len(X_train):,}")
    print(f"  Features: {X_train.shape[1]}")
    
    # ⭐ OPTIMIZATION 1: Convert to float32 BEFORE SMOTE
    X_train = optimize_memory_float32(X_train, "input features")
    
    import psutil
    available_mb = psutil.virtual_memory().available / 1024**2
    print(f"  Available RAM: {available_mb:.0f} MB")
    
    print(f"\n⏳ Applying Borderline-SMOTE...")
    
    try:
        borderline_smote = BorderlineSMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=k_neighbors,
            m_neighbors=m_neighbors,
            kind=kind,
            random_state=random_state
        )
        
        X_smote, y_smote = borderline_smote.fit_resample(X_train, y_train)
        
        # ⭐ OPTIMIZATION 2: Convert synthetic to float32 immediately
        X_smote = optimize_memory_float32(X_smote, "SMOTE output")
        
        print(f"\n✅ Borderline-SMOTE completed!")
        print(f"\nOutput:")
        print(f"  Samples: {len(X_smote):,} (+{len(X_smote)-len(X_train):,})")
        print(f"  Synthetic: {len(X_smote)-len(X_train):,}")
        
        gc.collect()
        return X_smote, y_smote
        
    except MemoryError as e:
        print(f"\n❌ MEMORY ERROR: {e}")
        return X_train, y_train
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return X_train, y_train

# =============================================================================
# EXTRACTION & MERGE
# =============================================================================

def extract_synthetic_only(X_smote, y_smote, X_sample_original, feature_cols):
    """Estrae synthetic samples."""
    print_section("EXTRACTING SYNTHETIC SAMPLES")
    
    n_original = len(X_sample_original)
    n_smote = len(X_smote)
    n_synthetic = n_smote - n_original
    
    print(f"Breakdown:")
    print(f"  Total output: {n_smote:,} rows")
    print(f"  Original: {n_original:,} rows")
    print(f"  Synthetic: {n_synthetic:,} rows")
    
    if n_synthetic <= 0:
        print(f"\n⚠️ No synthetic samples generated!")
        return pd.DataFrame()
    
    X_synthetic = X_smote[n_original:]
    y_synthetic = y_smote[n_original:]
    
    df_synthetic = pd.DataFrame(X_synthetic, columns=feature_cols)
    df_synthetic['y_macro_encoded'] = y_synthetic
    df_synthetic['y_specific'] = y_synthetic
    
    print(f"\n✅ Extracted {len(df_synthetic):,} synthetic samples")
    
    return df_synthetic

def merge_synthetic_with_original_streaming(df_original_path, df_synthetic, 
                                            output_path, feature_cols):
    """
    Merge con STREAMING WRITE (no full dataset in RAM).
    
    STRATEGIA:
    1. Scrivi original su disco (read chunks → write chunks)
    2. Append synthetic
    3. Shuffle finale (opzionale, costoso)
    
    Args:
        df_original_path: Path parquet original
        df_synthetic: DataFrame synthetic (già in RAM)
        output_path: Path parquet output
        feature_cols: Colonne da mantenere
    """
    print_section("MERGING (STREAMING)")
    
    print(f"Original path: {df_original_path}")
    print(f"Synthetic rows: {len(df_synthetic):,}")
    
    output_cols = feature_cols + ['y_macro_encoded', 'y_specific']
    
    # Setup ParquetWriter
    parquet_writer = None
    
    # FASE 1: Stream original chunks
    print(f"\nPhase 1: Streaming original dataset...")
    
    parquet_file = pq.ParquetFile(df_original_path)
    total_original = 0
    
    for batch in parquet_file.iter_batches(batch_size=100_000):
        df_batch = batch.to_pandas()[output_cols]
        
        # ⭐ Optimize batch memory
        df_batch = optimize_memory_float32(df_batch, f"original batch")
        
        table = pa.Table.from_pandas(df_batch, preserve_index=False)
        
        if parquet_writer is None:
            parquet_writer = pq.ParquetWriter(
                output_path,
                table.schema,
                compression=PARQUET_COMPRESSION,
                version=PARQUET_VERSION
            )
        
        parquet_writer.write_table(table)
        total_original += len(df_batch)
        print(f"  Original: {total_original:,} rows written...", end="\r")
        
        del df_batch, table
        gc.collect()
    
    print(f"\n  ✅ Original: {total_original:,} rows written")
    
    # FASE 2: Append synthetic
    print(f"\nPhase 2: Appending synthetic...")
    
    if len(df_synthetic) > 0:
        # ⭐ Optimize synthetic memory
        df_synthetic = optimize_memory_float32(df_synthetic, "synthetic")
        
        table_synthetic = pa.Table.from_pandas(df_synthetic[output_cols], preserve_index=False)
        parquet_writer.write_table(table_synthetic)
        print(f"  ✅ Synthetic: {len(df_synthetic):,} rows appended")
    
    # Close writer
    parquet_writer.close()
    
    total_final = total_original + len(df_synthetic)
    size_mb = Path(output_path).stat().st_size / 1024**2
    
    print(f"\n✅ Merged dataset saved: {output_path}")
    print(f"   Total rows: {total_final:,}")
    print(f"   Size: {size_mb:.2f} MB")
    
    gc.collect()
    return total_final

# =============================================================================
# SAVE & INFO
# =============================================================================

def create_info(original_dist, sample_dist, smote_dist, final_dist,
               label_encoder, output_dir):
    """Salva info."""
    class_names = label_encoder.classes_
    
    info = {
        'method': 'Borderline-SMOTE',
        'description': 'Smart sampling + Borderline-SMOTE (boundary focus)',
        'format': 'parquet',
        'compression': PARQUET_COMPRESSION,
        'original_distribution': {
            class_names[cls]: int(count) for cls, count in original_dist.items()
        },
        'sample_distribution': {
            class_names[cls]: int(count) for cls, count in sample_dist.items()
        },
        'smote_distribution': {
            class_names[cls]: int(count) for cls, count in smote_dist.items()
        },
        'final_distribution': {
            class_names[cls]: int(count) for cls, count in final_dist.items()
        }
    }
    
    info_path = f"{output_dir}/augmentation_info.json"
    save_json(info, info_path)

def analyze_class_distribution(y, label_encoder, dataset_name="Dataset"):
    """Analizza distribuzione."""
    unique, counts = np.unique(y, return_counts=True)
    class_names = label_encoder.classes_
    
    print(f"\n{dataset_name}:")
    print("-" * 60)
    
    total = len(y)
    for cls_idx, count in zip(unique, counts):
        cls_name = class_names[cls_idx] if cls_idx < len(class_names) else f"Class_{cls_idx}"
        pct = count / total * 100
        print(f"  {cls_name:<15} {count:>10,} ({pct:>6.2f}%)")
    
    print("-" * 60)
    
    return dict(zip(unique, counts))

# =============================================================================
# MAIN
# =============================================================================

def main(input_dir, output_dir, target_sample=DEFAULT_TARGET_SAMPLE,
         min_minority=DEFAULT_MIN_MINORITY, sampling_strategy='auto',
         k_neighbors=5, m_neighbors=10, kind='borderline-1', random_state=42):
    
    print_header("🎯 BORDERLINE-SMOTE (Parquet Optimized)")
    
    print(f"Configuration:")
    print(f"  Method: Borderline-SMOTE")
    print(f"  Target sample: {target_sample:,} rows")
    print(f"  Format: Parquet (streaming)")
    
    # Load
    print_header("STEP 1: LOADING DATASET")
    
    train_path = f"{input_dir}/train_processed.parquet"
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Not found: {train_path}")
    
    # Leggi metadata
    parquet_file = pq.ParquetFile(train_path)
    print(f"✅ Found: {train_path}")
    print(f"   Rows: {parquet_file.metadata.num_rows:,}")
    
    # Carica tutto (per sampling - unavoidable)
    df_train = pq.read_table(train_path).to_pandas()
    print(f"   Loaded in RAM for sampling")
    
    # ⭐ OPTIMIZATION 1: float32 subito dopo load
    feature_cols = [col for col in df_train.columns
                   if col not in ['y_macro_encoded', 'y_specific']]
    df_train = optimize_memory_float32(df_train, "loaded train data")
    
    encoder_path = f"{input_dir}/label_encoder.pkl"
    label_encoder = joblib.load(encoder_path)
    print(f"✅ Label encoder: {label_encoder.classes_}")
    
    # Analyze original
    print_header("STEP 2: ORIGINAL DISTRIBUTION")
    y_train = df_train['y_macro_encoded'].values
    original_dist = analyze_class_distribution(y_train, label_encoder, "ORIGINAL")
    
    # Smart sample
    print_header("STEP 3: SMART SAMPLING")
    df_sample = extract_smart_stratified_sample(
        df_train, 'y_macro_encoded', target_sample, min_minority, random_state
    )
    
    X_sample = df_sample[feature_cols].values
    y_sample = df_sample['y_macro_encoded'].values
    
    sample_dist = analyze_class_distribution(y_sample, label_encoder, "SAMPLE")
    
    X_sample_original = X_sample.copy()
    
    del df_sample
    gc.collect()
    
    # Apply Borderline-SMOTE
    print_header("STEP 4: BORDERLINE-SMOTE")
    
    X_smote, y_smote = apply_borderline_smote(
        X_sample, y_sample, sampling_strategy, k_neighbors, m_neighbors, kind, random_state
    )
    
    del X_sample, y_sample
    gc.collect()
    
    smote_dist = analyze_class_distribution(y_smote, label_encoder, "BORDERLINE-SMOTE")
    
    # Extract synthetic
    print_header("STEP 5: EXTRACTING SYNTHETIC")
    
    df_synthetic = extract_synthetic_only(X_smote, y_smote, X_sample_original, feature_cols)
    
    del X_smote, y_smote, X_sample_original
    gc.collect()
    
    # Merge (STREAMING)
    print_header("STEP 6: MERGING (STREAMING)")
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/train_borderline_smote.parquet"
    
    total_final = merge_synthetic_with_original_streaming(
        train_path, df_synthetic, output_path, feature_cols
    )
    
    del df_train
    gc.collect()
    
    # Analyze final (sample from parquet for speed)
    print_header("STEP 7: FINAL ANALYSIS")
    df_final_sample = pq.read_table(output_path, columns=['y_macro_encoded']).to_pandas().sample(n=50000, random_state=42)
    final_dist = analyze_class_distribution(
        df_final_sample['y_macro_encoded'].values, label_encoder, "FINAL (sampled)"
    )
    
    # Save info
    print_header("STEP 8: SAVING METADATA")
    
    create_info(original_dist, sample_dist, smote_dist, final_dist,
               label_encoder, output_dir)
    
    # Copy artifacts
    import shutil
    for artifact in ['scaler.pkl', 'label_encoder.pkl', 'mapping_info.json']:
        src = Path(input_dir) / artifact
        dst = Path(output_dir) / artifact
        if src.exists():
            shutil.copy(src, dst)
            print(f"  ✅ Copied: {artifact}")
    
    # Summary
    print_header("✅ COMPLETE!")
    
    print("Summary:")
    print(f"  Output: {output_path}")
    print(f"  Format: Parquet (snappy compressed)")
    print(f"  Total rows: {total_final:,}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Borderline-SMOTE (Parquet)')
    parser.add_argument('--input-dir', type=str, default='data/processed/CICIOT23')
    parser.add_argument('--output-dir', type=str, default='data/processed/BorderlineSMOTE')
    parser.add_argument('--target-sample', type=int, default=DEFAULT_TARGET_SAMPLE)
    parser.add_argument('--min-minority', type=int, default=DEFAULT_MIN_MINORITY)
    parser.add_argument('--sampling-strategy', type=str, default='auto')
    parser.add_argument('--k-neighbors', type=int, default=5)
    parser.add_argument('--m-neighbors', type=int, default=10)
    parser.add_argument('--kind', type=str, default='borderline-1',
                       choices=['borderline-1', 'borderline-2'])
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"❌ Not found: {args.input_dir}")
        exit(1)
    
    main(args.input_dir, args.output_dir, args.target_sample,
         args.min_minority, args.sampling_strategy, args.k_neighbors,
         args.m_neighbors, args.kind, args.seed)