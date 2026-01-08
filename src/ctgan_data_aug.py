"""
CTGAN (Conditional Tabular GAN) - PARQUET OPTIMIZED + SMART MINORITY SAMPLING

OTTIMIZZAZIONI v3.0 (FINALE):
- INPUT/OUTPUT: .parquet
- MEMORY: float32 conversion
- STREAMING: Chunked write
- SMART SAMPLING: 100% minority + proportional majority (BEST PRACTICE!)

STRATEGIA SAMPLING INTELLIGENTE:
1. Identifica classi minoritarie (threshold configurabile)
2. Prendi 100% delle minoritarie (preserva distribuzione rara)
3. Sample proporzionalmente le maggioritarie
4. CTGAN impara meglio con più esempi reali delle classi rare

Usage:
    python src/ctgan_data_aug.py --input-dir data/processed/CICIOT23 \
                        --minority-threshold-pct 1.0 \
                        --minority-threshold-abs 10000 \
                        --subset-size 200000 --epochs 300
"""

import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
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

DEFAULT_SUBSET_SIZE = 100_000
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 500

# MINORITY CLASS THRESHOLDS
DEFAULT_MINORITY_THRESHOLD_PCT = 1.0    # <1% del dataset = minoritaria
DEFAULT_MINORITY_THRESHOLD_ABS = 10_000  # <10k samples = minoritaria

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
    """Converte a float32 per dimezzare RAM."""
    print(f"🔧 Optimizing {description} memory (float64 → float32)...")
    
    if isinstance(X, pd.DataFrame):
        mem_before = X.memory_usage(deep=True).sum() / 1024**2
        float64_cols = X.select_dtypes(include=[np.float64]).columns
        X = X.astype({col: np.float32 for col in float64_cols})
        mem_after = X.memory_usage(deep=True).sum() / 1024**2
    else:  # numpy array
        mem_before = X.nbytes / 1024**2
        if X.dtype == np.float64:
            X = X.astype(np.float32)
        mem_after = X.nbytes / 1024**2
    
    savings = mem_before - mem_after
    if savings > 0:
        print(f"   {description.capitalize()}: {mem_before:.1f} MB → {mem_after:.1f} MB "
              f"(saved {savings:.1f} MB, {savings/mem_before*100:.1f}%)")
    else:
        print(f"   {description.capitalize()}: {mem_before:.1f} MB (already optimized)")
    
    return X

# =============================================================================
# CTGAN IMPORT & SETUP
# =============================================================================

def check_ctgan_installed():
    """Verifica CTGAN installato."""
    try:
        from ctgan import CTGAN
        print("✅ CTGAN library found")
        return True
    except ImportError:
        print("❌ CTGAN not installed!")
        print("\nInstall with:")
        print("  pip install ctgan")
        return False

def check_gpu_available():
    """Verifica GPU disponibile."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
            return True
        else:
            print("⚠️ GPU not available, will use CPU (SLOW!)")
            return False
    except ImportError:
        print("⚠️ PyTorch not found, assuming CPU")
        return False

# =============================================================================
# SMART MINORITY SAMPLING (核心優化!)
# =============================================================================

def identify_minority_classes(df, y_col, threshold_pct, threshold_abs):
    """
    Identifica classi minoritarie usando DOPPIA SOGLIA.
    
    Una classe è MINORITARIA se:
    - Percentuale < threshold_pct% DEL DATASET
    - O conteggio < threshold_abs samples
    
    LOGICA: Cattura sia classi percentualmente rare CHE piccole in assoluto.
    
    Args:
        df: DataFrame completo
        y_col: Colonna label
        threshold_pct: Soglia percentuale (es. 1.0 = <1%)
        threshold_abs: Soglia assoluta (es. 10000 = <10k samples)
    
    Returns:
        minority_classes: Set di class labels minoritarie
        class_counts: Series con conteggi
    """
    print_section("IDENTIFYING MINORITY CLASSES")
    
    class_counts = df[y_col].value_counts().sort_values()
    total_rows = len(df)
    
    minority_classes = set()
    
    print(f"Thresholds:")
    print(f"  Percentage: <{threshold_pct:.1f}% of dataset")
    print(f"  Absolute: <{threshold_abs:,} samples")
    print(f"\nClass Analysis:")
    print("-" * 70)
    print(f"{'Class':<10} {'Count':>10} {'Percentage':>12} {'Status':>15}")
    print("-" * 70)
    
    for cls, count in class_counts.items():
        pct = (count / total_rows) * 100
        
        # Determina se minoritaria
        is_minority_pct = pct < threshold_pct
        is_minority_abs = count < threshold_abs
        is_minority = is_minority_pct or is_minority_abs
        
        if is_minority:
            minority_classes.add(cls)
            status = "🔴 MINORITY"
            reason = []
            if is_minority_pct:
                reason.append(f"<{threshold_pct}%")
            if is_minority_abs:
                reason.append(f"<{threshold_abs:,}")
            status += f" ({', '.join(reason)})"
        else:
            status = "✅ MAJORITY"
        
        print(f"{cls:<10} {count:>10,} {pct:>11.2f}% {status:>15}")
    
    print("-" * 70)
    print(f"\n📊 Summary:")
    print(f"   Total classes: {len(class_counts)}")
    print(f"   Minority classes: {len(minority_classes)}")
    print(f"   Majority classes: {len(class_counts) - len(minority_classes)}")
    
    if len(minority_classes) > 0:
        minority_samples = sum(class_counts[cls] for cls in minority_classes)
        print(f"   Minority samples: {minority_samples:,} ({minority_samples/total_rows*100:.2f}%)")
    
    return minority_classes, class_counts


def extract_smart_minority_aware_subset(df, y_col, subset_size, 
                                        minority_threshold_pct, 
                                        minority_threshold_abs,
                                        random_state=42):
    """
    SMART SUBSET EXTRACTION con focus su minority classes.
    
    STRATEGIA OTTIMALE:
    1. Identifica classi minoritarie (doppia soglia)
    2. Prendi 100% delle minoritarie (preserva distribuzione rara!)
    3. Usa budget rimanente per sample proporzionale delle maggioritarie
    4. CTGAN impara meglio con più real examples delle classi rare
    
    Args:
        df: DataFrame completo
        y_col: Colonna label
        subset_size: Target size totale
        minority_threshold_pct: Soglia % minoritaria
        minority_threshold_abs: Soglia assoluta minoritaria
        random_state: Seed
    
    Returns:
        df_subset: DataFrame subset ottimizzato
    """
    print_section("SMART MINORITY-AWARE SUBSET EXTRACTION")
    
    total_rows = len(df)
    print(f"Original dataset: {total_rows:,} rows")
    print(f"Target subset: {subset_size:,} rows")
    
    # Identifica minoritarie
    minority_classes, class_counts = identify_minority_classes(
        df, y_col, minority_threshold_pct, minority_threshold_abs
    )
    
    # FASE 1: Prendi TUTTO dalle minoritarie
    print(f"\n📥 Phase 1: Extracting 100% of MINORITY classes...")
    
    minority_dfs = []
    minority_budget = 0
    
    for cls in minority_classes:
        df_cls = df[df[y_col] == cls]
        n_samples = len(df_cls)
        minority_dfs.append(df_cls)
        minority_budget += n_samples
        print(f"  Class {cls}: {n_samples:,} samples (100%)")
    
    print(f"\n  ✅ Minority total: {minority_budget:,} samples")
    
    # FASE 2: Sample proporzionalmente le maggioritarie
    remaining_budget = subset_size - minority_budget
    
    if remaining_budget <= 0:
        print(f"\n⚠️  Warning: Minority classes exceed target!")
        print(f"   Adjusting target to {minority_budget:,}")
        remaining_budget = 0
    
    majority_classes = [cls for cls in class_counts.index if cls not in minority_classes]
    
    print(f"\n📥 Phase 2: Sampling MAJORITY classes proportionally...")
    print(f"  Remaining budget: {remaining_budget:,} samples")
    
    majority_dfs = []
    majority_total = sum(class_counts[cls] for cls in majority_classes)
    
    for cls in majority_classes:
        df_cls = df[df[y_col] == cls]
        
        # Proporzione relativa tra maggioritarie
        proportion = class_counts[cls] / majority_total
        n_sample = int(remaining_budget * proportion)
        n_sample = min(n_sample, len(df_cls))
        
        if n_sample > 0:
            sampled = df_cls.sample(n=n_sample, random_state=random_state)
            majority_dfs.append(sampled)
            
            sample_pct = (n_sample / len(df_cls)) * 100
            print(f"  Class {cls}: {n_sample:,} / {len(df_cls):,} ({sample_pct:.1f}%)")
    
    # MERGE
    all_dfs = minority_dfs + majority_dfs
    df_subset = pd.concat(all_dfs, ignore_index=True)
    df_subset = df_subset.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    print(f"\n✅ Smart subset extracted: {len(df_subset):,} rows")
    
    # Report finale
    print(f"\n📊 Final subset distribution:")
    subset_counts = df_subset[y_col].value_counts()
    print("-" * 60)
    
    for cls in class_counts.index:
        if cls in subset_counts.index:
            original = class_counts[cls]
            subset = subset_counts[cls]
            coverage = (subset / original) * 100
            
            status = "🔴 MINORITY (100%)" if cls in minority_classes else f"✅ MAJORITY ({coverage:.1f}%)"
            print(f"  Class {cls}: {subset:>7,} / {original:>10,} - {status}")
    
    print("-" * 60)
    
    gc.collect()
    return df_subset


# =============================================================================
# CTGAN TRAINING
# =============================================================================

def train_ctgan(df_train, feature_cols, label_col, epochs=DEFAULT_EPOCHS,
                batch_size=DEFAULT_BATCH_SIZE, use_gpu=False, random_state=42):
    """Train CTGAN su subset ottimizzato."""
    print_section("TRAINING CTGAN")
    
    from ctgan import CTGAN
    import torch
    
    print(f"Configuration:")
    print(f"  Training samples: {len(df_train):,}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Device: {'GPU (CUDA)' if use_gpu and torch.cuda.is_available() else 'CPU'}")
    
    # Optimize memory
    df_train = optimize_memory_float32(df_train, "training subset")
    
    discrete_columns = [label_col]
    train_data = df_train[feature_cols + [label_col]]
    
    print(f"\n⏳ Training CTGAN...")
    
    ctgan = CTGAN(
        epochs=epochs,
        batch_size=batch_size,
        verbose=True,
        cuda=use_gpu and torch.cuda.is_available()
    )
    
    import time
    start_time = time.time()
    
    try:
        ctgan.fit(train_data, discrete_columns=discrete_columns)
        
        elapsed = time.time() - start_time
        print(f"\n✅ CTGAN training complete!")
        print(f"   Time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        
        return ctgan
        
    except Exception as e:
        print(f"\n❌ CTGAN training failed: {e}")
        raise


# =============================================================================
# SYNTHETIC GENERATION
# =============================================================================

def generate_synthetic_samples(ctgan, label_encoder, target_per_class=None,
                               feature_cols=None, random_state=42):
    """Genera synthetic samples con CTGAN."""
    print_section("GENERATING SYNTHETIC SAMPLES")
    
    if target_per_class is None:
        n_classes = len(label_encoder.classes_)
        target_per_class = {i: 50000 for i in range(n_classes)}
    
    print(f"Target generation per class:")
    for cls_idx, n_samples in target_per_class.items():
        cls_name = label_encoder.classes_[cls_idx]
        print(f"  {cls_name:<15}: {n_samples:>7,} samples")
    
    total_to_generate = sum(target_per_class.values())
    print(f"\nTotal synthetic to generate: {total_to_generate:,}")
    
    synthetic_dfs = []
    
    for cls_idx, n_samples in target_per_class.items():
        if n_samples <= 0:
            continue
        
        cls_name = label_encoder.classes_[cls_idx]
        print(f"\n  Generating {n_samples:,} samples for {cls_name}...", end=" ")
        
        try:
            conditions = pd.DataFrame({
                'y_macro_encoded': [cls_idx] * n_samples
            })
            
            synthetic = ctgan.sample(n=n_samples, conditions=conditions)
            synthetic = optimize_memory_float32(synthetic, f"synthetic {cls_name}")
            synthetic_dfs.append(synthetic)
            print(f"✅")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    if not synthetic_dfs:
        raise RuntimeError("No synthetic samples generated!")
    
    df_synthetic = pd.concat(synthetic_dfs, ignore_index=True)
    df_synthetic = df_synthetic.sample(frac=1, random_state=random_state).reset_index(drop=True)
    df_synthetic['y_specific'] = df_synthetic['y_macro_encoded']
    
    print(f"\n✅ Generated {len(df_synthetic):,} synthetic samples")
    
    gc.collect()
    return df_synthetic


# =============================================================================
# MERGE & SAVE
# =============================================================================

def merge_synthetic_with_original_streaming(df_original_path, df_synthetic, 
                                            output_path, feature_cols):
    """Merge con streaming write."""
    print_section("MERGING (STREAMING)")
    
    print(f"Original path: {df_original_path}")
    print(f"Synthetic rows: {len(df_synthetic):,}")
    
    output_cols = feature_cols + ['y_macro_encoded', 'y_specific']
    df_synthetic = df_synthetic[output_cols]
    
    parquet_writer = None
    parquet_file = pq.ParquetFile(df_original_path)
    total_original = 0
    
    print(f"\nPhase 1: Streaming original...")
    
    for batch in parquet_file.iter_batches(batch_size=100_000):
        df_batch = batch.to_pandas()[output_cols]
        df_batch = optimize_memory_float32(df_batch, "batch")
        
        table = pa.Table.from_pandas(df_batch, preserve_index=False)
        
        if parquet_writer is None:
            parquet_writer = pq.ParquetWriter(
                output_path, table.schema,
                compression=PARQUET_COMPRESSION,
                version=PARQUET_VERSION
            )
        
        parquet_writer.write_table(table)
        total_original += len(df_batch)
        print(f"  Written: {total_original:,} rows...", end="\r")
        
        del df_batch, table
        gc.collect()
    
    print(f"\n  ✅ Original: {total_original:,} rows")
    
    print(f"\nPhase 2: Appending synthetic...")
    
    if len(df_synthetic) > 0:
        table_synthetic = pa.Table.from_pandas(df_synthetic, preserve_index=False)
        parquet_writer.write_table(table_synthetic)
        print(f"  ✅ Synthetic: {len(df_synthetic):,} rows")
    
    parquet_writer.close()
    
    total_final = total_original + len(df_synthetic)
    size_mb = Path(output_path).stat().st_size / 1024**2
    
    print(f"\n✅ Merged: {output_path}")
    print(f"   Total: {total_final:,} rows")
    print(f"   Size: {size_mb:.2f} MB")
    
    gc.collect()
    return total_final


def save_ctgan_model(ctgan, output_dir):
    """Salva modello CTGAN."""
    print(f"\nSaving CTGAN model...")
    
    model_path = f"{output_dir}/ctgan_model.pkl"
    
    try:
        import pickle
        with open(model_path, 'wb') as f:
            pickle.dump(ctgan, f)
        
        size_mb = Path(model_path).stat().st_size / 1024**2
        print(f"✅ Model saved: {model_path} ({size_mb:.2f} MB)")
    except Exception as e:
        print(f"⚠️ Could not save model: {e}")


def create_info(original_dist, subset_dist, final_dist, label_encoder,
               output_dir, ctgan_config):
    """Salva metadata."""
    class_names = label_encoder.classes_
    
    info = {
        'method': 'CTGAN',
        'description': 'Conditional Tabular GAN with smart minority sampling',
        'format': 'parquet',
        'compression': PARQUET_COMPRESSION,
        'ctgan_config': ctgan_config,
        'original_distribution': {
            class_names[cls]: int(count) for cls, count in original_dist.items()
        },
        'training_subset_distribution': {
            class_names[cls]: int(count) for cls, count in subset_dist.items()
        },
        'final_distribution': {
            class_names[cls]: int(count) for cls, count in final_dist.items()
        }
    }
    
    save_json(info, f"{output_dir}/augmentation_info.json")


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

def main(input_dir, output_dir, subset_size=DEFAULT_SUBSET_SIZE,
         minority_threshold_pct=DEFAULT_MINORITY_THRESHOLD_PCT,
         minority_threshold_abs=DEFAULT_MINORITY_THRESHOLD_ABS,
         epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH_SIZE,
         use_gpu=False, save_model=False, random_state=42):
    
    print_header("🤖 CTGAN - SMART MINORITY-AWARE AUGMENTATION")
    
    if not check_ctgan_installed():
        return
    
    gpu_available = check_gpu_available()
    
    print(f"\nConfiguration:")
    print(f"  Method: CTGAN (Conditional GAN)")
    print(f"  Subset strategy: 100% minority + proportional majority")
    print(f"  Minority threshold: <{minority_threshold_pct}% OR <{minority_threshold_abs:,} samples")
    print(f"  Target subset: {subset_size:,} rows")
    print(f"  Epochs: {epochs}")
    print(f"  Device: {'GPU' if use_gpu and gpu_available else 'CPU'}")
    
    # Load
    print_header("STEP 1: LOADING DATASET")
    
    train_path = f"{input_dir}/train_processed.parquet"
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Not found: {train_path}")
    
    parquet_file = pq.ParquetFile(train_path)
    print(f"✅ Found: {train_path}")
    print(f"   Rows: {parquet_file.metadata.num_rows:,}")
    
    df_train = pq.read_table(train_path).to_pandas()
    
    feature_cols = [col for col in df_train.columns
                   if col not in ['y_macro_encoded', 'y_specific']]
    df_train = optimize_memory_float32(df_train, "loaded data")
    
    encoder_path = f"{input_dir}/label_encoder.pkl"
    label_encoder = joblib.load(encoder_path)
    print(f"✅ Encoder: {label_encoder.classes_}")
    
    # Analyze original
    print_header("STEP 2: ORIGINAL DISTRIBUTION")
    y_train = df_train['y_macro_encoded'].values
    original_dist = analyze_class_distribution(y_train, label_encoder, "ORIGINAL")
    
    # Smart subset extraction
    print_header("STEP 3: SMART MINORITY-AWARE SUBSET")
    
    df_subset = extract_smart_minority_aware_subset(
        df_train, 'y_macro_encoded', subset_size,
        minority_threshold_pct, minority_threshold_abs, random_state
    )
    
    y_subset = df_subset['y_macro_encoded'].values
    subset_dist = analyze_class_distribution(y_subset, label_encoder, "SUBSET")
    
    # Train CTGAN
    print_header("STEP 4: TRAINING CTGAN")
    
    ctgan = train_ctgan(df_subset, feature_cols, 'y_macro_encoded',
                       epochs, batch_size, use_gpu, random_state)
    
    del df_subset
    gc.collect()
    
    # Generate synthetic
    print_header("STEP 5: GENERATING SYNTHETIC")
    
    class_counts = dict(zip(*np.unique(y_train, return_counts=True)))
    max_count = max(class_counts.values())
    
    target_per_class = {}
    for cls_idx, count in class_counts.items():
        target = int(max_count * 0.8) - count
        target = max(0, min(target, 100000))
        target_per_class[cls_idx] = target
    
    df_synthetic = generate_synthetic_samples(
        ctgan, label_encoder, target_per_class, feature_cols, random_state
    )
    
    # Merge
    print_header("STEP 6: MERGING")
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/train_ctgan.parquet"
    
    total_final = merge_synthetic_with_original_streaming(
        train_path, df_synthetic, output_path, feature_cols
    )
    
    del df_train
    gc.collect()
    
    # Final analysis
    print_header("STEP 7: FINAL ANALYSIS")
    df_final_sample = pq.read_table(
        output_path, columns=['y_macro_encoded']
    ).to_pandas().sample(n=50000, random_state=42)
    final_dist = analyze_class_distribution(
        df_final_sample['y_macro_encoded'].values, label_encoder, "FINAL"
    )
    
    # Save
    print_header("STEP 8: SAVING")
    
    if save_model:
        save_ctgan_model(ctgan, output_dir)
    
    ctgan_config = {
        'subset_size': subset_size,
        'minority_threshold_pct': minority_threshold_pct,
        'minority_threshold_abs': minority_threshold_abs,
        'epochs': epochs,
        'batch_size': batch_size,
        'device': 'GPU' if use_gpu and gpu_available else 'CPU'
    }
    
    create_info(original_dist, subset_dist, final_dist,
               label_encoder, output_dir, ctgan_config)
    
    import shutil
    for artifact in ['scaler.pkl', 'label_encoder.pkl', 'mapping_info.json']:
        src = Path(input_dir) / artifact
        dst = Path(output_dir) / artifact
        if src.exists():
            shutil.copy(src, dst)
            print(f"  ✅ Copied: {artifact}")
    
    print_header("✅ COMPLETE!")
    print(f"\nSummary:")
    print(f"  Output: {output_path}")
    print(f"  Total: {total_final:,} rows")
    print(f"  Strategy: 100% minority + proportional majority sampling")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='CTGAN with smart minority-aware sampling'
    )
    
    parser.add_argument('--input-dir', type=str, 
                       default=str(CICIOT_DIR))
    parser.add_argument('--output-dir', type=str, 
                       default=str(DATA_DIR / 'CTGAN'))
    parser.add_argument('--subset-size', type=int, 
                       default=DEFAULT_SUBSET_SIZE)
    parser.add_argument('--minority-threshold-pct', type=float,
                       default=DEFAULT_MINORITY_THRESHOLD_PCT,
                       help='Classes <X%% are minority (default: 1.0)')
    parser.add_argument('--minority-threshold-abs', type=int,
                       default=DEFAULT_MINORITY_THRESHOLD_ABS,
                       help='Classes <X samples are minority (default: 10000)')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--use-gpu', action='store_true')
    parser.add_argument('--save-model', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"❌ Not found: {args.input_dir}")
        exit(1)
    
    main(args.input_dir, args.output_dir, args.subset_size,
         args.minority_threshold_pct, args.minority_threshold_abs,
         args.epochs, args.batch_size, args.use_gpu, 
         args.save_model, args.seed)