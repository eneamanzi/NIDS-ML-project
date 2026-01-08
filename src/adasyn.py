"""
ADASYN (Adaptive Synthetic Sampling) per CICIoT2023.

VANTAGGI per Network Traffic:
- Genera più sample per classi "più difficili" (densità adaptive)
- Ottimo per minority classes estremamente piccole (BruteForce, Web)
- Bilancia non solo numero ma anche difficoltà di apprendimento

DIFFERENZA da Borderline-SMOTE:
- Borderline: Focus su boundary
- ADASYN: Focus su densità locale (più sample dove la classe è sparsa)

QUANDO USARLO:
- Se hai minority classes molto piccole (<1% del totale)
- Se alcune classi sono sparse nello spazio feature

Usage:
    python src/adasyn.py --input-dir data/processed/CICIOT23
                          --output-dir data/processed/ADASYN
                          --target-sample 800000
"""

import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from imblearn.over_sampling import ADASYN
import json
import gc
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
CICIOT_DIR = DATA_DIR / "CICIOT23"

DEFAULT_TARGET_SAMPLE = 800_000
DEFAULT_MIN_MINORITY = 50_000

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
# SMART SAMPLING
# =============================================================================

def extract_smart_stratified_sample(df, y_col, target_total=DEFAULT_TARGET_SAMPLE,
                                   min_minority=DEFAULT_MIN_MINORITY, random_state=42):
    """Smart stratified sampling."""
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
# ADASYN APPLICATION
# =============================================================================

def apply_adasyn(X_train, y_train, sampling_strategy='auto',
                n_neighbors=5, random_state=42):
    """
    Applica ADASYN (Adaptive Synthetic Sampling).
    
    LOGICA:
    1. Calcola densità locale per ogni minority sample
    2. Genera più synthetic dove densità è bassa (zone sparse)
    3. Adaptive: numero synthetic proporzionale alla difficoltà
    
    Args:
        X_train: Training features
        y_train: Training labels
        sampling_strategy: 'auto' o dict
        n_neighbors: Vicini per calcolare densità
        random_state: Seed
    
    Returns:
        X_adasyn, y_adasyn
    """
    print_section("APPLYING ADASYN")
    
    print(f"Parameters:")
    print(f"  sampling_strategy: {sampling_strategy}")
    print(f"  n_neighbors: {n_neighbors}")
    print(f"  random_state: {random_state}")
    
    # Adjust n_neighbors
    class_counts = np.bincount(y_train)
    min_samples = class_counts[class_counts > 0].min()
    
    if min_samples <= n_neighbors:
        n_neighbors = max(1, min_samples - 1)
        print(f"  ⚠️ Adjusted n_neighbors to {n_neighbors} (min class: {min_samples})")
    
    print(f"\nInput:")
    print(f"  Samples: {len(X_train):,}")
    print(f"  Features: {X_train.shape[1]}")
    
    # Optimize memory
    X_train = X_train.astype(np.float32)
    
    # Memory check
    import psutil
    available_mb = psutil.virtual_memory().available / 1024**2
    print(f"  Available RAM: {available_mb:.0f} MB")
    
    # Check class distribution for ADASYN suitability
    print(f"\nClass size analysis (ADASYN excels with small minorities):")
    for cls_idx, count in enumerate(class_counts):
        if count > 0:
            pct = count / len(y_train) * 100
            status = "⚠️ TINY" if pct < 1 else "✅ OK"
            print(f"  Class {cls_idx}: {count:>7,} ({pct:>5.2f}%) {status}")
    
    print(f"\n⏳ Applying ADASYN (calculating local densities)...")
    print(f"   ADASYN adapts to local difficulty (more samples in sparse areas)")
    
    try:
        adasyn = ADASYN(
            sampling_strategy=sampling_strategy,
            n_neighbors=n_neighbors,
            random_state=random_state
        )
        
        X_adasyn, y_adasyn = adasyn.fit_resample(X_train, y_train)
        
        print(f"\n✅ ADASYN completed!")
        print(f"\nOutput:")
        print(f"  Samples: {len(X_adasyn):,} (+{len(X_adasyn)-len(X_train):,})")
        print(f"  Synthetic: {len(X_adasyn)-len(X_train):,}")
        print(f"  Increase: {(len(X_adasyn)/len(X_train)-1)*100:.1f}%")
        
        # Mostra dove ADASYN ha generato più sample
        print(f"\nAdaptive generation analysis:")
        original_counts = np.bincount(y_train)
        adasyn_counts = np.bincount(y_adasyn)
        
        for cls_idx in range(len(original_counts)):
            if original_counts[cls_idx] > 0:
                orig = original_counts[cls_idx]
                ada = adasyn_counts[cls_idx]
                generated = ada - orig
                ratio = generated / orig if orig > 0 else 0
                print(f"  Class {cls_idx}: {orig:>7,} → {ada:>7,} (+{generated:>6,}, {ratio:>5.1f}x)")
        
        gc.collect()
        return X_adasyn, y_adasyn
        
    except MemoryError as e:
        print(f"\n❌ MEMORY ERROR: {e}")
        print(f"   Reduce --target-sample or --min-minority")
        return X_train, y_train
    except ValueError as e:
        print(f"\n❌ ADASYN Error: {e}")
        print(f"   This can happen if classes are too small or too close")
        print(f"   Returning original data...")
        return X_train, y_train
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print(f"   Returning original data...")
        return X_train, y_train

# =============================================================================
# EXTRACTION & MERGE
# =============================================================================

def extract_synthetic_only(X_adasyn, y_adasyn, X_sample_original, feature_cols):
    """Estrae solo synthetic samples."""
    print_section("EXTRACTING SYNTHETIC SAMPLES")
    
    n_original = len(X_sample_original)
    n_adasyn = len(X_adasyn)
    n_synthetic = n_adasyn - n_original
    
    print(f"Breakdown:")
    print(f"  Total output: {n_adasyn:,} rows")
    print(f"  Original: {n_original:,} rows")
    print(f"  Synthetic: {n_synthetic:,} rows")
    
    if n_synthetic <= 0:
        print(f"\n⚠️ No synthetic samples generated!")
        return pd.DataFrame()
    
    X_synthetic = X_adasyn[n_original:]
    y_synthetic = y_adasyn[n_original:]
    
    df_synthetic = pd.DataFrame(X_synthetic, columns=feature_cols)
    df_synthetic['y_macro_encoded'] = y_synthetic
    df_synthetic['y_specific'] = y_synthetic
    
    print(f"\n✅ Extracted {len(df_synthetic):,} synthetic samples")
    
    return df_synthetic

def merge_synthetic_with_original(df_original, df_synthetic):
    """Merge synthetic con original."""
    print_section("MERGING")
    
    print(f"Original: {len(df_original):,} rows")
    print(f"Synthetic: {len(df_synthetic):,} rows")
    
    if len(df_synthetic) == 0:
        print(f"\n⚠️ No synthetic to merge. Returning original.")
        return df_original
    
    df_merged = pd.concat([df_original, df_synthetic], ignore_index=True)
    df_merged = df_merged.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\n✅ Merged: {len(df_merged):,} rows")
    
    gc.collect()
    return df_merged

# =============================================================================
# SAVE & INFO
# =============================================================================

def save_dataset(df_final, output_dir):
    """Salva dataset finale."""
    print_section("SAVING DATASET")
    
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = f"{output_dir}/train_adasyn.pkl"
    df_final.to_pickle(train_path)
    
    size_mb = Path(train_path).stat().st_size / 1024**2
    print(f"✅ Saved: {train_path}")
    print(f"   Shape: {df_final.shape}")
    print(f"   Size: {size_mb:.2f} MB")

def create_info(original_dist, sample_dist, adasyn_dist, final_dist,
               label_encoder, output_dir):
    """Salva info."""
    class_names = label_encoder.classes_
    
    info = {
        'method': 'ADASYN',
        'description': 'Adaptive Synthetic Sampling (density-based generation)',
        'original_distribution': {
            class_names[cls]: int(count) for cls, count in original_dist.items()
        },
        'sample_distribution': {
            class_names[cls]: int(count) for cls, count in sample_dist.items()
        },
        'adasyn_distribution': {
            class_names[cls]: int(count) for cls, count in adasyn_dist.items()
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
         n_neighbors=5, random_state=42):
    
    print_header("🎯 ADASYN DATA AUGMENTATION")
    
    print(f"Configuration:")
    print(f"  Method: ADASYN (Adaptive Synthetic Sampling)")
    print(f"  Target sample: {target_sample:,} rows")
    print(f"  Min minority: {min_minority:,} rows")
    print(f"  Adaptive: More samples in sparse regions")
    
    # Load
    print_header("STEP 1: LOADING DATASET")
    
    train_path = f"{input_dir}/train_processed.pkl"
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Not found: {train_path}")
    
    df_train = pd.read_pickle(train_path)
    print(f"✅ Loaded: {train_path}")
    print(f"   Shape: {df_train.shape}")
    
    feature_cols = [col for col in df_train.columns
                   if col not in ['y_macro_encoded', 'y_specific']]
    
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
    
    # Apply ADASYN
    print_header("STEP 4: ADASYN")
    
    X_adasyn, y_adasyn = apply_adasyn(
        X_sample, y_sample, sampling_strategy, n_neighbors, random_state
    )
    
    del X_sample, y_sample
    gc.collect()
    
    adasyn_dist = analyze_class_distribution(y_adasyn, label_encoder, "ADASYN")
    
    # Extract synthetic
    print_header("STEP 5: EXTRACTING SYNTHETIC")
    
    df_synthetic = extract_synthetic_only(X_adasyn, y_adasyn, X_sample_original, feature_cols)
    
    del X_adasyn, y_adasyn, X_sample_original
    gc.collect()
    
    # Merge
    print_header("STEP 6: MERGING")
    
    df_final = merge_synthetic_with_original(df_train, df_synthetic)
    
    del df_train
    gc.collect()
    
    final_dist = analyze_class_distribution(
        df_final['y_macro_encoded'].values, label_encoder, "FINAL"
    )
    
    # Save
    print_header("STEP 7: SAVING")
    
    save_dataset(df_final, output_dir)
    create_info(original_dist, sample_dist, adasyn_dist, final_dist,
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
    
    n_original = len(y_train)
    n_final = len(df_final)
    n_synthetic = n_final - n_original
    
    print("Summary:")
    print(f"  Original: {n_original:,} rows")
    print(f"  Synthetic: {n_synthetic:,} rows")
    print(f"  Final: {n_final:,} rows")
    print(f"  Increase: +{n_synthetic:,} ({(n_final/n_original-1)*100:.1f}%)")
    print(f"\nOutput: {output_dir}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='ADASYN augmentation')
    parser.add_argument('--input-dir', type=str, default='data/processed/CICIOT23')
    parser.add_argument('--output-dir', type=str, default='data/processed/ADASYN')
    parser.add_argument('--target-sample', type=int, default=DEFAULT_TARGET_SAMPLE)
    parser.add_argument('--min-minority', type=int, default=DEFAULT_MIN_MINORITY)
    parser.add_argument('--sampling-strategy', type=str, default='auto')
    parser.add_argument('--n-neighbors', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"❌ Not found: {args.input_dir}")
        exit(1)
    
    main(args.input_dir, args.output_dir, args.target_sample,
         args.min_minority, args.sampling_strategy, args.n_neighbors, args.seed)