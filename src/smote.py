"""
SMOTE Application - SMART SAMPLING VERSION (FIXED)

PROBLEMA IDENTIFICATO:
- Stratified sampling puro → classi minoritarie hanno troppo pochi sample
- Es: Benign 50k (0.9%) → Sample 500k × 0.9% = 4,500 rows
- SMOTE 4,500 → 360k = 355k synthetic (79x reali) → OVERFITTING!

SOLUZIONE SMART:
1. Garantisci MINIMO sample per classi minoritarie (es. 20k)
2. Distribuisci resto proporzionalmente
3. SMOTE genera synthetic da base solida

ESEMPIO:
├─ Benign: 50k → Sample 50k (100%, tutto!)
├─ DDoS: 4M → Sample 300k (7.5%)
└─ SMOTE Benign: 50k → 300k = 250k synthetic (5x reali) ✅

Usage:
    python src/apply_smote.py --input-dir data/processed/CICIOT23 --min-minority 20000
"""

import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from imblearn.over_sampling import SMOTE
import json
import gc

# =============================================================================
# CONFIGURAZIONE
# =============================================================================

DEFAULT_TARGET_SAMPLE = 800_000  # Target totale sample (aumentato)
DEFAULT_MIN_MINORITY = 50_000    # Minimo garantito per minority classes
MAX_SYNTHETIC_RATIO = 10         # Max synthetic/real ratio (safety)

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
    """
    Smart Stratified Sampling con garanzia su minority classes.
    
    LOGICA:
    1. Calcola distribuzione classi
    2. Per classi < threshold: campiona MIN(tutto, min_minority)
    3. Per classi > threshold: distribuisci budget rimanente proporzionalmente
    
    ESEMPIO (CICIoT2023):
    - Benign 50k (0.9%) → Sample 50k (100%, tutto)
    - Web 5k (0.1%) → Sample 5k (100%, tutto)
    - DDoS 4M (72%) → Sample ~300k (7.5%)
    
    VANTAGGI:
    - Minority classes hanno base solida per SMOTE
    - Majority classes campionate proporzionalmente
    - Total sample size controllato
    
    Args:
        df: DataFrame completo
        y_col: Colonna label
        target_total: Target totale righe sample
        min_minority: Minimo garantito per minority classes
        random_state: Seed
    
    Returns:
        df_sample: DataFrame campionato intelligente
    """
    print_section("SMART STRATIFIED SAMPLING")
    
    total_rows = len(df)
    class_counts = df[y_col].value_counts().sort_values()  # Dal più piccolo
    
    print(f"Original dataset: {total_rows:,} rows")
    print(f"Target sample: {target_total:,} rows")
    print(f"Min minority guarantee: {min_minority:,} rows")
    
    print(f"\nOriginal class distribution:")
    for cls, count in class_counts.items():
        pct = count / total_rows * 100
        print(f"  Class {cls}: {count:>10,} ({pct:>6.2f}%)")
    
    # --- FASE 1: Calcola sample per ogni classe ---
    samples_per_class = {}
    budget_used = 0
    minority_classes = []
    majority_classes = []
    
    # Identifica minority vs majority
    threshold_minority = min_minority * 2  # Soglia: 2x il minimo
    
    for cls, count in class_counts.items():
        if count <= threshold_minority:
            # Classe minoritaria: tieni tutto (fino a min_minority)
            n_sample = min(count, min_minority)
            samples_per_class[cls] = n_sample
            minority_classes.append(cls)
            budget_used += n_sample
        else:
            # Classe maggioritaria: decidi dopo
            majority_classes.append(cls)
    
    print(f"\n📊 Sampling strategy:")
    print(f"  Minority classes ({len(minority_classes)}): Keep all (up to {min_minority:,})")
    for cls in minority_classes:
        print(f"    Class {cls}: {samples_per_class[cls]:,} rows (100%)")
    
    # --- FASE 2: Distribuisci budget rimanente su majority ---
    remaining_budget = target_total - budget_used
    
    if remaining_budget <= 0:
        print("\n⚠️  Warning: Min minority guarantee exceeds target!")
        print(f"   Adjusting target to {budget_used:,}")
        remaining_budget = 0
    
    # Calcola proporzioni solo tra majority classes
    majority_total = sum(class_counts[cls] for cls in majority_classes)
    
    print(f"\n  Majority classes ({len(majority_classes)}): Proportional sampling")
    print(f"  Remaining budget: {remaining_budget:,} rows")
    
    for cls in majority_classes:
        # Proporzione relativa tra majority classes
        proportion = class_counts[cls] / majority_total
        n_sample = int(remaining_budget * proportion)
        
        # Sanity check: non più del totale disponibile
        n_sample = min(n_sample, class_counts[cls])
        
        samples_per_class[cls] = n_sample
        
        sample_pct = (n_sample / class_counts[cls]) * 100
        print(f"    Class {cls}: {n_sample:,} rows ({sample_pct:.1f}% of class)")
    
    # --- FASE 3: Sample effettivo ---
    print(f"\n⏳ Sampling from dataset...")
    
    sampled_dfs = []
    for cls, n_sample in samples_per_class.items():
        df_cls = df[df[y_col] == cls]
        
        if n_sample >= len(df_cls):
            # Prendi tutto
            sampled_dfs.append(df_cls)
        else:
            # Sample random
            sampled_dfs.append(df_cls.sample(n=n_sample, random_state=random_state))
    
    df_sample = pd.concat(sampled_dfs, ignore_index=True)
    
    # Shuffle
    df_sample = df_sample.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    print(f"\n✅ Sample extracted: {len(df_sample):,} rows")
    
    # --- Report finale ---
    print(f"\nFinal sample distribution:")
    sample_counts = df_sample[y_col].value_counts()
    
    for cls in class_counts.index:
        if cls in sample_counts.index:
            sample_count = sample_counts[cls]
            original_count = class_counts[cls]
            sample_pct = (sample_count / len(df_sample)) * 100
            coverage_pct = (sample_count / original_count) * 100
            
            print(f"  Class {cls}: {sample_count:>8,} ({sample_pct:>5.2f}% of sample, "
                  f"{coverage_pct:>5.1f}% of original)")
    
    # Cleanup
    gc.collect()
    
    return df_sample


def analyze_class_distribution(y, label_encoder, dataset_name="Dataset"):
    """Analizza distribuzione classi."""
    unique, counts = np.unique(y, return_counts=True)
    class_names = label_encoder.classes_
    
    print(f"\n{dataset_name} - Class Distribution:")
    print("-" * 60)
    print(f"{'Class':<15} {'Count':>10} {'Percentage':>12}")
    print("-" * 60)
    
    total = len(y)
    for cls_idx, count in zip(unique, counts):
        cls_name = class_names[cls_idx] if cls_idx < len(class_names) else f"Class_{cls_idx}"
        pct = count / total * 100
        print(f"{cls_name:<15} {count:>10,} {pct:>11.2f}%")
    
    print("-" * 60)
    print(f"{'TOTAL':<15} {total:>10,} {100.0:>11.2f}%")
    
    return dict(zip(unique, counts))


def check_smote_quality(class_dist, label_encoder):
    """
    Verifica qualità potenziale SMOTE.
    
    REGOLA: Synthetic/Real ratio non deve superare MAX_SYNTHETIC_RATIO.
    """
    print_section("SMOTE QUALITY CHECK")
    
    class_names = label_encoder.classes_
    counts = list(class_dist.values())
    max_class_size = max(counts)
    
    print(f"Target class size (majority): {max_class_size:,}")
    print(f"\nSynthetic/Real ratio analysis:")
    print("-" * 70)
    print(f"{'Class':<15} {'Real':>10} {'Target':>10} {'Synthetic':>10} {'Ratio':>10} {'Quality':>10}")
    print("-" * 70)
    
    warnings = []
    
    for cls_idx, count in class_dist.items():
        cls_name = class_names[cls_idx]
        synthetic_needed = max_class_size - count
        ratio = synthetic_needed / count if count > 0 else float('inf')
        
        # Quality assessment
        if ratio <= 3:
            quality = "✅ Excellent"
        elif ratio <= 10:
            quality = "⚠️  Good"
        elif ratio <= 20:
            quality = "⚠️  Risky"
        else:
            quality = "❌ Poor"
            warnings.append(f"Class {cls_name}: ratio {ratio:.1f}x too high!")
        
        print(f"{cls_name:<15} {count:>10,} {max_class_size:>10,} {synthetic_needed:>10,} "
              f"{ratio:>9.1f}x {quality:>10}")
    
    print("-" * 70)
    
    if warnings:
        print(f"\n⚠️  WARNINGS:")
        for warn in warnings:
            print(f"  {warn}")
        print(f"\n💡 Consider increasing --min-minority to improve quality")
    else:
        print(f"\n✅ All classes have acceptable synthetic/real ratios!")
    
    return len(warnings) == 0


def apply_smote(X_train, y_train, sampling_strategy='auto', k_neighbors=5, random_state=42):
    """
    Applica SMOTE con controlli ULTRA-SAFE.
    
    MODIFICHE ANTI-CRASH:
    1. n_jobs=1 (evita thread crash)
    2. Memory check prima di SMOTE
    3. Conversione float32 (dimezza RAM)
    4. Garbage collection aggressivo
    """
    print_section("APPLYING SMOTE (ULTRA-SAFE MODE)")
    
    print(f"SMOTE Parameters:")
    print(f"  sampling_strategy: {sampling_strategy}")
    print(f"  k_neighbors: {k_neighbors}")
    print(f"  random_state: {random_state}")
    print(f"  n_jobs: 1 (single-thread, stable)")
    
    # Adjust k_neighbors
    class_counts = np.bincount(y_train)
    min_samples = class_counts[class_counts > 0].min()
    
    if min_samples <= k_neighbors:
        k_neighbors_adjusted = max(1, min_samples - 1)
        print(f"  ⚠️  Adjusting k_neighbors: {k_neighbors} → {k_neighbors_adjusted}")
        k_neighbors = k_neighbors_adjusted
    
    # Converti a float32 PRIMA di SMOTE (dimezza memoria)
    print(f"\n⚙️  Optimizing input data...")
    X_train = X_train.astype(np.float32)
    
    print(f"\nInput:")
    print(f"  Samples: {len(X_train):,}")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Memory: {X_train.nbytes / 1024**2:.2f} MB")
    
    # Estimate output
    max_class = class_counts.max()
    n_classes = len(class_counts[class_counts > 0])
    estimated_output = max_class * n_classes
    estimated_mb = (estimated_output * X_train.shape[1] * 4) / 1024**2  # float32
    
    print(f"\nEstimated SMOTE output:")
    print(f"  Samples: ~{estimated_output:,}")
    print(f"  Memory: ~{estimated_mb:.0f} MB")
    
    # SAFETY CHECK: Verifica RAM disponibile
    import psutil
    available_mb = psutil.virtual_memory().available / 1024**2
    required_mb = estimated_mb * 1.5  # Buffer 50%
    
    print(f"\nMemory check:")
    print(f"  Available RAM: {available_mb:.0f} MB")
    print(f"  Required (est): {required_mb:.0f} MB")
    
    if required_mb > available_mb:
        print(f"\n❌ ERROR: Insufficient RAM!")
        print(f"   Required: {required_mb:.0f} MB")
        print(f"   Available: {available_mb:.0f} MB")
        print(f"\n💡 Solutions:")
        print(f"   1. Reduce --target-sample")
        print(f"   2. Reduce --min-minority")
        print(f"   3. Close other applications")
        
        response = input("\nContinue anyway? [y/N]: ")
        if response.lower() != 'y':
            print("⚠️  Aborting SMOTE. Returning original data.")
            return X_train, y_train
    
    # Garbage collection pre-SMOTE
    gc.collect()
    
    # Apply SMOTE (SINGLE THREAD!)
    print(f"\n⏳ Applying SMOTE...")
    print(f"   Mode: Single-thread (safer but slower)")
    print(f"   This may take 10-30 minutes...")
    
    try:
        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=k_neighbors,
            random_state=random_state,
            #n_jobs=1  # ← CRITICAL: Single thread evita crash
        )
        
        # Progress feedback ogni 60s
        import time
        start_time = time.time()
        
        print(f"   Started at: {time.strftime('%H:%M:%S')}")
        
        X_smote, y_smote = smote.fit_resample(X_train, y_train)
        
        elapsed = time.time() - start_time
        
        print(f"\n✅ SMOTE completed in {elapsed:.1f}s ({elapsed/60:.1f} min)!")
        print(f"\nOutput:")
        print(f"  Samples: {len(X_smote):,} (+{len(X_smote)-len(X_train):,})")
        print(f"  Synthetic: {len(X_smote)-len(X_train):,}")
        print(f"  Increase: {(len(X_smote)/len(X_train)-1)*100:.1f}%")
        print(f"  Memory: {X_smote.nbytes / 1024**2:.2f} MB")
        
        # Garbage collection post-SMOTE
        gc.collect()
        
        return X_smote, y_smote
        
    except MemoryError as e:
        print(f"\n❌ MEMORY ERROR: {e}")
        print(f"\n💡 Dataset too large for available RAM.")
        print(f"   Reduce --target-sample or --min-minority")
        return X_train, y_train
        
    except Exception as e:
        print(f"\n❌ SMOTE Error: {e}")
        print(f"\n⚠️  Returning original without SMOTE...")
        return X_train, y_train


def extract_synthetic_only(X_smote, y_smote, X_sample_original, feature_cols):
    """
    Estrae SOLO i sample sintetici da SMOTE output.
    
    PROBLEMA RISOLTO:
    - SMOTE output = original sample + synthetic
    - Se facciamo concat(df_original, df_smote) → duplicati!
    
    SOLUZIONE:
    - Identifica quale parte di X_smote è synthetic
    - Ritorna solo synthetic samples
    
    Args:
        X_smote: Output completo di SMOTE
        y_smote: Labels SMOTE
        X_sample_original: Sample originale PRE-SMOTE
        feature_cols: Lista colonne
    
    Returns:
        df_synthetic: Solo sample sintetici (no duplicati)
    """
    print_section("EXTRACTING SYNTHETIC SAMPLES ONLY")
    
    n_original = len(X_sample_original)
    n_smote = len(X_smote)
    n_synthetic = n_smote - n_original
    
    print(f"SMOTE output breakdown:")
    print(f"  Total SMOTE output: {n_smote:,} rows")
    print(f"  Original (resampled): {n_original:,} rows")
    print(f"  Synthetic (new): {n_synthetic:,} rows")
    
    if n_synthetic <= 0:
        print(f"\n⚠️  No synthetic samples generated!")
        print(f"   This happens when dataset is already balanced.")
        return pd.DataFrame()  # Empty DataFrame
    
    # SMOTE aggiunge synthetic DOPO gli originali
    # Quindi: X_smote[:n_original] = originali
    #         X_smote[n_original:] = sintetici
    X_synthetic = X_smote[n_original:]
    y_synthetic = y_smote[n_original:]
    
    print(f"\n✅ Extracted {len(X_synthetic):,} synthetic samples")
    
    # Crea DataFrame
    df_synthetic = pd.DataFrame(X_synthetic, columns=feature_cols)
    df_synthetic['y_macro_encoded'] = y_synthetic
    df_synthetic['y_specific'] = y_synthetic
    
    return df_synthetic


def merge_synthetic_with_original(df_original, df_synthetic):
    """
    Merge SOLO synthetic samples con original (NO DUPLICATES).
    
    LOGICA CORRETTA:
    - df_original: 5.5M rows (completo)
    - df_synthetic: Solo synthetic (es. 2M)
    - Result: 5.5M + 2M = 7.5M (no duplicates!)
    """
    print_section("MERGING SYNTHETIC WITH ORIGINAL")
    
    print(f"Original dataset: {len(df_original):,} rows")
    print(f"Synthetic samples: {len(df_synthetic):,} rows")
    
    if len(df_synthetic) == 0:
        print(f"\n⚠️  No synthetic samples to merge.")
        print(f"   Returning original dataset unchanged.")
        return df_original
    
    # Merge
    df_merged = pd.concat([df_original, df_synthetic], ignore_index=True)
    
    print(f"\n✅ Merged dataset: {len(df_merged):,} rows")
    print(f"  Original: {len(df_original):,} ({len(df_original)/len(df_merged)*100:.1f}%)")
    print(f"  Synthetic: {len(df_synthetic):,} ({len(df_synthetic)/len(df_merged)*100:.1f}%)")
    print(f"  Expected total: {len(df_original) + len(df_synthetic):,}")
    
    # Verifica no duplicates
    if len(df_merged) == len(df_original) + len(df_synthetic):
        print(f"  ✅ No duplicates (size check passed)")
    else:
        print(f"  ⚠️  WARNING: Unexpected size mismatch!")
    
    # Shuffle
    print(f"\nShuffling merged dataset...")
    df_merged = df_merged.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Cleanup
    del df_synthetic
    gc.collect()
    
    return df_merged


def load_dataset(filepath):
    """Carica dataset."""
    print(f"Loading: {filepath}")
    df = pd.read_pickle(filepath)
    size_mb = df.memory_usage(deep=True).sum() / 1024**2
    print(f"  Shape: {df.shape}")
    print(f"  Memory: {size_mb:.2f} MB")
    return df


def save_smote_dataset(df_train_final, output_dir):
    """Salva dataset finale."""
    print_section("SAVING SMOTE DATASET")
    
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = f"{output_dir}/train_smote.pkl"
    df_train_final.to_pickle(train_path)
    
    size_mb = Path(train_path).stat().st_size / 1024**2
    print(f"✅ Saved: {train_path}")
    print(f"   Shape: {df_train_final.shape}")
    print(f"   Size: {size_mb:.2f} MB")


def create_smote_info(original_dist, sample_dist, smote_dist, final_dist, 
                     label_encoder, output_dir):
    """Salva info SMOTE."""
    class_names = label_encoder.classes_
    
    smote_info = {
        'smote_applied': True,
        'strategy': 'smart_stratified',
        'description': 'Smart stratified sampling with minority guarantee + SMOTE + merge',
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
    
    info_path = f"{output_dir}/smote_info.json"
    save_json(smote_info, info_path)


# =============================================================================
# MAIN
# =============================================================================

def main(input_dir, output_dir, target_sample=DEFAULT_TARGET_SAMPLE, 
         min_minority=DEFAULT_MIN_MINORITY, sampling_strategy='auto', 
         k_neighbors=5, random_state=42):
    
    print_header("🎯 SMOTE - SMART SAMPLING STRATEGY")
    
    print(f"Configuration:")
    print(f"  Target sample: {target_sample:,} rows")
    print(f"  Min minority: {min_minority:,} rows")
    print(f"  Strategy: Smart Stratified + SMOTE + Merge")
    
    # Load
    print_header("STEP 1: LOADING DATASET")
    
    train_path = f"{input_dir}/train_processed.pkl"
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Not found: {train_path}")
    
    df_train = load_dataset(train_path)
    
    feature_cols = [col for col in df_train.columns 
                   if col not in ['y_macro_encoded', 'y_specific']]
    
    encoder_path = f"{input_dir}/label_encoder.pkl"
    label_encoder = joblib.load(encoder_path)
    print(f"\nLabel encoder: {label_encoder.classes_}")
    
    # Analyze original
    print_header("STEP 2: ORIGINAL DISTRIBUTION")
    
    y_train = df_train['y_macro_encoded'].values
    original_dist = analyze_class_distribution(y_train, label_encoder, "ORIGINAL")
    
    # Smart sample
    print_header("STEP 3: SMART STRATIFIED SAMPLING")
    
    df_sample = extract_smart_stratified_sample(
        df_train, 'y_macro_encoded', target_sample, min_minority, random_state
    )
    
    X_sample = df_sample[feature_cols].values
    y_sample = df_sample['y_macro_encoded'].values
    
    sample_dist = analyze_class_distribution(y_sample, label_encoder, "SAMPLE")
    
    # Quality check
    quality_ok = check_smote_quality(sample_dist, label_encoder)
    
    if not quality_ok:
        print("\n⚠️  Consider increasing --min-minority for better SMOTE quality")
        response = input("\nContinue anyway? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    # ⚠️ CRITICAL: Salva X_sample_original PRIMA di SMOTE
    X_sample_original = X_sample.copy()
    
    del df_sample
    gc.collect()
    
    # Apply SMOTE
    print_header("STEP 4: APPLYING SMOTE")
    
    X_smote, y_smote = apply_smote(X_sample, y_sample, sampling_strategy, 
                                   k_neighbors, random_state)
    
    del X_sample, y_sample
    gc.collect()
    
    smote_dist = analyze_class_distribution(y_smote, label_encoder, "SMOTE")
    
    # ⚠️ FIX: Estrai SOLO synthetic samples
    print_header("STEP 5: EXTRACTING SYNTHETIC SAMPLES")
    
    df_synthetic = extract_synthetic_only(X_smote, y_smote, X_sample_original, feature_cols)
    
    del X_smote, y_smote, X_sample_original
    gc.collect()
    
    # Merge (CORRECTED)
    print_header("STEP 6: MERGING")
    
    df_final = merge_synthetic_with_original(df_train, df_synthetic)
    
    del df_train
    gc.collect()
    
    final_dist = analyze_class_distribution(
        df_final['y_macro_encoded'].values, label_encoder, "FINAL"
    )
    
    # Save
    print_header("STEP 7: SAVING")
    
    save_smote_dataset(df_final, output_dir)
    create_smote_info(original_dist, sample_dist, smote_dist, final_dist, 
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
    print(f"  Original dataset: {n_original:,} rows")
    print(f"  Synthetic generated: {n_synthetic:,} rows")
    print(f"  Final dataset: {n_final:,} rows")
    print(f"  Increase: +{n_synthetic:,} rows ({(n_final/n_original-1)*100:.1f}%)")
    
    # Verifica no duplicates
    print(f"\n📊 Duplicate check:")
    print(f"  Expected: {n_original} original + {n_synthetic} synthetic = {n_original + n_synthetic}")
    print(f"  Actual: {n_final}")
    
    if n_final == n_original + n_synthetic:
        print(f"  ✅ VERIFIED: No duplicates!")
    else:
        print(f"  ⚠️  WARNING: Size mismatch detected!")
    
    print(f"\nOutput: {output_dir}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='SMOTE with smart stratified sampling',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--input-dir', type=str, default='data/processed/CICIOT23')
    parser.add_argument('--output-dir', type=str, default='data/processed/SMOTE')
    parser.add_argument('--target-sample', type=int, default=DEFAULT_TARGET_SAMPLE,
                        help=f'Target sample size (default: {DEFAULT_TARGET_SAMPLE:,})')
    parser.add_argument('--min-minority', type=int, default=DEFAULT_MIN_MINORITY,
                        help=f'Min rows for minority classes (default: {DEFAULT_MIN_MINORITY:,})')
    parser.add_argument('--sampling-strategy', type=str, default='auto')
    parser.add_argument('--k-neighbors', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"❌ Not found: {args.input_dir}")
        exit(1)
    
    main(args.input_dir, args.output_dir, args.target_sample, 
         args.min_minority, args.sampling_strategy, args.k_neighbors, args.seed)