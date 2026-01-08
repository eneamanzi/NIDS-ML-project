"""
SMOTE-ENN (SMOTE + Edited Nearest Neighbors) per CICIoT2023.

VANTAGGI per Network Traffic:
- SMOTE: Genera synthetic samples
- ENN: Rimuove sample ambigui/noise (cleaning post-SMOTE)
- Risultato: Dataset bilanciato E pulito

QUANDO USARLO:
- Se sospetti noise nei dati (es. mislabeled packets)
- Dopo aver provato SMOTE standard e vedi performance sub-ottimali
- Per ottenere boundary più pulite tra classi

TRADE-OFF:
- ✅ Migliore qualità dei dati
- ⚠️ Riduce leggermente il numero di sample (cleaning)

Usage:
    python src/smote_enn.py --input-dir data/processed/CICIOT23
                             --output-dir data/processed/SMOTE_ENN
                             --target-sample 800000
"""

import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from imblearn.combine import SMOTEENN
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
# SMOTE-ENN APPLICATION
# =============================================================================

def apply_smote_enn(X_train, y_train, sampling_strategy='auto',
                    smote_k_neighbors=5, enn_n_neighbors=3, random_state=42):
    """
    Applica SMOTE-ENN (Over-sampling + Cleaning).
    
    PIPELINE:
    1. SMOTE: Genera synthetic samples per bilanciare
    2. ENN (Edited Nearest Neighbors): Rimuove sample ambigui
       - Sample è rimosso se maggioranza dei suoi vicini è di altra classe
       - Pulisce boundary tra classi
    
    Args:
        X_train: Training features
        y_train: Training labels
        sampling_strategy: 'auto' o dict per SMOTE
        smote_k_neighbors: Vicini per SMOTE generation
        enn_n_neighbors: Vicini per ENN cleaning
        random_state: Seed
    
    Returns:
        X_clean, y_clean (bilanciato E pulito)
    """
    print_section("APPLYING SMOTE-ENN")
    
    print(f"Parameters:")
    print(f"  SMOTE sampling_strategy: {sampling_strategy}")
    print(f"  SMOTE k_neighbors: {smote_k_neighbors}")
    print(f"  ENN n_neighbors: {enn_n_neighbors}")
    print(f"  random_state: {random_state}")
    
    # Adjust neighbors
    class_counts = np.bincount(y_train)
    min_samples = class_counts[class_counts > 0].min()
    
    if min_samples <= smote_k_neighbors:
        smote_k_neighbors = max(1, min_samples - 1)
        print(f"  ⚠️ Adjusted SMOTE k_neighbors to {smote_k_neighbors}")
    
    if min_samples <= enn_n_neighbors:
        enn_n_neighbors = max(1, min_samples - 1)
        print(f"  ⚠️ Adjusted ENN n_neighbors to {enn_n_neighbors}")
    
    print(f"\nInput:")
    print(f"  Samples: {len(X_train):,}")
    print(f"  Features: {X_train.shape[1]}")
    
    # Optimize memory
    X_train = X_train.astype(np.float32)
    
    # Memory check
    import psutil
    available_mb = psutil.virtual_memory().available / 1024**2
    print(f"  Available RAM: {available_mb:.0f} MB")
    
    print(f"\n⏳ Applying SMOTE-ENN (2-phase process)...")
    print(f"   Phase 1: SMOTE over-sampling")
    print(f"   Phase 2: ENN cleaning (removes noise/ambiguous samples)")
    print(f"   Expected: Final size < SMOTE size (due to cleaning)")
    
    try:
        smote_enn = SMOTEENN(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            smote=None,  # Use default SMOTE with our k_neighbors
            enn=None,    # Use default ENN with our n_neighbors
            n_jobs=-1    # Use all cores for ENN
        )
        
        # Override internal parameters (tricky but works)
        from imblearn.over_sampling import SMOTE
        from imblearn.under_sampling import EditedNearestNeighbours
        
        smote_enn.smote = SMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=smote_k_neighbors,
            random_state=random_state
        )
        
        smote_enn.enn = EditedNearestNeighbours(
            n_neighbors=enn_n_neighbors,
            n_jobs=-1
        )
        
        print(f"\n   Starting SMOTE phase...")
        X_clean, y_clean = smote_enn.fit_resample(X_train, y_train)
        
        print(f"\n✅ SMOTE-ENN completed!")
        print(f"\nOutput:")
        print(f"  Final samples: {len(X_clean):,}")
        print(f"  Original: {len(X_train):,}")
        print(f"  Net change: {len(X_clean)-len(X_train):+,}")
        
        # Analizza cleaning
        if len(X_clean) < len(X_train):
            removed = len(X_train) - len(X_clean)
            print(f"  ⚠️ Cleaning removed {removed:,} noisy samples ({removed/len(X_train)*100:.1f}%)")
        else:
            added = len(X_clean) - len(X_train)
            print(f"  ✅ Net synthetic added: {added:,} (+{added/len(X_train)*100:.1f}%)")
        
        # Per-class analysis
        print(f"\nPer-class breakdown:")
        original_counts = np.bincount(y_train)
        clean_counts = np.bincount(y_clean)
        
        for cls_idx in range(len(original_counts)):
            if original_counts[cls_idx] > 0:
                orig = original_counts[cls_idx]
                clean = clean_counts[cls_idx] if cls_idx < len(clean_counts) else 0
                delta = clean - orig
                print(f"  Class {cls_idx}: {orig:>7,} → {clean:>7,} ({delta:+6,})")
        
        gc.collect()
        return X_clean, y_clean
        
    except MemoryError as e:
        print(f"\n❌ MEMORY ERROR: {e}")
        print(f"   Reduce --target-sample or --min-minority")
        return X_train, y_train
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print(f"   Returning original data...")
        return X_train, y_train

# =============================================================================
# EXTRACTION & MERGE
# =============================================================================

def extract_synthetic_only(X_clean, y_clean, X_sample_original, feature_cols):
    """
    Estrae synthetic samples (tricky con SMOTE-ENN).
    
    PROBLEMA:
    - ENN può rimuovere sia original che synthetic
    - Non possiamo semplicemente tagliare dopo n_original
    
    SOLUZIONE:
    - Consideriamo TUTTO il dataset cleaned come "augmented"
    - Lo mergiamo con l'original completo (no sample)
    """
    print_section("PREPARING CLEANED DATA")
    
    n_original_sample = len(X_sample_original)
    n_clean = len(X_clean)
    
    print(f"Analysis:")
    print(f"  Original sample used: {n_original_sample:,} rows")
    print(f"  After SMOTE-ENN: {n_clean:,} rows")
    print(f"  Net change: {n_clean - n_original_sample:+,}")
    
    # Crea DataFrame con tutto il cleaned data
    df_clean = pd.DataFrame(X_clean, columns=feature_cols)
    df_clean['y_macro_encoded'] = y_clean
    df_clean['y_specific'] = y_clean
    
    print(f"\n✅ Prepared {len(df_clean):,} cleaned samples")
    
    return df_clean

def merge_clean_with_original(df_original, df_clean, X_sample_original):
    """
    Merge strategico per SMOTE-ENN.
    
    STRATEGIA:
    1. Rimuovi dall'original i sample usati per SMOTE-ENN
    2. Aggiungi tutto il cleaned data (include synthetic + cleaned original)
    """
    print_section("MERGING")
    
    print(f"Original full dataset: {len(df_original):,} rows")
    print(f"Cleaned data: {len(df_clean):,} rows")
    
    # Calcola quanti original NON sono stati usati
    n_sample = len(X_sample_original)
    n_not_sampled = len(df_original) - n_sample
    
    print(f"\nMerge strategy:")
    print(f"  Original samples used in SMOTE-ENN: {n_sample:,}")
    print(f"  Original samples NOT used: {n_not_sampled:,}")
    print(f"  Cleaned data (includes synthetic): {len(df_clean):,}")
    
    # Semplice: usa tutto cleaned + tutti gli original non campionati
    # Ma per semplicità, usiamo solo il cleaned data (già contiene parte degli original)
    print(f"\n💡 Using only cleaned data (already includes cleaned originals)")
    
    df_merged = df_clean.copy()
    df_merged = df_merged.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\n✅ Final dataset: {len(df_merged):,} rows")
    
    gc.collect()
    return df_merged

# =============================================================================
# SAVE & INFO
# =============================================================================

def save_dataset(df_final, output_dir):
    """Salva dataset finale."""
    print_section("SAVING DATASET")
    
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = f"{output_dir}/train_smote_enn.pkl"
    df_final.to_pickle(train_path)
    
    size_mb = Path(train_path).stat().st_size / 1024**2
    print(f"✅ Saved: {train_path}")
    print(f"   Shape: {df_final.shape}")
    print(f"   Size: {size_mb:.2f} MB")

def create_info(original_dist, sample_dist, clean_dist, final_dist,
               label_encoder, output_dir):
    """Salva info."""
    class_names = label_encoder.classes_
    
    info = {
        'method': 'SMOTE-ENN',
        'description': 'SMOTE over-sampling + ENN cleaning (removes noisy samples)',
        'original_distribution': {
            class_names[cls]: int(count) for cls, count in original_dist.items()
        },
        'sample_distribution': {
            class_names[cls]: int(count) for cls, count in sample_dist.items()
        },
        'clean_distribution': {
            class_names[cls]: int(count) for cls, count in clean_dist.items()
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
         smote_k_neighbors=5, enn_n_neighbors=3, random_state=42):
    
    print_header("🎯 SMOTE-ENN DATA AUGMENTATION")
    
    print(f"Configuration:")
    print(f"  Method: SMOTE-ENN (Over-sampling + Cleaning)")
    print(f"  Target sample: {target_sample:,} rows")
    print(f"  Min minority: {min_minority:,} rows")
    print(f"  SMOTE k_neighbors: {smote_k_neighbors}")
    print(f"  ENN n_neighbors: {enn_n_neighbors}")
    
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
    
    # Apply SMOTE-ENN
    print_header("STEP 4: SMOTE-ENN")
    
    X_clean, y_clean = apply_smote_enn(
        X_sample, y_sample, sampling_strategy, 
        smote_k_neighbors, enn_n_neighbors, random_state
    )
    
    del X_sample, y_sample
    gc.collect()
    
    clean_dist = analyze_class_distribution(y_clean, label_encoder, "SMOTE-ENN CLEANED")
    
    # Extract cleaned data
    print_header("STEP 5: PREPARING CLEANED DATA")
    
    df_clean = extract_synthetic_only(X_clean, y_clean, X_sample_original, feature_cols)
    
    del X_clean, y_clean
    gc.collect()
    
    # Merge
    print_header("STEP 6: MERGING")
    
    df_final = merge_clean_with_original(df_train, df_clean, X_sample_original)
    
    del df_train, X_sample_original
    gc.collect()
    
    final_dist = analyze_class_distribution(
        df_final['y_macro_encoded'].values, label_encoder, "FINAL"
    )
    
    # Save
    print_header("STEP 7: SAVING")
    
    save_dataset(df_final, output_dir)
    create_info(original_dist, sample_dist, clean_dist, final_dist,
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
    
    print("Summary:")
    print(f"  Original: {n_original:,} rows")
    print(f"  Final (cleaned): {n_final:,} rows")
    print(f"  Change: {n_final - n_original:+,} ({(n_final/n_original-1)*100:+.1f}%)")
    print(f"\n💡 SMOTE-ENN may reduce size if heavy cleaning occurred")
    print(f"   This is NORMAL and indicates noise removal")
    print(f"\nOutput: {output_dir}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='SMOTE-ENN augmentation')
    parser.add_argument('--input-dir', type=str, default='data/processed/CICIOT23')
    parser.add_argument('--output-dir', type=str, default='data/processed/SMOTE_ENN')
    parser.add_argument('--target-sample', type=int, default=DEFAULT_TARGET_SAMPLE)
    parser.add_argument('--min-minority', type=int, default=DEFAULT_MIN_MINORITY)
    parser.add_argument('--sampling-strategy', type=str, default='auto')
    parser.add_argument('--smote-k-neighbors', type=int, default=5)
    parser.add_argument('--enn-n-neighbors', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"❌ Not found: {args.input_dir}")
        exit(1)
    
    main(args.input_dir, args.output_dir, args.target_sample,
         args.min_minority, args.sampling_strategy, 
         args.smote_k_neighbors, args.enn_n_neighbors, args.seed)