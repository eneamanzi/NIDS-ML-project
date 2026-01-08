"""
CTGAN (Conditional Tabular GAN) per CICIoT2023.

PERCHÉ CTGAN (non TVAE):
- CTGAN: GAN-based, genera sample più diversi e realistici
- TVAE: VAE-based, più conservativo ma meno diversità
- Per network traffic: CTGAN è migliore (cattura meglio variabilità)

VANTAGGI:
- Genera sample MOLTO realistici (deep learning)
- Cattura distribuzioni complesse multi-modali
- Ottimo per feature continue (flow_duration, IAT, etc.)

SVANTAGGI:
- LENTO (training richiede 30-60 min su subset)
- Richiede GPU (fortemente consigliata)
- Stabile ma può divergere se mal configurato

STRATEGIA:
1. Train su SUBSET (100k samples, non tutto!)
2. Genera synthetic per minority classes
3. Merge con full dataset

Usage:
    # CPU (slow, ~60 min)
    python src/ctgan.py --input-dir data/processed/CICIOT23
                        --subset-size 100000 --epochs 100
    
    # GPU (fast, ~10 min)
    python src/ctgan.py --input-dir data/processed/CICIOT23
                        --subset-size 200000 --epochs 300 --use-gpu
"""

import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
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

DEFAULT_SUBSET_SIZE = 100_000  # Subset for CTGAN training
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 500

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
# CTGAN IMPORT & SETUP
# =============================================================================

def check_ctgan_installed():
    """Verifica che CTGAN sia installato."""
    try:
        from ctgan import CTGAN
        print("✅ CTGAN library found")
        return True
    except ImportError:
        print("❌ CTGAN not installed!")
        print("\nInstall with:")
        print("  pip install ctgan")
        print("\nOr:")
        print("  pip install sdv")
        return False

def check_gpu_available():
    """Verifica disponibilità GPU."""
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
# SUBSET EXTRACTION
# =============================================================================

def extract_stratified_subset(df, y_col, subset_size, random_state=42):
    """
    Estrae subset stratificato per CTGAN training.
    
    IMPORTANTE: CTGAN va trainato su SUBSET per efficienza.
    """
    print_section("EXTRACTING STRATIFIED SUBSET FOR CTGAN")
    
    print(f"Target subset size: {subset_size:,} rows")
    
    class_counts = df[y_col].value_counts()
    total_rows = len(df)
    
    # Proportional sampling
    sampled_dfs = []
    actual_samples = 0
    
    for cls in class_counts.index:
        df_cls = df[df[y_col] == cls]
        n_cls = len(df_cls)
        
        # Proporzione
        proportion = n_cls / total_rows
        n_sample = int(subset_size * proportion)
        
        # Non più del disponibile
        n_sample = min(n_sample, n_cls)
        
        # Almeno 100 per classe (se possibile)
        n_sample = max(n_sample, min(100, n_cls))
        
        sampled = df_cls.sample(n=n_sample, random_state=random_state)
        sampled_dfs.append(sampled)
        actual_samples += len(sampled)
        
        print(f"  Class {cls}: {n_cls:>8,} → {n_sample:>7,} ({n_sample/n_cls*100:>5.1f}%)")
    
    df_subset = pd.concat(sampled_dfs, ignore_index=True)
    df_subset = df_subset.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    print(f"\n✅ Subset extracted: {len(df_subset):,} rows")
    
    gc.collect()
    return df_subset

# =============================================================================
# CTGAN TRAINING
# =============================================================================

def train_ctgan(df_train, feature_cols, label_col, epochs=DEFAULT_EPOCHS,
                batch_size=DEFAULT_BATCH_SIZE, use_gpu=False, random_state=42):
    """
    Train CTGAN su subset.
    
    Args:
        df_train: DataFrame training subset
        feature_cols: Lista feature columns
        label_col: Colonna label (condizionale)
        epochs: Numero epoch (100-300)
        batch_size: Batch size (500-1000)
        use_gpu: Use CUDA se disponibile
        random_state: Seed
    
    Returns:
        Trained CTGAN model
    """
    print_section("TRAINING CTGAN")
    
    from ctgan import CTGAN
    import torch
    
    print(f"Configuration:")
    print(f"  Training samples: {len(df_train):,}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Device: {'GPU (CUDA)' if use_gpu and torch.cuda.is_available() else 'CPU'}")
    
    # Prepara discrete columns (solo label)
    discrete_columns = [label_col]
    
    # Select data
    train_data = df_train[feature_cols + [label_col]]
    
    print(f"\n⏳ Training CTGAN...")
    print(f"   This will take time:")
    print(f"   - CPU: ~{epochs * len(df_train) / 1000:.0f} seconds")
    print(f"   - GPU: ~{epochs * len(df_train) / 5000:.0f} seconds")
    print(f"\n   Progress will be shown every 10 epochs...")
    
    # Initialize CTGAN
    ctgan = CTGAN(
        epochs=epochs,
        batch_size=batch_size,
        verbose=True,
        cuda=use_gpu and torch.cuda.is_available()
    )
    
    # Train
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
        print(f"\n💡 Tips:")
        print(f"   - Reduce --epochs or --subset-size")
        print(f"   - Check GPU memory if using GPU")
        print(f"   - Increase --batch-size if RAM available")
        raise

# =============================================================================
# SYNTHETIC GENERATION
# =============================================================================

def generate_synthetic_samples(ctgan, label_encoder, target_per_class=None,
                               feature_cols=None, random_state=42):
    """
    Genera synthetic samples con CTGAN.
    
    Args:
        ctgan: Trained CTGAN model
        label_encoder: Label encoder per class names
        target_per_class: Dict {class_idx: n_samples} o None (auto)
        feature_cols: Lista feature columns
        random_state: Seed
    
    Returns:
        DataFrame con synthetic samples
    """
    print_section("GENERATING SYNTHETIC SAMPLES")
    
    if target_per_class is None:
        # Default: genera 50k per ogni classe
        n_classes = len(label_encoder.classes_)
        target_per_class = {i: 50000 for i in range(n_classes)}
    
    print(f"Target generation per class:")
    for cls_idx, n_samples in target_per_class.items():
        cls_name = label_encoder.classes_[cls_idx]
        print(f"  {cls_name:<15}: {n_samples:>7,} samples")
    
    total_to_generate = sum(target_per_class.values())
    print(f"\nTotal synthetic to generate: {total_to_generate:,}")
    
    # Generate per class (condizionale)
    synthetic_dfs = []
    
    for cls_idx, n_samples in target_per_class.items():
        if n_samples <= 0:
            continue
        
        cls_name = label_encoder.classes_[cls_idx]
        print(f"\n  Generating {n_samples:,} samples for {cls_name}...", end=" ")
        
        try:
            # Conditional sampling
            conditions = pd.DataFrame({
                'y_macro_encoded': [cls_idx] * n_samples
            })
            
            synthetic = ctgan.sample(
                n=n_samples,
                conditions=conditions
            )
            
            synthetic_dfs.append(synthetic)
            print(f"✅")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    if not synthetic_dfs:
        raise RuntimeError("No synthetic samples generated!")
    
    df_synthetic = pd.concat(synthetic_dfs, ignore_index=True)
    df_synthetic = df_synthetic.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Add y_specific (same as y_macro for synthetic)
    df_synthetic['y_specific'] = df_synthetic['y_macro_encoded']
    
    print(f"\n✅ Generated {len(df_synthetic):,} synthetic samples")
    
    gc.collect()
    return df_synthetic

# =============================================================================
# MERGE & SAVE
# =============================================================================

def merge_synthetic_with_original(df_original, df_synthetic):
    """Merge synthetic con original."""
    print_section("MERGING")
    
    print(f"Original: {len(df_original):,} rows")
    print(f"Synthetic: {len(df_synthetic):,} rows")
    
    # Ensure same columns order
    df_synthetic = df_synthetic[df_original.columns]
    
    df_merged = pd.concat([df_original, df_synthetic], ignore_index=True)
    df_merged = df_merged.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\n✅ Merged: {len(df_merged):,} rows")
    
    gc.collect()
    return df_merged

def save_dataset(df_final, output_dir):
    """Salva dataset finale."""
    print_section("SAVING DATASET")
    
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = f"{output_dir}/train_ctgan.pkl"
    df_final.to_pickle(train_path)
    
    size_mb = Path(train_path).stat().st_size / 1024**2
    print(f"✅ Saved: {train_path}")
    print(f"   Shape: {df_final.shape}")
    print(f"   Size: {size_mb:.2f} MB")

def save_ctgan_model(ctgan, output_dir):
    """Salva modello CTGAN (optional, per riutilizzo)."""
    print(f"\nSaving CTGAN model...")
    
    model_path = f"{output_dir}/ctgan_model.pkl"
    
    try:
        import pickle
        with open(model_path, 'wb') as f:
            pickle.dump(ctgan, f)
        
        size_mb = Path(model_path).stat().st_size / 1024**2
        print(f"✅ CTGAN model saved: {model_path}")
        print(f"   Size: {size_mb:.2f} MB")
        print(f"   Can be reused for future generation")
    except Exception as e:
        print(f"⚠️ Could not save CTGAN model: {e}")

def create_info(original_dist, subset_dist, final_dist, label_encoder,
               output_dir, ctgan_config):
    """Salva info."""
    class_names = label_encoder.classes_
    
    info = {
        'method': 'CTGAN',
        'description': 'Conditional Tabular GAN (deep learning-based generation)',
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

def main(input_dir, output_dir, subset_size=DEFAULT_SUBSET_SIZE,
         epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH_SIZE,
         use_gpu=False, save_model=False, random_state=42):
    
    print_header("🤖 CTGAN DATA AUGMENTATION")
    
    # Check dependencies
    if not check_ctgan_installed():
        return
    
    gpu_available = check_gpu_available()
    
    print(f"\nConfiguration:")
    print(f"  Method: CTGAN (Conditional Tabular GAN)")
    print(f"  Training subset: {subset_size:,} rows")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Device: {'GPU' if use_gpu and gpu_available else 'CPU'}")
    
    # Load
    print_header("STEP 1: LOADING DATASET")
    
    train_path = f"{input_dir}/train_processed.pkl"
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Not found: {train_path}")
    
    df_train = pd.read_pickle(train_path)
    print(f"✅ Loaded: {train_path}")
    print(f"   Full dataset: {df_train.shape}")
    
    feature_cols = [col for col in df_train.columns
                   if col not in ['y_macro_encoded', 'y_specific']]
    
    encoder_path = f"{input_dir}/label_encoder.pkl"
    label_encoder = joblib.load(encoder_path)
    print(f"✅ Label encoder: {label_encoder.classes_}")
    
    # Analyze original
    print_header("STEP 2: ORIGINAL DISTRIBUTION")
    y_train = df_train['y_macro_encoded'].values
    original_dist = analyze_class_distribution(y_train, label_encoder, "ORIGINAL")
    
    # Extract subset
    print_header("STEP 3: EXTRACTING TRAINING SUBSET")
    
    df_subset = extract_stratified_subset(
        df_train, 'y_macro_encoded', subset_size, random_state
    )
    
    y_subset = df_subset['y_macro_encoded'].values
    subset_dist = analyze_class_distribution(y_subset, label_encoder, "TRAINING SUBSET")
    
    # Train CTGAN
    print_header("STEP 4: TRAINING CTGAN")
    
    ctgan = train_ctgan(
        df_subset, feature_cols, 'y_macro_encoded',
        epochs, batch_size, use_gpu, random_state
    )
    
    del df_subset
    gc.collect()
    
    # Generate synthetic
    print_header("STEP 5: GENERATING SYNTHETIC SAMPLES")
    
    # Calculate target: minority classes get more synthetic
    class_counts = dict(zip(*np.unique(y_train, return_counts=True)))
    max_count = max(class_counts.values())
    
    target_per_class = {}
    for cls_idx, count in class_counts.items():
        # Generate to reach 80% of max class
        target = int(max_count * 0.8) - count
        target = max(0, target)  # Non negativo
        target = min(target, 100000)  # Cap a 100k per classe
        target_per_class[cls_idx] = target
    
    print(f"\nGeneration strategy (balance to 80% of majority):")
    for cls_idx, target in target_per_class.items():
        cls_name = label_encoder.classes_[cls_idx]
        current = class_counts[cls_idx]
        print(f"  {cls_name:<15}: {current:>8,} + {target:>7,} synthetic")
    
    df_synthetic = generate_synthetic_samples(
        ctgan, label_encoder, target_per_class, feature_cols, random_state
    )
    
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
    
    if save_model:
        save_ctgan_model(ctgan, output_dir)
    
    ctgan_config = {
        'subset_size': subset_size,
        'epochs': epochs,
        'batch_size': batch_size,
        'device': 'GPU' if use_gpu and gpu_available else 'CPU'
    }
    
    create_info(original_dist, subset_dist, final_dist,
               label_encoder, output_dir, ctgan_config)
    
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
    print(f"  Synthetic (CTGAN): {n_synthetic:,} rows")
    print(f"  Final: {n_final:,} rows")
    print(f"  Increase: +{n_synthetic:,} ({(n_final/n_original-1)*100:.1f}%)")
    print(f"\nOutput: {output_dir}")
    
    print("\n💡 CTGAN generates HIGH-QUALITY synthetic data")
    print("   Compare with other methods (Borderline-SMOTE, ADASYN) to see difference")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='CTGAN augmentation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
IMPORTANT:
  CTGAN requires: pip install ctgan
  GPU strongly recommended (10x faster)
  
  Subset size: 100k = ~10 min CPU, 200k = ~20 min CPU
  With GPU: 5x faster
        """
    )
    
    parser.add_argument('--input-dir', type=str, default='data/processed/CICIOT23')
    parser.add_argument('--output-dir', type=str, default='data/processed/CTGAN')
    parser.add_argument('--subset-size', type=int, default=DEFAULT_SUBSET_SIZE,
                       help='Size of subset for CTGAN training (default: 100k)')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS,
                       help='Training epochs (default: 100, try 300 for better quality)')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
                       help='Batch size (default: 500)')
    parser.add_argument('--use-gpu', action='store_true',
                       help='Use GPU if available (highly recommended)')
    parser.add_argument('--save-model', action='store_true',
                       help='Save trained CTGAN model for reuse')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"❌ Not found: {args.input_dir}")
        exit(1)
    
    main(args.input_dir, args.output_dir, args.subset_size,
         args.epochs, args.batch_size, args.use_gpu, args.save_model, args.seed)