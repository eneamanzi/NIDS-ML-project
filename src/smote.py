# """
# Applicazione SMOTE (Synthetic Minority Over-sampling) al training set.

# IMPORTANTE:
# - SMOTE applicato SOLO al training set
# - Test e Validation restano con distribuzione originale
# - Questo evita data leakage e mantiene valutazione realistica

# Usage:
#     python src/apply_smote.py --input-dir data/processed/original \\
#                                --output-dir data/processed/smote
# """

# import pandas as pd
# import numpy as np
# import joblib
# import os
# from pathlib import Path
# from imblearn.over_sampling import SMOTE
# import json


# def print_header(text):
#     """Stampa header formattato."""
#     print("\n" + "="*80)
#     print(text.center(80))
#     print("="*80 + "\n")


# def print_section(text):
#     """Stampa section formattato."""
#     print("\n" + "-"*80)
#     print(text)
#     print("-"*80)


# def load_dataset(filepath):
#     """Carica dataset in formato PKL."""
#     df = pd.read_pickle(filepath)
#     print(f"Loaded: {filepath}")
#     print(f"  Shape: {df.shape}")
#     return df


# def analyze_class_distribution(y, label_encoder, dataset_name="Dataset"):
#     """Analizza e stampa distribuzione delle classi."""
#     unique, counts = np.unique(y, return_counts=True)
#     class_names = label_encoder.classes_
    
#     print(f"\n{dataset_name} - Class Distribution:")
#     print("-" * 60)
#     print(f"{'Class':<15} {'Count':>10} {'Percentage':>12}")
#     print("-" * 60)
    
#     total = len(y)
#     for cls_idx, count in zip(unique, counts):
#         cls_name = class_names[cls_idx] if cls_idx < len(class_names) else f"Class_{cls_idx}"
#         pct = count / total * 100
#         print(f"{cls_name:<15} {count:>10,} {pct:>11.2f}%")
    
#     print("-" * 60)
#     print(f"{'TOTAL':<15} {total:>10,} {100.0:>11.2f}%")
    
#     return dict(zip(unique, counts))


# def apply_smote(X_train, y_train, sampling_strategy='auto', k_neighbors=5, random_state=42):
#     """
#     Applica SMOTE per bilanciare il training set.
    
#     Args:
#         X_train: Features training
#         y_train: Labels training (macro-categories encoded)
#         sampling_strategy: Strategia di sampling
#             - 'auto': Bilancia tutte le classi alla maggioritaria
#             - dict: Specifica numero campioni per classe {class: n_samples}
#             - float: Ratio desiderato per minority class
#         k_neighbors: Numero vicini per interpolazione SMOTE
#         random_state: Seed per riproducibilità
    
#     Returns:
#         X_train_smote, y_train_smote
#     """
#     print_section("APPLYING SMOTE")
    
#     print(f"SMOTE Parameters:")
#     print(f"  sampling_strategy: {sampling_strategy}")
#     print(f"  k_neighbors: {k_neighbors}")
#     print(f"  random_state: {random_state}")
    
#     # Verifica che ci siano abbastanza campioni per k_neighbors
#     min_samples = np.min(np.bincount(y_train))
#     if min_samples <= k_neighbors:
#         k_neighbors = max(1, min_samples - 1)
#         print(f"  ⚠️  Adjusting k_neighbors to {k_neighbors} (min class has {min_samples} samples)")
    
#     print("\nOriginal training set:")
#     print(f"  Samples: {len(X_train):,}")
#     print(f"  Features: {X_train.shape[1]}")
#     print(f"  Classes: {len(np.unique(y_train))}")
    
#     # Applica SMOTE
#     print("\nApplying SMOTE (this may take a few minutes)...")
#     smote = SMOTE(
#         sampling_strategy=sampling_strategy,
#         k_neighbors=k_neighbors,
#         random_state=random_state
#         #n_jobs=-1
#     )
    
#     X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
#     print("\n✅ SMOTE completed!")
#     print(f"\nSMOTE-balanced training set:")
#     print(f"  Samples: {len(X_train_smote):,} (増加: +{len(X_train_smote) - len(X_train):,})")
#     print(f"  Features: {X_train_smote.shape[1]}")
#     print(f"  Synthetic samples generated: {len(X_train_smote) - len(X_train):,}")
#     print(f"  Increase: {(len(X_train_smote) / len(X_train) - 1) * 100:.1f}%")
    
#     return X_train_smote, y_train_smote


# def save_smote_dataset(X_train_smote, y_train_smote, y_specific_train, output_dir, feature_cols):
#     """
#     Salva dataset con SMOTE.
    
#     IMPORTANTE: Solo train è modificato, test e val restano originali!
#     """
#     print_section("SAVING SMOTE DATASET")
    
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Train (con SMOTE)
#     # Nota: y_specific_train non può essere perfettamente mappato ai nuovi sample sintetici
#     # Per semplicità, replichiamo y_specific per matching con y_macro
#     # In produzione, y_specific sarà derivato dalla predizione
#     train_smote = pd.DataFrame(X_train_smote, columns=feature_cols)
#     train_smote['y_macro_encoded'] = y_train_smote
    
#     # Per y_specific: usiamo un mapping approssimativo
#     # I sample sintetici avranno y_specific basato sulla classe macro
#     train_smote['y_specific'] = y_train_smote  # Semplificazione: usa macro come proxy
    
#     train_path = f"{output_dir}/train_processed.pkl"
#     train_smote.to_pickle(train_path)
#     size_mb = os.path.getsize(train_path) / 1024**2
#     print(f"✅ Train (SMOTE): {train_path}")
#     print(f"   Shape: {train_smote.shape} | Size: {size_mb:.2f} MB")
    
#     print("\n⚠️ IMPORTANTE:")
#     print("   - Train set: SMOTE applied (synthetic samples)")
#     print("\nQuesto garantisce valutazione realistica su dati non sintetici!")


# def create_smote_info(original_dist, smote_dist, label_encoder, output_dir):
#     """Salva informazioni su SMOTE per riferimento."""
#     class_names = label_encoder.classes_
    
#     smote_info = {
#         'smote_applied': True,
#         'smote_strategy': 'auto',
#         'original_distribution': {
#             class_names[cls]: int(count) 
#             for cls, count in original_dist.items()
#         },
#         'smote_distribution': {
#             class_names[cls]: int(count) 
#             for cls, count in smote_dist.items()
#         },
#         'samples_generated': {
#             class_names[cls]: int(smote_dist.get(cls, 0) - original_dist.get(cls, 0))
#             for cls in original_dist.keys()
#         }
#     }
    
#     info_path = f"{output_dir}/smote_info.json"
#     with open(info_path, 'w') as f:
#         json.dump(smote_info, f, indent=2)
    
#     print(f"\n✅ SMOTE info saved: {info_path}")


# def main(input_dir, output_dir, sampling_strategy='auto', k_neighbors=5, random_state=42):
#     """
#     Main pipeline per applicare SMOTE.
#     """
#     print_header("🎯 SMOTE APPLICATION - CICIoT2023")
    
#     print("This script applies SMOTE to balance the training set.")
#     print("Test and Validation sets remain UNCHANGED (original distribution).")
    
#     # =========================================================================
#     # STEP 1: LOAD ORIGINAL DATASETS
#     # =========================================================================
#     print_header("STEP 1: LOADING ORIGINAL DATASETS")
    
#     train_path = f"{input_dir}/train_processed.pkl"
    
#     df_train = load_dataset(train_path)
    
#     # Identifica colonne
#     feature_cols = [col for col in df_train.columns if col not in ['y_macro_encoded', 'y_specific']]
    
#     # Separa features e labels
#     X_train = df_train[feature_cols].values
#     y_train = df_train['y_macro_encoded'].values
#     y_specific_train = df_train['y_specific'].values
    
#     print(f"\nFeatures: {len(feature_cols)}")
    
#     # Carica label encoder
#     encoder_path = f"{input_dir}/label_encoder.pkl"
#     label_encoder = joblib.load(encoder_path)
#     print(f"Label encoder loaded: {encoder_path}")
#     print(f"  Classes: {label_encoder.classes_}")
    
#     # =========================================================================
#     # STEP 2: ANALYZE ORIGINAL DISTRIBUTION
#     # =========================================================================
#     print_header("STEP 2: ORIGINAL CLASS DISTRIBUTION")
    
#     original_dist = analyze_class_distribution(y_train, label_encoder, "ORIGINAL Training Set")
    
#     # Calcola imbalance ratio
#     min_class = min(original_dist.values())
#     max_class = max(original_dist.values())
#     imbalance_ratio = max_class / min_class
    
#     print(f"\n📊 Imbalance Analysis:")
#     print(f"   Min class samples: {min_class:,}")
#     print(f"   Max class samples: {max_class:,}")
#     print(f"   Imbalance ratio: {imbalance_ratio:.2f}x")
    
#     if imbalance_ratio < 2:
#         print("\n✅ Dataset is relatively balanced (ratio < 2x)")
#         print("   SMOTE may not be necessary, but can still help minority classes.")
#     else:
#         print(f"\n⚠️ Dataset is imbalanced (ratio {imbalance_ratio:.1f}x)")
#         print("   SMOTE will significantly help balance the classes.")
    
#     # =========================================================================
#     # STEP 3: APPLY SMOTE
#     # =========================================================================
#     print_header("STEP 3: APPLYING SMOTE")
    
#     X_train_smote, y_train_smote = apply_smote(
#         X_train, y_train,
#         sampling_strategy=sampling_strategy,
#         k_neighbors=k_neighbors,
#         random_state=random_state
#     )
    
#     # =========================================================================
#     # STEP 4: ANALYZE SMOTE DISTRIBUTION
#     # =========================================================================
#     print_header("STEP 4: SMOTE-BALANCED DISTRIBUTION")
    
#     smote_dist = analyze_class_distribution(y_train_smote, label_encoder, "SMOTE Training Set")
    
#     # Confronto
#     print("\n📊 Before vs After SMOTE:")
#     print("-" * 70)
#     print(f"{'Class':<15} {'Original':>12} {'SMOTE':>12} {'Increase':>12}")
#     print("-" * 70)
    
#     class_names = label_encoder.classes_
#     for cls in sorted(original_dist.keys()):
#         cls_name = class_names[cls]
#         orig_count = original_dist[cls]
#         smote_count = smote_dist[cls]
#         increase = smote_count - orig_count
#         increase_pct = (increase / orig_count * 100) if orig_count > 0 else 0
        
#         print(f"{cls_name:<15} {orig_count:>12,} {smote_count:>12,} +{increase:>10,} ({increase_pct:>5.1f}%)")
    
#     print("-" * 70)
    
#     # =========================================================================
#     # STEP 5: SAVE SMOTE DATASET
#     # =========================================================================
#     print_header("STEP 5: SAVING DATASET")
    
#     save_smote_dataset(
#         X_train_smote, y_train_smote, y_specific_train,
#         output_dir, feature_cols
#     )
    
#     # Salva info SMOTE
#     create_smote_info(original_dist, smote_dist, label_encoder, output_dir)
    
#     # =========================================================================
#     # SUMMARY
#     # =========================================================================
#     print_header("✅ SMOTE APPLICATION COMPLETE!")
    
#     print("Summary:")
#     print(f"  Original train samples: {len(X_train):,}")
#     print(f"  SMOTE train samples: {len(X_train_smote):,}")
#     print(f"  Increase: +{len(X_train_smote) - len(X_train):,} samples ({(len(X_train_smote)/len(X_train)-1)*100:.1f}%)")

#     print(f"\nOutput directory: {output_dir}/")
#     print("\n💡 Next steps:")
#     print("   1. Train model on SMOTE dataset:")
#     print(f"      python src/train_random_forest_multiclass.py --data-dir {output_dir}")
#     print("   2. Compare with original dataset results")
#     print("   3. Evaluate which performs better on test set")


# if __name__ == '__main__':
#     import argparse
    
#     parser = argparse.ArgumentParser(
#         description='Apply SMOTE to training set for class balancing'
#     )
#     parser.add_argument('--input-dir', type=str, default='../data/processed/CICIOT23',
#                         help='Input directory with original processed data')
#     parser.add_argument('--output-dir', type=str, default='../data/processed/SMOTE',
#                         help='Output directory for SMOTE dataset')
#     parser.add_argument('--sampling-strategy', type=str, default='auto',
#                         help='SMOTE sampling strategy (auto, minority, all)')
#     parser.add_argument('--k-neighbors', type=int, default=5,
#                         help='Number of neighbors for SMOTE interpolation')
#     parser.add_argument('--seed', type=int, default=42,
#                         help='Random seed for reproducibility')
    
#     args = parser.parse_args()
    
#     # Verifica che input directory esista
#     if not os.path.exists(args.input_dir):
#         print(f"❌ Error: Input directory not found: {args.input_dir}")
#         print("\n💡 First run preprocessing:")
#         print("   python src/data_processing.py --train-path ... --output-dir data/processed/original")
#         exit(1)
    
#     # Run SMOTE
#     main(
#         input_dir=args.input_dir,
#         output_dir=args.output_dir,
#         sampling_strategy=args.sampling_strategy,
#         k_neighbors=args.k_neighbors,
#         random_state=args.seed
#     )


"""
Applicazione SMOTE (Synthetic Minority Over-sampling) al training set.

IMPORTANTE:
- SMOTE applicato SOLO al training set
- Test e Validation restano con distribuzione originale
- Questo evita data leakage e mantiene valutazione realistica

Usage:
    python src/apply_smote.py --input-dir data/processed/original \\
                               --output-dir data/processed/smote
"""

import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from imblearn.over_sampling import SMOTE
import json


def print_header(text):
    """Stampa header formattato."""
    print("\n" + "="*80)
    print(text.center(80))
    print("="*80 + "\n")


def print_section(text):
    """Stampa section formattato."""
    print("\n" + "-"*80)
    print(text)
    print("-"*80)


def load_dataset(filepath):
    """Carica dataset in formato PKL."""
    df = pd.read_pickle(filepath)
    print(f"Loaded: {filepath}")
    print(f"  Shape: {df.shape}")
    return df


def analyze_class_distribution(y, label_encoder, dataset_name="Dataset"):
    """Analizza e stampa distribuzione delle classi."""
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


def apply_smote(X_train, y_train, sampling_strategy='auto', k_neighbors=5, random_state=42):
    """
    Applica SMOTE per bilanciare il training set.
    
    Args:
        X_train: Features training
        y_train: Labels training (macro-categories encoded)
        sampling_strategy: Strategia di sampling
            - 'auto': Bilancia tutte le classi alla maggioritaria
            - dict: Specifica numero campioni per classe {class: n_samples}
            - float: Ratio desiderato per minority class
        k_neighbors: Numero vicini per interpolazione SMOTE
        random_state: Seed per riproducibilità
    
    Returns:
        X_train_smote, y_train_smote
    """
    print_section("APPLYING SMOTE")
    
    print(f"SMOTE Parameters:")
    print(f"  sampling_strategy: {sampling_strategy}")
    print(f"  k_neighbors: {k_neighbors}")
    print(f"  random_state: {random_state}")
    
    # Verifica che ci siano abbastanza campioni per k_neighbors
    min_samples = np.min(np.bincount(y_train))
    
    # LOGICA CORRETTA:
    # SMOTE richiede k_neighbors < numero campioni nella classe più piccola
    # Quindi k_neighbors_effective deve essere STRETTAMENTE minore di min_samples
    if min_samples <= k_neighbors:
        # Calcola il massimo k_neighbors possibile
        max_possible_k = min_samples - 1
        
        # Usa almeno 1, ma mai più del massimo possibile
        k_neighbors_adjusted = max(1, max_possible_k)
        
        print(f"  ⚠️  Adjusting k_neighbors from {k_neighbors} to {k_neighbors_adjusted}")
        print(f"      (smallest class has {min_samples} samples)")
        print(f"      SMOTE requires k_neighbors < min_class_samples")
        
        k_neighbors = k_neighbors_adjusted
    
    print("\nOriginal training set:")
    print(f"  Samples: {len(X_train):,}")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Classes: {len(np.unique(y_train))}")
    
    # Applica SMOTE
    print("\nApplying SMOTE (this may take a few minutes)...")
    
    try:
        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=k_neighbors,
            random_state=random_state,
        )
        
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        
        print("\n✅ SMOTE completed!")
        print(f"\nSMOTE-balanced training set:")
        print(f"  Samples: {len(X_train_smote):,} (+{len(X_train_smote) - len(X_train):,})")
        print(f"  Features: {X_train_smote.shape[1]}")
        print(f"  Synthetic samples generated: {len(X_train_smote) - len(X_train):,}")
        print(f"  Increase: {(len(X_train_smote) / len(X_train) - 1) * 100:.1f}%")
        
        return X_train_smote, y_train_smote
        
    except ValueError as e:
        print(f"\n❌ SMOTE Error: {e}")
        print("\n🔍 Debugging Info:")
        print(f"   k_neighbors used: {k_neighbors}")
        print(f"   min_samples in smallest class: {min_samples}")
        print(f"   Class distribution:")
        for cls, count in zip(*np.unique(y_train, return_counts=True)):
            print(f"      Class {cls}: {count} samples")
        
        # Fallback: ritorna il dataset originale
        print("\n⚠️  Returning original dataset without SMOTE...")
        return X_train, y_train


def save_smote_dataset(X_train_smote, y_train_smote, y_specific_train, 
                       output_dir, feature_cols):
    """
    Salva dataset con SMOTE.
    
    IMPORTANTE: Solo train è modificato, test e val restano originali!
    """
    print_section("SAVING SMOTE DATASET")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Train (con SMOTE)
    train_smote = pd.DataFrame(X_train_smote, columns=feature_cols)
    train_smote['y_macro_encoded'] = y_train_smote
    
    # Per y_specific: replica basandosi sulla classe macro
    # Nota: i sample sintetici avranno y_specific approssimativo
    train_smote['y_specific'] = y_train_smote
    
    train_path = f"{output_dir}/train_smote.pkl"
    train_smote.to_pickle(train_path)
    size_mb = os.path.getsize(train_path) / 1024**2
    print(f"✅ Train (SMOTE): {train_path}")
    print(f"   Shape: {train_smote.shape} | Size: {size_mb:.2f} MB")
    
    print("\n⚠️ IMPORTANTE:")
    print("   - Train set: SMOTE applied (synthetic samples)")
    print("\nQuesto garantisce valutazione realistica su dati non sintetici!")


def create_smote_info(original_dist, smote_dist, label_encoder, output_dir):
    """Salva informazioni su SMOTE per riferimento."""
    class_names = label_encoder.classes_
    
    smote_info = {
        'smote_applied': True,
        'smote_strategy': 'auto',
        'original_distribution': {
            class_names[cls]: int(count) 
            for cls, count in original_dist.items()
        },
        'smote_distribution': {
            class_names[cls]: int(count) 
            for cls, count in smote_dist.items()
        },
        'samples_generated': {
            class_names[cls]: int(smote_dist.get(cls, 0) - original_dist.get(cls, 0))
            for cls in original_dist.keys()
        }
    }
    
    info_path = f"{output_dir}/smote_info.json"
    with open(info_path, 'w') as f:
        json.dump(smote_info, f, indent=2)
    
    print(f"\n✅ SMOTE info saved: {info_path}")


def main(input_dir, output_dir, sampling_strategy='auto', k_neighbors=5, random_state=42):
    """
    Main pipeline per applicare SMOTE.
    """
    print_header("🎯 SMOTE APPLICATION - CICIoT2023")
    
    print("This script applies SMOTE to balance the training set.")
    print("Test and Validation sets remain UNCHANGED (original distribution).")
    
    # =========================================================================
    # STEP 1: LOAD ORIGINAL DATASETS
    # =========================================================================
    print_header("STEP 1: LOADING ORIGINAL DATASETS")
    
    train_path = f"{input_dir}/train_processed.pkl"
    
    df_train = load_dataset(train_path)
    
    # Identifica colonne
    feature_cols = [col for col in df_train.columns if col not in ['y_macro_encoded', 'y_specific']]
    
    # Separa features e labels
    X_train = df_train[feature_cols].values
    y_train = df_train['y_macro_encoded'].values
    y_specific_train = df_train['y_specific'].values
    
    print(f"\nFeatures: {len(feature_cols)}")
    
    # Carica label encoder
    encoder_path = f"{input_dir}/label_encoder.pkl"
    label_encoder = joblib.load(encoder_path)
    print(f"Label encoder loaded: {encoder_path}")
    print(f"  Classes: {label_encoder.classes_}")
    
    # =========================================================================
    # STEP 2: ANALYZE ORIGINAL DISTRIBUTION
    # =========================================================================
    print_header("STEP 2: ORIGINAL CLASS DISTRIBUTION")
    
    original_dist = analyze_class_distribution(y_train, label_encoder, "ORIGINAL Training Set")
    
    # Calcola imbalance ratio
    min_class = min(original_dist.values())
    max_class = max(original_dist.values())
    imbalance_ratio = max_class / min_class
    
    print(f"\n📊 Imbalance Analysis:")
    print(f"   Min class samples: {min_class:,}")
    print(f"   Max class samples: {max_class:,}")
    print(f"   Imbalance ratio: {imbalance_ratio:.2f}x")
    
    if imbalance_ratio < 2:
        print("\n✅ Dataset is relatively balanced (ratio < 2x)")
        print("   SMOTE may not be necessary, but can still help minority classes.")
    else:
        print(f"\n⚠️ Dataset is imbalanced (ratio {imbalance_ratio:.1f}x)")
        print("   SMOTE will significantly help balance the classes.")
    
    # =========================================================================
    # STEP 3: APPLY SMOTE
    # =========================================================================
    print_header("STEP 3: APPLYING SMOTE")
    
    X_train_smote, y_train_smote = apply_smote(
        X_train, y_train,
        sampling_strategy=sampling_strategy,
        k_neighbors=k_neighbors,
        random_state=random_state
    )
    
    # =========================================================================
    # STEP 4: ANALYZE SMOTE DISTRIBUTION
    # =========================================================================
    print_header("STEP 4: SMOTE-BALANCED DISTRIBUTION")
    
    smote_dist = analyze_class_distribution(y_train_smote, label_encoder, "SMOTE Training Set")
    
    # Confronto
    print("\n📊 Before vs After SMOTE:")
    print("-" * 70)
    print(f"{'Class':<15} {'Original':>12} {'SMOTE':>12} {'Increase':>12}")
    print("-" * 70)
    
    class_names = label_encoder.classes_
    for cls in sorted(original_dist.keys()):
        cls_name = class_names[cls]
        orig_count = original_dist[cls]
        smote_count = smote_dist[cls]
        increase = smote_count - orig_count
        increase_pct = (increase / orig_count * 100) if orig_count > 0 else 0
        
        print(f"{cls_name:<15} {orig_count:>12,} {smote_count:>12,} +{increase:>10,} ({increase_pct:>5.1f}%)")
    
    print("-" * 70)
    
    # =========================================================================
    # STEP 5: SAVE SMOTE DATASET
    # =========================================================================
    print_header("STEP 5: SAVING DATASET")
    
    save_smote_dataset(
        X_train_smote, y_train_smote, y_specific_train,
        output_dir, feature_cols
    )
    
    # Salva info SMOTE
    create_smote_info(original_dist, smote_dist, label_encoder, output_dir)
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print_header("✅ SMOTE APPLICATION COMPLETE!")
    
    print("Summary:")
    print(f"  Original train samples: {len(X_train):,}")
    print(f"  SMOTE train samples: {len(X_train_smote):,}")
    print(f"  Increase: +{len(X_train_smote) - len(X_train):,} samples ({(len(X_train_smote)/len(X_train)-1)*100:.1f}%)")
    
    print(f"\nOutput directory: {output_dir}/")
    print("\n💡 Next steps:")
    print("   1. Train model on SMOTE dataset:")
    print(f"      python src/train_random_forest_multiclass.py --data-dir {output_dir}")
    print("   2. Compare with original dataset results")
    print("   3. Evaluate which performs better on test set")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Apply SMOTE to training set for class balancing'
    )
    parser.add_argument('--input-dir', type=str, default='data/processed/CICIOT23',
                        help='Input directory with original processed data')
    parser.add_argument('--output-dir', type=str, default='data/processed/SMOTE',
                        help='Output directory for SMOTE dataset')
    parser.add_argument('--sampling-strategy', type=str, default='auto',
                        help='SMOTE sampling strategy (auto, minority, all)')
    parser.add_argument('--k-neighbors', type=int, default=5,
                        help='Number of neighbors for SMOTE interpolation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')  
    
    args = parser.parse_args()
    
    # TODO SISTEMARE LA COSA DIPATH PER ESEGURIE LO SCRIPT
    # Verifica che input directory esista
    if not os.path.exists(args.input_dir):
        print(f"❌ Error: Input directory not found: {args.input_dir}")
        print("\n💡 First run preprocessing:")
        print("   python src/data_processing.py --train-path ... --output-dir data/processed/CICIOT23")
        exit(1)
    
    # Run SMOTE
    main(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        sampling_strategy=args.sampling_strategy,
        k_neighbors=args.k_neighbors,
        random_state=args.seed
    )