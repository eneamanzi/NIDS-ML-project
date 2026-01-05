"""
Advanced Data Processing per CICIoT2023 con Macro-Categorie.

Features:
- Mapping 34 classi → 7-8 macro-categorie
- Doppia label: y_macro (training) + y_specific (logging/dashboard)
- Anti-leakage: fit solo su train, transform su test/val
- Gestione memoria ottimizzata per dataset grandi
- Output: 3 dataset processati + artifacts (scaler, encoder)

Usage:
    python src/data_processing.py --train-path data/raw/train.csv \\
                                   --test-path data/raw/test.csv \\
                                   --val-path data/raw/val.csv
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURAZIONE MAPPING ATTACCHI → MACRO-CATEGORIE
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
    
    # === Web (5 varianti) ===
    'SqlInjection': 'Web',
    'XSS': 'Web',
    'CommandInjection': 'Web',
    'Uploading_Attack': 'Web',
    'BrowserHijacking': 'Web',
    
    # === BruteForce (1 variante) ===
    'DictionaryBruteForce': 'BruteForce',
    
    # === Backdoor (1 variante) ===
    'Backdoor_Malware': 'Backdoor',
    
    # === Benign ===
    'BenignTraffic': 'Benign'
}

# Ordine macro-categorie (per encoding consistente)
MACRO_CATEGORIES = ['Benign', 'DDoS', 'DoS', 'Mirai', 'Recon', 'Web', 'Spoofing', 'BruteForce', 'Backdoor']


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

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


def save_json(data, filepath):
    """Salva dictionary come JSON."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"✅ Saved: {filepath}")


# =============================================================================
# CORE PROCESSING FUNCTIONS
# =============================================================================

def load_dataset(filepath, nrows=None, label_col='label'):
    """
    Carica dataset CSV con gestione memoria ottimizzata.
    
    Args:
        filepath: Path al CSV
        nrows: Limite righe (per testing)
        label_col: Nome colonna label
    
    Returns:
        DataFrame, feature_columns
    """
    print(f"Loading: {filepath}")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Carica dataset
    if nrows:
        df = pd.read_csv(filepath, nrows=nrows)
        print(f"  Sample loaded: {len(df):,} rows (limit: {nrows:,})")
    else:
        df = pd.read_csv(filepath)
        print(f"  Loaded: {len(df):,} rows")
    
    # Identifica colonne feature (tutte tranne label)
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in dataset!")
    
    feature_cols = [col for col in df.columns if col != label_col]
    
    print(f"  Features: {len(feature_cols)}")
    print(f"  Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return df, feature_cols


def clean_data(df):
    """
    Pulizia dati: missing, infiniti, duplicati.
    
    Args:
        df: DataFrame
    
    Returns:
        DataFrame pulito
    """
    print_section("DATA CLEANING")
    
    initial_rows = len(df)
    
    # 1. Missing values
    missing = df.isnull().sum().sum()
    if missing > 0:
        print(f"⚠️  Found {missing:,} missing values → filling with 0")
        df = df.fillna(0)
    else:
        print("✅ No missing values")
    
    # 2. Infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_count = 0
    
    for col in numeric_cols:
        inf_mask = np.isinf(df[col])
        if inf_mask.any():
            inf_count += inf_mask.sum()
            max_val = df[col][~inf_mask].max()
            df.loc[inf_mask, col] = max_val
    
    if inf_count > 0:
        print(f"⚠️  Found {inf_count:,} infinite values → replaced with column max")
    else:
        print("✅ No infinite values")
    
    # 3. Duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"⚠️  Found {duplicates:,} duplicate rows → removing")
        df = df.drop_duplicates()
    else:
        print("✅ No duplicates")
    
    removed = initial_rows - len(df)
    print(f"\nCleaning summary:")
    print(f"  Rows removed: {removed:,} ({removed/initial_rows*100:.2f}%)")
    print(f"  Final rows: {len(df):,}")
    
    return df


def create_macro_labels(df, label_col='label'):
    """
    Crea le due colonne target: y_macro e y_specific.
    
    Args:
        df: DataFrame con label originale
        label_col: Nome colonna label
    
    Returns:
        df con y_macro e y_specific, mapping_info dict
    """
    print_section("LABEL MAPPING: 34 Classes → Macro-Categories")
    
    # Verifica che tutte le label siano nel mapping
    unique_labels = df[label_col].unique()
    unmapped = [lbl for lbl in unique_labels if lbl not in ATTACK_MAPPING]
    
    if unmapped:
        print(f"⚠️  WARNING: {len(unmapped)} unmapped labels found:")
        for lbl in unmapped:
            print(f"    - {lbl}")
        raise ValueError("Please add missing labels to ATTACK_MAPPING!")
    
    # Crea y_specific: indice numerico per ogni attacco specifico
    # (mantiene traccia dell'attacco originale per logging/dashboard)
    specific_to_idx = {lbl: idx for idx, lbl in enumerate(sorted(unique_labels))}
    df['y_specific'] = df[label_col].map(specific_to_idx)
    
    # Crea y_macro: macro-categoria per training
    df['y_macro'] = df[label_col].map(ATTACK_MAPPING)
    
    # Statistiche
    print("\nLabel distribution:")
    print("\nSpecific labels (top 10):")
    print(df[label_col].value_counts().head(10))
    
    print("\nMacro-categories:")
    macro_dist = df['y_macro'].value_counts()
    for cat in MACRO_CATEGORIES:
        if cat in macro_dist.index:
            count = macro_dist[cat]
            print(f"  {cat:15s}: {count:>8,} ({count/len(df)*100:>6.2f}%)")
    
    # Info mapping
    mapping_info = {
        'specific_to_idx': specific_to_idx,
        'idx_to_specific': {v: k for k, v in specific_to_idx.items()},
        'attack_mapping': ATTACK_MAPPING,
        'macro_categories': MACRO_CATEGORIES,
        'n_specific_classes': len(specific_to_idx),
        'n_macro_classes': len(MACRO_CATEGORIES)
    }
    
    return df, mapping_info


def encode_macro_labels(df_train, df_test, df_val, mapping_info):
    """
    Encoding macro-labels con LabelEncoder.
    FIT solo su train, TRANSFORM su test/val (anti-leakage).
    
    Args:
        df_train, df_test, df_val: DataFrame con y_macro
        mapping_info: Dict con info mapping
    
    Returns:
        df_train, df_test, df_val con y_macro encoded, label_encoder
    """
    print_section("ENCODING MACRO-LABELS (Anti-Leakage)")
    
    # Crea encoder
    label_encoder = LabelEncoder()
    
    # FIT solo su train
    print("FIT LabelEncoder on TRAIN set...")
    label_encoder.fit(df_train['y_macro'])
    
    print(f"  Classes found: {label_encoder.classes_}")
    print(f"  Encoding: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
    
    # TRANSFORM su tutti i set
    print("\nTRANSFORM labels:")
    df_train['y_macro_encoded'] = label_encoder.transform(df_train['y_macro'])
    print(f"  ✅ Train: {len(df_train)} rows")
    
    if df_test is not None:
        # Gestisci eventuali label non viste in train
        unseen_test = set(df_test['y_macro']) - set(label_encoder.classes_)
        if unseen_test:
            print(f"  ⚠️  Test has unseen labels: {unseen_test}")
            # Mappa unseen labels a classe più simile o -1
            df_test['y_macro_encoded'] = df_test['y_macro'].apply(
                lambda x: label_encoder.transform([x])[0] if x in label_encoder.classes_ else -1
            )
        else:
            df_test['y_macro_encoded'] = label_encoder.transform(df_test['y_macro'])
        print(f"  ✅ Test: {len(df_test)} rows")
    
    if df_val is not None:
        unseen_val = set(df_val['y_macro']) - set(label_encoder.classes_)
        if unseen_val:
            print(f"  ⚠️  Val has unseen labels: {unseen_val}")
            df_val['y_macro_encoded'] = df_val['y_macro'].apply(
                lambda x: label_encoder.transform([x])[0] if x in label_encoder.classes_ else -1
            )
        else:
            df_val['y_macro_encoded'] = label_encoder.transform(df_val['y_macro'])
        print(f"  ✅ Val: {len(df_val)} rows")
    
    # Aggiungi encoder a mapping_info
    mapping_info['label_encoder_classes'] = label_encoder.classes_.tolist()
    
    return df_train, df_test, df_val, label_encoder


def normalize_features(df_train, df_test, df_val, feature_cols):
    """
    Normalizzazione con StandardScaler.
    FIT solo su train, TRANSFORM su test/val (anti-leakage).
    
    Args:
        df_train, df_test, df_val: DataFrame
        feature_cols: Lista colonne feature
    
    Returns:
        df_train, df_test, df_val con features scalate, scaler
    """
    print_section("FEATURE NORMALIZATION (Anti-Leakage)")
    
    # Crea scaler
    scaler = StandardScaler()
    
    # FIT solo su train
    print("FIT StandardScaler on TRAIN set...")
    X_train = df_train[feature_cols].values
    scaler.fit(X_train)
    
    print(f"  Features: {len(feature_cols)}")
    print(f"  Mean: ~{scaler.mean_.mean():.6f}")
    print(f"  Std: ~{scaler.scale_.mean():.6f}")
    
    # TRANSFORM su tutti i set
    print("\nTRANSFORM features:")
    df_train[feature_cols] = scaler.transform(df_train[feature_cols])
    print(f"  ✅ Train scaled")
    
    if df_test is not None:
        df_test[feature_cols] = scaler.transform(df_test[feature_cols])
        print(f"  ✅ Test scaled")
    
    if df_val is not None:
        df_val[feature_cols] = scaler.transform(df_val[feature_cols])
        print(f"  ✅ Val scaled")
    
    return df_train, df_test, df_val, scaler


def save_processed_datasets(df_train, df_test, df_val, output_dir, feature_cols):
    """
    Salva i dataset processati in formato PICKLE (.pkl) per performance elevate.
    
    Output columns: features + y_macro_encoded + y_specific
    """
    print_section("SAVING PROCESSED DATASETS (PICKLE FORMAT)")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Colonne da salvare
    output_cols = feature_cols + ['y_macro_encoded', 'y_specific']
    
    # Train
    train_path = f"{output_dir}/train_processed.pkl"
    df_train[output_cols].to_pickle(train_path) # <--- CAMBIATO QUI
    print(f"✅ Train: {train_path}")
    print(f"   Shape: {df_train[output_cols].shape}")
    
    # Test
    if df_test is not None:
        test_path = f"{output_dir}/test_processed.pkl"
        df_test[output_cols].to_pickle(test_path) # <--- CAMBIATO QUI
        print(f"✅ Test: {test_path}")
        print(f"   Shape: {df_test[output_cols].shape}")
    
    # Val
    if df_val is not None:
        val_path = f"{output_dir}/validation_processed.pkl"
        df_val[output_cols].to_pickle(val_path) # <--- CAMBIATO QUI
        print(f"✅ Val: {val_path}")
        print(f"   Shape: {df_val[output_cols].shape}")
        
    return train_path, test_path if df_test is not None else None, val_path if df_val is not None else None


def save_artifacts(scaler, label_encoder, mapping_info, output_dir):
    """
    Salva artifacts per uso futuro (produzione, dashboard).
    """
    print_section("SAVING ARTIFACTS")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Scaler
    scaler_path = f"{output_dir}/scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"✅ Scaler: {scaler_path}")
    
    # Label Encoder
    encoder_path = f"{output_dir}/label_encoder.pkl"
    joblib.dump(label_encoder, encoder_path)
    print(f"✅ Label Encoder: {encoder_path}")
    
    # Mapping Info
    mapping_path = f"{output_dir}/mapping_info.json"
    save_json(mapping_info, mapping_path)
    
    print("\nArtifacts ready for production use!")


# =============================================================================
# MAIN PROCESSING PIPELINE
# =============================================================================

def process_pipeline(
    train_path,
    test_path=None,
    val_path=None,
    output_dir='data/processed',
    nrows=None,
    label_col='label'
):
    """
    Pipeline completa di preprocessing con anti-leakage.
    
    Args:
        train_path: Path dataset train
        test_path: Path dataset test (opzionale)
        val_path: Path dataset validation (opzionale)
        output_dir: Directory output
        nrows: Limite righe per testing
        label_col: Nome colonna label
    
    Returns:
        Paths dei file salvati
    """
    print_header("🚀 ADVANCED DATA PROCESSING - CICIoT2023")
    
    # =========================================================================
    # STEP 1: LOAD DATASETS
    # =========================================================================
    print_header("STEP 1: LOADING DATASETS")
    
    df_train, feature_cols = load_dataset(train_path, nrows=nrows, label_col=label_col)
    
    if test_path:
        df_test, _ = load_dataset(test_path, nrows=nrows, label_col=label_col)
    else:
        df_test = None
        print("ℹ️  No test set provided")
    
    if val_path:
        df_val, _ = load_dataset(val_path, nrows=nrows, label_col=label_col)
    else:
        df_val = None
        print("ℹ️  No validation set provided")
    
    # =========================================================================
    # STEP 2: CLEAN DATA
    # =========================================================================
    print_header("STEP 2: DATA CLEANING")
    
    df_train = clean_data(df_train)
    if df_test is not None:
        df_test = clean_data(df_test)
    if df_val is not None:
        df_val = clean_data(df_val)
    
    # =========================================================================
    # STEP 3: CREATE MACRO-LABELS
    # =========================================================================
    print_header("STEP 3: LABEL MAPPING")
    
    df_train, mapping_info = create_macro_labels(df_train, label_col=label_col)
    
    if df_test is not None:
        df_test, _ = create_macro_labels(df_test, label_col=label_col)
    
    if df_val is not None:
        df_val, _ = create_macro_labels(df_val, label_col=label_col)
    
    # =========================================================================
    # STEP 4: ENCODE LABELS (Anti-Leakage)
    # =========================================================================
    print_header("STEP 4: LABEL ENCODING")
    
    df_train, df_test, df_val, label_encoder = encode_macro_labels(
        df_train, df_test, df_val, mapping_info
    )
    
    # =========================================================================
    # STEP 5: NORMALIZE FEATURES (Anti-Leakage)
    # =========================================================================
    print_header("STEP 5: FEATURE NORMALIZATION")
    
    df_train, df_test, df_val, scaler = normalize_features(
        df_train, df_test, df_val, feature_cols
    )
    
    # =========================================================================
    # STEP 6: SAVE OUTPUTS
    # =========================================================================
    print_header("STEP 6: SAVING OUTPUTS")
    
    save_processed_datasets(df_train, df_test, df_val, output_dir, feature_cols)
    save_artifacts(scaler, label_encoder, mapping_info, output_dir)
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print_header("✅ PROCESSING COMPLETE!")
    
    print("Summary:")
    print(f"  Train samples: {len(df_train):,}")
    if df_test is not None:
        print(f"  Test samples: {len(df_test):,}")
    if df_val is not None:
        print(f"  Val samples: {len(df_val):,}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Macro-categories: {len(MACRO_CATEGORIES)}")
    print(f"  Specific classes: {mapping_info['n_specific_classes']}")
    print(f"\nOutput directory: {output_dir}/")
    
    return {
        'train': f"{output_dir}/train_processed.pkl", # .csv -> .pkl
        'test': f"{output_dir}/test_processed.pkl" if df_test is not None else None,
        'val': f"{output_dir}/validation_processed.pkl" if df_val is not None else None,
        'scaler': f"{output_dir}/scaler.pkl",
        'label_encoder': f"{output_dir}/label_encoder.pkl",
        'mapping_info': f"{output_dir}/mapping_info.json"
    }


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Advanced preprocessing for CICIoT2023 with macro-categories'
    )
    parser.add_argument('--train-path', type=str, default='../data/raw/CICIOT23/train/train.csv',
                        help='Path to training CSV')
    parser.add_argument('--test-path', type=str, default='../data/raw/CICIOT23/test/test.csv',
                        help='Path to test CSV (optional)')
    parser.add_argument('--val-path', type=str, default='../data/raw/CICIOT23/validation/validation.csv',
                        help='Path to validation CSV (optional)')
    parser.add_argument('--output-dir', type=str, default='../data/processed/CICIOT23',
                        help='Output directory')
    parser.add_argument('--nrows', type=int, default=None,
                        help='Limit rows for testing (default: load all)')
    parser.add_argument('--label-col', type=str, default='label',
                        help='Name of label column')
    
    args = parser.parse_args()
    
    # Run pipeline
    paths = process_pipeline(
        train_path=args.train_path,
        test_path=args.test_path,
        val_path=args.val_path,
        output_dir=args.output_dir,
        nrows=args.nrows,
        label_col=args.label_col
    )
    
    print("\n" + "="*80)
    print("📁 OUTPUT FILES:")
    print("="*80)
    for key, path in paths.items():
        if path:
            print(f"  {key:15s}: {path}")