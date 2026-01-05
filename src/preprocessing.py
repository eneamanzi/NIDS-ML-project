# """
# Preprocessing pipeline per NSL-KDD dataset.
# """

# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# import joblib

# # Definizione colonne (globale per consistenza)
# COLUMN_NAMES = [
#     'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
#     'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
#     'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
#     'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
#     'num_access_files', 'num_outbound_cmds', 'is_host_login',
#     'is_guest_login', 'count', 'srv_count', 'serror_rate',
#     'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
#     'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
#     'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
#     'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
#     'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
#     'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
#     'dst_host_srv_rerror_rate', 'label', 'difficulty'
# ]

# # Features categoriche
# CATEGORICAL_FEATURES = ['protocol_type', 'service', 'flag']


# def load_nsl_kdd(filepath, binary=True):
#     """
#     Carica dataset NSL-KDD.
    
#     Args:
#         filepath: Path al file .txt
#         binary: Se True, converti a Normal (0) vs Anomaly (1)
    
#     Returns:
#         DataFrame pandas
#     """
#     df = pd.read_csv(filepath, names=COLUMN_NAMES)
    
#     if binary:
#         df['label'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)
    
#     # Rimuovi colonna difficulty (non utile per training)
#     df = df.drop('difficulty', axis=1)
    
#     print(f"Loaded {len(df)} samples from {filepath}")
#     print(f"Class distribution:\n{df['label'].value_counts()}\n")
    
#     return df


# def encode_categorical(df, encoders=None, fit=True):
#     """
#     Encoding features categoriche con LabelEncoder.
    
#     Args:
#         df: DataFrame
#         encoders: Dict con LabelEncoder già fitted (per test set)
#         fit: Se True, fit encoders su df. Se False, usa encoders passati
    
#     Returns:
#         df_encoded, encoders_dict
#     """
#     df_encoded = df.copy()
    
#     if encoders is None:
#         encoders = {}
    
#     for col in CATEGORICAL_FEATURES:
#         if fit:
#             le = LabelEncoder()
#             df_encoded[col] = le.fit_transform(df[col])
#             encoders[col] = le
#         else:
#             # Usa encoder già fitted (test set)
#             le = encoders[col]
#             # Gestisci valori non visti nel training
#             df_encoded[col] = df[col].apply(
#                 lambda x: le.transform([x])[0] if x in le.classes_ else -1
#             )
    
#     return df_encoded, encoders


# def normalize_features(X, scaler=None, fit=True):
#     """
#     Normalizzazione con StandardScaler (media=0, std=1).
    
#     Args:
#         X: Array numpy o DataFrame (solo features numeriche)
#         scaler: StandardScaler già fitted
#         fit: Se True, fit scaler
    
#     Returns:
#         X_scaled, scaler
#     """
#     if scaler is None:
#         scaler = StandardScaler()
    
#     if fit:
#         X_scaled = scaler.fit_transform(X)
#     else:
#         X_scaled = scaler.transform(X)
    
#     return X_scaled, scaler


# def preprocess_nsl_kdd(train_path, test_path, save_dir='../data/processed'):
#     """
#     Pipeline completa: load → encode → normalize → save.
    
#     Args:
#         train_path: Path a KDDTrain+.txt
#         test_path: Path a KDDTest+.txt
#         save_dir: Dove salvare dati preprocessati
    
#     Returns:
#         X_train, X_test, y_train, y_test, encoders, scaler
#     """
#     print("="*60)
#     print("PREPROCESSING NSL-KDD DATASET")
#     print("="*60)
    
#     # 1. Load
#     print("\n[1/4] Loading data...")
#     train_df = load_nsl_kdd(train_path)
#     test_df = load_nsl_kdd(test_path)
    
#     # 2. Encode categorical
#     print("[2/4] Encoding categorical features...")
#     train_encoded, encoders = encode_categorical(train_df, fit=True)
#     test_encoded, _ = encode_categorical(test_df, encoders=encoders, fit=False)
    
#     # 3. Separa features e labels
#     X_train = train_encoded.drop('label', axis=1).values
#     y_train = train_encoded['label'].values
#     X_test = test_encoded.drop('label', axis=1).values
#     y_test = test_encoded['label'].values
    
#     print(f"X_train shape: {X_train.shape}")
#     print(f"X_test shape: {X_test.shape}")
    
#     # 4. Normalize
#     print("[3/4] Normalizing features...")
#     X_train_scaled, scaler = normalize_features(X_train, fit=True)
#     X_test_scaled, _ = normalize_features(X_test, scaler=scaler, fit=False)
    
#     # 5. Save
#     print("[4/4] Saving preprocessed data...")
#     import os
#     os.makedirs(save_dir, exist_ok=True)
    
#     joblib.dump(X_train_scaled, f'{save_dir}/X_train.pkl')
#     joblib.dump(X_test_scaled, f'{save_dir}/X_test.pkl')
#     joblib.dump(y_train, f'{save_dir}/y_train.pkl')
#     joblib.dump(y_test, f'{save_dir}/y_test.pkl')
#     joblib.dump(encoders, f'{save_dir}/encoders.pkl')
#     joblib.dump(scaler, f'{save_dir}/scaler.pkl')
    
#     print(f"\n✅ Preprocessing complete! Saved to {save_dir}/")
#     print("="*60)
    
#     return X_train_scaled, X_test_scaled, y_train, y_test, encoders, scaler


# # Test standalone
# if __name__ == '__main__':
#     preprocess_nsl_kdd(
#         train_path='../data/raw/KDDTrain+.txt',
#         test_path='../data/raw/KDDTest+.txt'
#     )


"""
Preprocessing pipeline per CICIoT2023 dataset.

Dataset Info:
- 46 features numeriche estratte da flussi IoT
- 33 tipi di attacco + Benign
- ~46 milioni di record (dataset completo)
- Format: Multiple CSV files (part-*.csv)

Pipeline:
1. Load & merge CSV files (con sampling opzionale)
2. Handle missing/infinite values
3. Binary label encoding (Normal=0, Anomaly=1)
4. Feature normalization (StandardScaler)
5. Train/test split
6. Save preprocessed data
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import glob
import os
from tqdm import tqdm


def load_ciciot2023(data_dir, sample_size=None, random_state=42):
    """
    Carica dataset CICIoT2023 da file CSV multipli.
    
    Args:
        data_dir: Directory contenente i file part-*.csv
        sample_size: Se specificato, carica solo N righe per file (utile per testing)
        random_state: Seed per riproducibilità
    
    Returns:
        DataFrame pandas
    """
    print("="*80)
    print("LOADING CICIoT2023 DATASET")
    print("="*80)
    
    # Trova tutti i CSV files
    csv_files = sorted(glob.glob(f'{data_dir}/*.csv'))
    
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {data_dir}\n"
            f"Download dataset from:\n"
            f"  - https://www.unb.ca/cic/datasets/iotdataset-2023.html\n"
            f"  - https://www.kaggle.com/datasets/himadri07/ciciot2023"
        )
    
    print(f"Found {len(csv_files)} CSV files")
    
    # Carica e concatena
    dfs = []
    total_rows = 0
    
    for csv_file in tqdm(csv_files, desc="Loading CSV files"):
        try:
            if sample_size:
                df_chunk = pd.read_csv(csv_file, nrows=sample_size)
            else:
                df_chunk = pd.read_csv(csv_file)
            
            dfs.append(df_chunk)
            total_rows += len(df_chunk)
            
            # Per dataset grandi, stampa progress
            if total_rows % 1000000 == 0:
                print(f"  Loaded {total_rows:,} rows so far...")
                
        except Exception as e:
            print(f"  ⚠️ Error loading {csv_file}: {e}")
            continue
    
    if not dfs:
        raise ValueError("No data loaded successfully!")
    
    # Concatena tutti i dataframe
    print("\nConcatenating dataframes...")
    df = pd.concat(dfs, ignore_index=True)
    
    print(f"\n✅ Loaded {len(df):,} total rows")
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Info sulle colonne
    print(f"\nColumns: {len(df.columns)}")
    print(f"  Features: {len(df.columns) - 1}")  # Escludi label
    print(f"  Label column: '{df.columns[-1]}'")
    
    return df


def clean_data(df):
    """
    Pulizia dati: gestisce missing values, infiniti, duplicati.
    
    Args:
        df: DataFrame originale
    
    Returns:
        DataFrame pulito
    """
    print("\n" + "="*80)
    print("DATA CLEANING")
    print("="*80)
    
    initial_rows = len(df)
    
    # 1. Controlla valori mancanti
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        print(f"\n⚠️ Found {missing_count:,} missing values")
        print("Columns with missing values:")
        print(df.isnull().sum()[df.isnull().sum() > 0])
        
        # Riempi con 0 (strategia comune per network features)
        df = df.fillna(0)
        print("✅ Missing values filled with 0")
    else:
        print("✅ No missing values")
    
    # 2. Controlla valori infiniti
    # Identifica colonne numeriche
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    inf_count = 0
    for col in numeric_cols:
        inf_mask = np.isinf(df[col])
        if inf_mask.any():
            inf_count += inf_mask.sum()
            # Sostituisci inf con max value non-inf della colonna
            max_val = df[col][~inf_mask].max()
            df.loc[inf_mask, col] = max_val
    
    if inf_count > 0:
        print(f"\n⚠️ Found {inf_count:,} infinite values")
        print("✅ Infinite values replaced with column max")
    else:
        print("✅ No infinite values")
    
    # 3. Rimuovi duplicati (opzionale, può essere lento)
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"\n⚠️ Found {duplicates:,} duplicate rows")
        df = df.drop_duplicates()
        print(f"✅ Duplicates removed")
    else:
        print("✅ No duplicates")
    
    final_rows = len(df)
    removed = initial_rows - final_rows
    
    print(f"\nRows removed: {removed:,} ({removed/initial_rows*100:.2f}%)")
    print(f"Final shape: {df.shape}")
    
    return df


def encode_labels(df, label_col=None):
    """
    Encoding labels binario: Benign=0, Attack=1.
    
    Args:
        df: DataFrame
        label_col: Nome colonna label (se None, usa ultima colonna)
    
    Returns:
        df con binary_label, label_col
    """
    print("\n" + "="*80)
    print("LABEL ENCODING")
    print("="*80)
    
    if label_col is None:
        label_col = df.columns[-1]
    
    print(f"Label column: '{label_col}'")
    
    # Mostra distribuzione classi originale
    print(f"\nOriginal classes: {df[label_col].nunique()}")
    print("\nTop 10 classes:")
    print(df[label_col].value_counts().head(10))
    
    # Binary encoding
    df['binary_label'] = df[label_col].apply(
        lambda x: 0 if str(x).strip().lower() == 'benign' else 1
    )
    
    # Statistiche
    binary_dist = df['binary_label'].value_counts().sort_index()
    print("\n" + "-"*80)
    print("Binary distribution:")
    print(f"  Normal (0):  {binary_dist.get(0, 0):>10,} ({binary_dist.get(0, 0)/len(df)*100:>6.2f}%)")
    print(f"  Anomaly (1): {binary_dist.get(1, 0):>10,} ({binary_dist.get(1, 0)/len(df)*100:>6.2f}%)")
    print(f"\nAnomaly rate: {df['binary_label'].mean():.2%}")
    
    return df, label_col


def prepare_features(df, label_col):
    """
    Prepara X (features) e y (labels).
    
    Args:
        df: DataFrame con binary_label
        label_col: Nome colonna label originale
    
    Returns:
        X, y, feature_names
    """
    print("\n" + "="*80)
    print("FEATURE PREPARATION")
    print("="*80)
    
    # Rimuovi colonne non-feature
    cols_to_drop = [label_col, 'binary_label']
    
    # Gestisci possibili colonne aggiuntive da rimuovere
    # (es. 'Unnamed: 0', index columns, ecc.)
    for col in df.columns:
        if 'unnamed' in col.lower() or col.lower() in ['index', 'id']:
            cols_to_drop.append(col)
    
    # Features
    feature_cols = [col for col in df.columns if col not in cols_to_drop]
    X = df[feature_cols].values
    y = df['binary_label'].values
    
    print(f"Features shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"\nFeature names ({len(feature_cols)}):")
    print(feature_cols[:10], "..." if len(feature_cols) > 10 else "")
    
    return X, y, feature_cols


def normalize_features(X_train, X_test):
    """
    Normalizzazione con StandardScaler.
    
    Args:
        X_train: Training features
        X_test: Test features
    
    Returns:
        X_train_scaled, X_test_scaled, scaler
    """
    print("\n" + "="*80)
    print("FEATURE NORMALIZATION")
    print("="*80)
    
    scaler = StandardScaler()
    
    print("Fitting scaler on training data...")
    X_train_scaled = scaler.fit_transform(X_train)
    
    print("Transforming test data...")
    X_test_scaled = scaler.transform(X_test)
    
    print("✅ Normalization complete")
    print(f"   Mean: ~{X_train_scaled.mean():.6f} (target: 0)")
    print(f"   Std:  ~{X_train_scaled.std():.6f} (target: 1)")
    
    return X_train_scaled, X_test_scaled, scaler


def preprocess_ciciot2023(
    data_dir, 
    output_dir='../data/processed',
    test_size=0.2,
    sample_size=None,
    random_state=42
):
    """
    Pipeline completa di preprocessing per CICIoT2023.
    
    Args:
        data_dir: Directory con file CSV del dataset
        output_dir: Dove salvare dati preprocessati
        test_size: Percentuale per test set
        sample_size: Se specificato, carica solo N righe per file (per testing)
        random_state: Seed per riproducibilità
    
    Returns:
        X_train, X_test, y_train, y_test, feature_names, scaler
    """
    print("\n" + "🚀"*40)
    print("CICIoT2023 PREPROCESSING PIPELINE")
    print("🚀"*40)
    
    # 1. Load
    df = load_ciciot2023(data_dir, sample_size=sample_size, random_state=random_state)
    
    # 2. Clean
    df = clean_data(df)
    
    # 3. Encode labels
    df, label_col = encode_labels(df)
    
    # 4. Prepare features
    X, y, feature_names = prepare_features(df, label_col)
    
    # 5. Train/test split
    print("\n" + "="*80)
    print("TRAIN/TEST SPLIT")
    print("="*80)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y  # Mantieni stessa proporzione Normal/Anomaly
    )
    
    print(f"Train set: {X_train.shape[0]:,} samples ({(1-test_size)*100:.0f}%)")
    print(f"Test set:  {X_test.shape[0]:,} samples ({test_size*100:.0f}%)")
    print(f"\nTrain anomaly rate: {y_train.mean():.2%}")
    print(f"Test anomaly rate:  {y_test.mean():.2%}")
    
    # 6. Normalize
    X_train_scaled, X_test_scaled, scaler = normalize_features(X_train, X_test)
    
    # 7. Save
    print("\n" + "="*80)
    print("SAVING PREPROCESSED DATA")
    print("="*80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    joblib.dump(X_train_scaled, f'{output_dir}/X_train.pkl')
    joblib.dump(X_test_scaled, f'{output_dir}/X_test.pkl')
    joblib.dump(y_train, f'{output_dir}/y_train.pkl')
    joblib.dump(y_test, f'{output_dir}/y_test.pkl')
    joblib.dump(feature_names, f'{output_dir}/feature_names.pkl')
    joblib.dump(scaler, f'{output_dir}/scaler.pkl')
    
    # Salva anche info dataset
    dataset_info = {
        'total_samples': len(df),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'num_features': len(feature_names),
        'anomaly_rate': y.mean(),
        'test_size': test_size,
        'random_state': random_state
    }
    joblib.dump(dataset_info, f'{output_dir}/dataset_info.pkl')
    
    print(f"\n✅ All files saved to {output_dir}/")
    print("\nSaved files:")
    print("  - X_train.pkl, X_test.pkl")
    print("  - y_train.pkl, y_test.pkl")
    print("  - feature_names.pkl")
    print("  - scaler.pkl")
    print("  - dataset_info.pkl")
    
    print("\n" + "="*80)
    print("✅ PREPROCESSING COMPLETE!")
    print("="*80)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names, scaler


# Test standalone
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess CICIoT2023 dataset')
    parser.add_argument('--data-dir', type=str, default='../data/raw/CICIoT2023',
                        help='Directory containing CSV files')
    parser.add_argument('--output-dir', type=str, default='../data/processed',
                        help='Output directory for preprocessed data')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Test set size (0-1)')
    parser.add_argument('--sample', type=int, default=None,
                        help='Sample N rows per file (for testing)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Run preprocessing
    preprocess_ciciot2023(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        test_size=args.test_size,
        sample_size=args.sample,
        random_state=args.seed
    )
    
    print("\n💡 Next steps:")
    print("   1. Run: python src/train_decision_tree.py")
    print("   2. Run: python src/train_random_forest.py")