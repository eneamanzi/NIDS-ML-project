"""
Preprocessing pipeline per NSL-KDD dataset.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# Definizione colonne (globale per consistenza)
COLUMN_NAMES = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
    'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
    'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
    'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
    'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'label', 'difficulty'
]

# Features categoriche
CATEGORICAL_FEATURES = ['protocol_type', 'service', 'flag']


def load_nsl_kdd(filepath, binary=True):
    """
    Carica dataset NSL-KDD.
    
    Args:
        filepath: Path al file .txt
        binary: Se True, converti a Normal (0) vs Anomaly (1)
    
    Returns:
        DataFrame pandas
    """
    df = pd.read_csv(filepath, names=COLUMN_NAMES)
    
    if binary:
        df['label'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)
    
    # Rimuovi colonna difficulty (non utile per training)
    df = df.drop('difficulty', axis=1)
    
    print(f"Loaded {len(df)} samples from {filepath}")
    print(f"Class distribution:\n{df['label'].value_counts()}\n")
    
    return df


def encode_categorical(df, encoders=None, fit=True):
    """
    Encoding features categoriche con LabelEncoder.
    
    Args:
        df: DataFrame
        encoders: Dict con LabelEncoder già fitted (per test set)
        fit: Se True, fit encoders su df. Se False, usa encoders passati
    
    Returns:
        df_encoded, encoders_dict
    """
    df_encoded = df.copy()
    
    if encoders is None:
        encoders = {}
    
    for col in CATEGORICAL_FEATURES:
        if fit:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df[col])
            encoders[col] = le
        else:
            # Usa encoder già fitted (test set)
            le = encoders[col]
            # Gestisci valori non visti nel training
            df_encoded[col] = df[col].apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
    
    return df_encoded, encoders


def normalize_features(X, scaler=None, fit=True):
    """
    Normalizzazione con StandardScaler (media=0, std=1).
    
    Args:
        X: Array numpy o DataFrame (solo features numeriche)
        scaler: StandardScaler già fitted
        fit: Se True, fit scaler
    
    Returns:
        X_scaled, scaler
    """
    if scaler is None:
        scaler = StandardScaler()
    
    if fit:
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    
    return X_scaled, scaler


def preprocess_nsl_kdd(train_path, test_path, save_dir='../data/processed'):
    """
    Pipeline completa: load → encode → normalize → save.
    
    Args:
        train_path: Path a KDDTrain+.txt
        test_path: Path a KDDTest+.txt
        save_dir: Dove salvare dati preprocessati
    
    Returns:
        X_train, X_test, y_train, y_test, encoders, scaler
    """
    print("="*60)
    print("PREPROCESSING NSL-KDD DATASET")
    print("="*60)
    
    # 1. Load
    print("\n[1/4] Loading data...")
    train_df = load_nsl_kdd(train_path)
    test_df = load_nsl_kdd(test_path)
    
    # 2. Encode categorical
    print("[2/4] Encoding categorical features...")
    train_encoded, encoders = encode_categorical(train_df, fit=True)
    test_encoded, _ = encode_categorical(test_df, encoders=encoders, fit=False)
    
    # 3. Separa features e labels
    X_train = train_encoded.drop('label', axis=1).values
    y_train = train_encoded['label'].values
    X_test = test_encoded.drop('label', axis=1).values
    y_test = test_encoded['label'].values
    
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    
    # 4. Normalize
    print("[3/4] Normalizing features...")
    X_train_scaled, scaler = normalize_features(X_train, fit=True)
    X_test_scaled, _ = normalize_features(X_test, scaler=scaler, fit=False)
    
    # 5. Save
    print("[4/4] Saving preprocessed data...")
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    joblib.dump(X_train_scaled, f'{save_dir}/X_train.pkl')
    joblib.dump(X_test_scaled, f'{save_dir}/X_test.pkl')
    joblib.dump(y_train, f'{save_dir}/y_train.pkl')
    joblib.dump(y_test, f'{save_dir}/y_test.pkl')
    joblib.dump(encoders, f'{save_dir}/encoders.pkl')
    joblib.dump(scaler, f'{save_dir}/scaler.pkl')
    
    print(f"\n✅ Preprocessing complete! Saved to {save_dir}/")
    print("="*60)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, encoders, scaler


# Test standalone
if __name__ == '__main__':
    preprocess_nsl_kdd(
        train_path='../data/raw/KDDTrain+.txt',
        test_path='../data/raw/KDDTest+.txt'
    )
