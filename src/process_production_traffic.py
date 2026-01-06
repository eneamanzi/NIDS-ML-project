"""
Processing per traffico reale catturato in produzione.

Questo script:
1. Carica traffico grezzo da PCAP o CSV
2. Applica stessa trasformazione del training (scaler, encoder)
3. Salva in formato processato per inferenza
4. Mantiene tracciabilità (timestamp, source)

Usage:
    python src/process_production_traffic.py \\
        --input traffic_capture.csv \\
        --artifacts-dir data/processed/original \\
        --output-dir data/processed/production
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from pathlib import Path
import json


def print_header(text):
    """Stampa header formattato."""
    print("\n" + "="*80)
    print(text.center(80))
    print("="*80 + "\n")


def load_artifacts(artifacts_dir):
    """
    Carica scaler e label encoder dal training.
    
    Returns:
        scaler, label_encoder, mapping_info
    """
    print_header("LOADING ARTIFACTS")
    
    # Scaler
    scaler_path = f"{artifacts_dir}/scaler.pkl"
    scaler = joblib.load(scaler_path)
    print(f"✅ Scaler loaded: {scaler_path}")
    
    # Label encoder
    encoder_path = f"{artifacts_dir}/label_encoder.pkl"
    label_encoder = joblib.load(encoder_path)
    print(f"✅ Label encoder loaded: {encoder_path}")
    print(f"   Classes: {label_encoder.classes_}")
    
    # Mapping info
    mapping_path = f"{artifacts_dir}/mapping_info.json"
    with open(mapping_path, 'r') as f:
        mapping_info = json.load(f)
    print(f"✅ Mapping info loaded: {mapping_path}")
    
    return scaler, label_encoder, mapping_info


def load_raw_traffic(input_path, label_col=None):
    """
    Carica traffico grezzo da CSV.
    
    Args:
        input_path: Path al file CSV
        label_col: Se presente, nome colonna label (per validazione)
    
    Returns:
        DataFrame, has_labels
    """
    print_header("LOADING RAW TRAFFIC")
    
    print(f"Input: {input_path}")
    
    # Determina formato
    if input_path.endswith('.csv'):
        df = pd.read_csv(input_path)
    elif input_path.endswith('.pkl'):
        df = pd.read_pickle(input_path)
    else:
        raise ValueError(f"Unsupported format: {input_path}. Use .csv or .pkl")
    
    print(f"Loaded: {len(df):,} records")
    print(f"Columns: {len(df.columns)}")
    
    # Check se ha labels
    has_labels = label_col is not None and label_col in df.columns
    
    if has_labels:
        print(f"✅ Labels found: '{label_col}'")
        print(f"   Unique labels: {df[label_col].nunique()}")
    else:
        print("ℹ️  No labels (unlabeled production traffic)")
    
    return df, has_labels


def clean_production_data(df):
    """Pulizia base per traffico di produzione."""
    print_header("CLEANING DATA")
    
    initial_rows = len(df)
    
    # Missing values
    missing = df.isnull().sum().sum()
    if missing > 0:
        print(f"⚠️  Found {missing:,} missing values → filling with 0")
        df = df.fillna(0)
    else:
        print("✅ No missing values")
    
    # Infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_count = 0
    
    for col in numeric_cols:
        inf_mask = np.isinf(df[col])
        if inf_mask.any():
            inf_count += inf_mask.sum()
            max_val = df[col][~inf_mask].max()
            df.loc[inf_mask, col] = max_val
    
    if inf_count > 0:
        print(f"⚠️  Found {inf_count:,} infinite values → replaced")
    else:
        print("✅ No infinite values")
    
    # Duplicates (opzionale per production)
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"ℹ️  Found {duplicates:,} duplicate flows (keeping all)")
    
    print(f"\nFinal: {len(df):,} records")
    
    return df


def process_production_traffic(df, scaler, feature_cols, label_col=None, 
                               label_encoder=None, mapping_info=None):
    """
    Processa traffico di produzione usando artifacts del training.
    
    Args:
        df: DataFrame grezzo
        scaler: StandardScaler fitted sul training
        feature_cols: Lista colonne feature (da training)
        label_col: Nome colonna label (se presente)
        label_encoder: LabelEncoder (se labels presenti)
        mapping_info: Info mapping (se labels presenti)
    
    Returns:
        DataFrame processato
    """
    print_header("PROCESSING TRAFFIC")
    
    # Verifica che features esistano
    missing_features = [col for col in feature_cols if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")
    
    print(f"Features: {len(feature_cols)}")
    
    # Estrai features
    X = df[feature_cols].values
    
    # Applica scaler (TRANSFORM, non fit!)
    print("\nApplying StandardScaler (transform only)...")
    X_scaled = scaler.transform(X)
    
    # Crea DataFrame processato
    df_processed = pd.DataFrame(X_scaled, columns=feature_cols, index=df.index)
    
    # Aggiungi metadata
    df_processed['timestamp'] = datetime.now().isoformat()
    df_processed['processed'] = True
    
    # Se ha labels, processale
    if label_col and label_col in df.columns:
        print("\nProcessing labels...")
        
        # Mappa a macro-categorie
        attack_mapping = mapping_info.get('attack_mapping', {})
        df_processed['y_macro'] = df[label_col].map(attack_mapping)
        
        # Encode
        df_processed['y_macro_encoded'] = label_encoder.transform(df_processed['y_macro'])
        
        # Specific
        specific_mapping = mapping_info.get('specific_to_idx', {})
        df_processed['y_specific'] = df[label_col].map(specific_mapping)
        
        print(f"✅ Labels processed")
        print(f"   Original labels preserved in metadata")
    else:
        print("\nℹ️  No labels to process (inference mode)")
        df_processed['y_macro_encoded'] = None
        df_processed['y_specific'] = None
    
    print(f"\n✅ Processing complete")
    print(f"   Output shape: {df_processed.shape}")
    
    return df_processed


def save_production_dataset(df_processed, output_dir, source_filename):
    """
    Salva dataset di produzione con naming timestamp-based.
    """
    print_header("SAVING PRODUCTION DATASET")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Genera nome file con timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    source_name = Path(source_filename).stem
    output_filename = f"production_{source_name}_{timestamp}.pkl"
    output_path = f"{output_dir}/{output_filename}"
    
    # Salva
    df_processed.to_pickle(output_path)
    size_mb = os.path.getsize(output_path) / 1024**2
    
    print(f"✅ Saved: {output_path}")
    print(f"   Shape: {df_processed.shape}")
    print(f"   Size: {size_mb:.2f} MB")
    
    # Salva metadata
    metadata = {
        'source_file': source_filename,
        'processed_timestamp': datetime.now().isoformat(),
        'num_records': len(df_processed),
        'num_features': len([col for col in df_processed.columns if col not in ['timestamp', 'processed', 'y_macro', 'y_macro_encoded', 'y_specific']]),
        'has_labels': df_processed['y_macro_encoded'].notna().any()
    }
    
    metadata_path = f"{output_dir}/{output_filename.replace('.pkl', '_metadata.json')}"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Metadata: {metadata_path}")
    
    return output_path


def main(input_path, artifacts_dir, output_dir, label_col=None):
    """
    Main pipeline per processare traffico di produzione.
    """
    print_header("🚀 PRODUCTION TRAFFIC PROCESSING")
    
    print("This script processes real-world traffic using trained artifacts.")
    print("Output can be used for:")
    print("  - Real-time inference")
    print("  - Model evaluation on production data")
    print("  - Drift detection")
    print("  - Model retraining")
    
    # Verifica che artifacts esistano
    if not os.path.exists(artifacts_dir):
        print(f"\n❌ Error: Artifacts directory not found: {artifacts_dir}")
        print("\n💡 First train a model and save artifacts:")
        print("   python src/data_processing.py --output-dir data/processed/original")
        exit(1)
    
    # =========================================================================
    # STEP 1: LOAD ARTIFACTS
    # =========================================================================
    scaler, label_encoder, mapping_info = load_artifacts(artifacts_dir)
    
    # Determina feature columns dal training
    # (Assumiamo che siano le stesse del training originale)
    # In alternativa, leggere da mapping_info
    print("\n💡 Feature columns will be determined from input data")
    
    # =========================================================================
    # STEP 2: LOAD RAW TRAFFIC
    # =========================================================================
    df, has_labels = load_raw_traffic(input_path, label_col=label_col)
    
    # =========================================================================
    # STEP 3: CLEAN DATA
    # =========================================================================
    df = clean_production_data(df)
    
    # =========================================================================
    # STEP 4: IDENTIFY FEATURES
    # =========================================================================
    print_header("IDENTIFYING FEATURES")
    
    # Rimuovi colonne non-feature
    exclude_cols = []
    if label_col and label_col in df.columns:
        exclude_cols.append(label_col)
    
    # Colonne che potrebbero essere metadata
    metadata_cols = ['timestamp', 'source', 'id', 'flow_id', 'Unnamed: 0']
    for col in metadata_cols:
        if col in df.columns:
            exclude_cols.append(col)
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    print(f"Feature columns identified: {len(feature_cols)}")
    print(f"Excluded columns: {exclude_cols}")
    
    # =========================================================================
    # STEP 5: PROCESS TRAFFIC
    # =========================================================================
    df_processed = process_production_traffic(
        df, scaler, feature_cols,
        label_col=label_col if has_labels else None,
        label_encoder=label_encoder if has_labels else None,
        mapping_info=mapping_info if has_labels else None
    )
    
    # =========================================================================
    # STEP 6: SAVE
    # =========================================================================
    output_path = save_production_dataset(
        df_processed, output_dir, 
        source_filename=os.path.basename(input_path)
    )
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print_header("✅ PRODUCTION PROCESSING COMPLETE!")
    
    print("Summary:")
    print(f"  Input: {input_path}")
    print(f"  Records: {len(df_processed):,}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Has labels: {'Yes' if has_labels else 'No'}")
    print(f"  Output: {output_path}")
    
    print("\n💡 Next steps:")
    if has_labels:
        print("   1. Evaluate model on production data:")
        print(f"      python src/evaluate_model.py --data {output_path}")
    else:
        print("   1. Run inference:")
        print(f"      python src/run_inference.py --data {output_path}")
    
    print("   2. Monitor for data drift")
    print("   3. Consider retraining if performance degrades")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Process production traffic using trained artifacts'
    )
    parser.add_argument('--input', type=str, required=True,
                        help='Input traffic file (CSV or PKL)')
    parser.add_argument('--artifacts-dir', type=str, required=True,
                        help='Directory with training artifacts (scaler, encoder)')
    parser.add_argument('--output-dir', type=str, default='data/processed/production',
                        help='Output directory for processed production data')
    parser.add_argument('--label-col', type=str, default=None,
                        help='Label column name (if present in production data)')
    
    args = parser.parse_args()
    
    main(
        input_path=args.input,
        artifacts_dir=args.artifacts_dir,
        output_dir=args.output_dir,
        label_col=args.label_col
    )