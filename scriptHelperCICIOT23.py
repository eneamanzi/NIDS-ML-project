#!/usr/bin/env python3
"""
Script helper per download e setup del dataset CICIoT2023.

Usage:
    python setup_dataset.py
"""

import os
import sys
import subprocess
import zipfile
from pathlib import Path


def print_header(text):
    """Stampa header formattato."""
    print("\n" + "="*80)
    print(text.center(80))
    print("="*80 + "\n")


def check_kaggle():
    """Controlla se Kaggle CLI è installato e configurato."""
    print_header("CHECKING KAGGLE CLI")
    
    # Check se kaggle è installato
    try:
        result = subprocess.run(['kaggle', '--version'], 
                              capture_output=True, text=True)
        print(f"✅ Kaggle CLI installed: {result.stdout.strip()}")
        kaggle_installed = True
    except FileNotFoundError:
        print("❌ Kaggle CLI not installed")
        kaggle_installed = False
    
    # Check se credenziali sono configurate
    kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'
    if kaggle_json.exists():
        print(f"✅ Kaggle credentials found: {kaggle_json}")
        kaggle_configured = True
    else:
        print(f"❌ Kaggle credentials not found: {kaggle_json}")
        kaggle_configured = False
    
    return kaggle_installed and kaggle_configured


def install_kaggle():
    """Installa Kaggle CLI."""
    print_header("INSTALLING KAGGLE CLI")
    
    print("Installing kaggle package...")
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'kaggle'])
    print("✅ Kaggle CLI installed!")


def setup_kaggle_credentials():
    """Guida l'utente nella configurazione delle credenziali Kaggle."""
    print_header("SETUP KAGGLE CREDENTIALS")
    
    print("To download datasets from Kaggle, you need an API key.")
    print("\nSteps:")
    print("1. Go to https://www.kaggle.com/")
    print("2. Sign in or create an account")
    print("3. Go to Account Settings (click your profile picture)")
    print("4. Scroll down to 'API' section")
    print("5. Click 'Create New API Token'")
    print("6. A file 'kaggle.json' will be downloaded")
    print()
    
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_json = kaggle_dir / 'kaggle.json'
    
    # Chiedi all'utente il path del file scaricato
    print(f"Where is your kaggle.json file?")
    print("(Press Enter for default: ~/Downloads/kaggle.json)")
    kaggle_file_path = input("> ").strip()
    
    if not kaggle_file_path:
        kaggle_file_path = str(Path.home() / 'Downloads' / 'kaggle.json')
    
    kaggle_file = Path(kaggle_file_path)
    
    if not kaggle_file.exists():
        print(f"\n❌ File not found: {kaggle_file}")
        print("Please download kaggle.json and try again.")
        return False
    
    # Crea directory .kaggle se non esiste
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    
    # Copia kaggle.json
    import shutil
    shutil.copy(kaggle_file, kaggle_json)
    
    # Set permissions (solo per Unix)
    if os.name != 'nt':  # Non Windows
        os.chmod(kaggle_json, 0o600)
    
    print(f"\n✅ Kaggle credentials configured: {kaggle_json}")
    return True


def download_dataset():
    """Scarica il dataset CICIoT2023 da Kaggle."""
    print_header("DOWNLOADING CICIoT2023 DATASET")
    
    data_dir = Path('data/raw/CICIoT2023')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Download location: {data_dir.absolute()}")
    print("\n⚠️ WARNING: Dataset size is ~2.7GB compressed, ~13GB uncompressed")
    print("This may take 10-30 minutes depending on your internet speed.")
    print()
    
    response = input("Continue with download? (y/n): ").strip().lower()
    if response != 'y':
        print("Download cancelled.")
        return False
    
    print("\nDownloading...")
    try:
        # Download dataset
        subprocess.run([
            'kaggle', 'datasets', 'download',
            '-d', 'himadri07/ciciot2023',
            '-p', str(data_dir)
        ], check=True)
        
        print("\n✅ Download complete!")
        
        # Unzip
        print("\nUnzipping files...")
        zip_file = data_dir / 'ciciot2023.zip'
        if zip_file.exists():
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            
            # Rimuovi zip
            zip_file.unlink()
            print("✅ Files extracted!")
        
        # Verifica file CSV
        csv_files = list(data_dir.glob('*.csv'))
        print(f"\n✅ Found {len(csv_files)} CSV files")
        
        if csv_files:
            print("\nFirst 5 files:")
            for f in csv_files[:5]:
                size_mb = f.stat().st_size / 1024**2
                print(f"  - {f.name} ({size_mb:.1f} MB)")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error downloading dataset: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


def manual_download_instructions():
    """Mostra istruzioni per download manuale."""
    print_header("MANUAL DOWNLOAD INSTRUCTIONS")
    
    print("If automatic download fails, download manually:")
    print()
    print("1. Go to: https://www.kaggle.com/datasets/himadri07/ciciot2023")
    print("2. Click 'Download' button (requires Kaggle account)")
    print("3. Extract the zip file")
    print("4. Copy all CSV files to: data/raw/CICIoT2023/")
    print()


def verify_dataset():
    """Verifica che il dataset sia stato scaricato correttamente."""
    print_header("VERIFYING DATASET")
    
    data_dir = Path('data/raw/CICIoT2023')
    
    if not data_dir.exists():
        print(f"❌ Directory not found: {data_dir}")
        return False
    
    csv_files = list(data_dir.glob('*.csv'))
    
    if not csv_files:
        print(f"❌ No CSV files found in {data_dir}")
        return False
    
    print(f"✅ Found {len(csv_files)} CSV files")
    
    # Calcola dimensione totale
    total_size = sum(f.stat().st_size for f in csv_files)
    total_size_gb = total_size / 1024**3
    
    print(f"✅ Total size: {total_size_gb:.2f} GB")
    
    if total_size_gb < 1:
        print("\n⚠️ Warning: Dataset size is smaller than expected.")
        print("   Make sure all files are downloaded correctly.")
    
    return True


def main():
    """Main function."""
    print("\n" + "🚀"*40)
    print("CICIoT2023 DATASET SETUP".center(80))
    print("🚀"*40)
    
    print("\nThis script will help you download the CICIoT2023 dataset.")
    print()
    
    # Check Kaggle
    kaggle_ready = check_kaggle()
    
    if not kaggle_ready:
        print("\n" + "-"*80)
        print("Kaggle CLI needs to be configured.")
        print()
        
        # Installa Kaggle se necessario
        try:
            import kaggle
        except ImportError:
            response = input("Install Kaggle CLI? (y/n): ").strip().lower()
            if response == 'y':
                install_kaggle()
            else:
                print("\nCannot proceed without Kaggle CLI.")
                manual_download_instructions()
                return
        
        # Setup credenziali
        response = input("\nSetup Kaggle credentials now? (y/n): ").strip().lower()
        if response == 'y':
            if not setup_kaggle_credentials():
                print("\nSetup failed.")
                manual_download_instructions()
                return
        else:
            print("\nCannot download without Kaggle credentials.")
            manual_download_instructions()
            return
    
    # Download dataset
    print()
    response = input("Download CICIoT2023 dataset now? (y/n): ").strip().lower()
    if response == 'y':
        success = download_dataset()
        if not success:
            manual_download_instructions()
            return
    else:
        print("\nSkipping download.")
        manual_download_instructions()
        return
    
    # Verifica
    if verify_dataset():
        print_header("✅ SETUP COMPLETE!")
        print("You can now run:")
        print("  python src/preprocessing_ciciot2023.py --sample 10000")
        print()
    else:
        print_header("❌ SETUP INCOMPLETE")
        print("Please check the instructions above.")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()