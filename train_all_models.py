"""
Automated Training Suite - Complete Model Training Pipeline

FEATURES:
- Train multiple algorithms (DT, RF, k-NN, SVM)
- Multiple datasets (original, smote)
- Flexible filtering via CLI arguments
- Parallel or sequential execution
- Progress tracking with timing
- Comprehensive error handling
- Result summary with comparison

Usage:
    # Train all models on all datasets (default)
    python src/train_all_models.py
    
    # Train specific algorithm
    python src/train_all_models.py --algorithms DecisionTree RandomForest
    
    # Train on specific dataset
    python src/train_all_models.py --datasets smote
    
    # Specific combination
    python src/train_all_models.py --algorithms RandomForest --datasets original smote
    
    # Parallel execution (faster)
    python src/train_all_models.py --parallel
    
    # Dry run (show what would be trained)
    python src/train_all_models.py --dry-run
"""

import subprocess
import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime
import argparse
from typing import List, Dict, Tuple

# =============================================================================
# CONFIGURATION
# =============================================================================

# Algorithm configurations
ALGORITHMS = {
    'DecisionTree': {
        'name': 'Decision Tree',
        'script': 'train_decision_tree.py',
        'emoji': '🌳',
        'color': '\033[94m'  # Blue
    },
    'RandomForest': {
        'name': 'Random Forest',
        'script': 'train_random_forest.py',
        'emoji': '🌲',
        'color': '\033[92m'  # Green
    },
    'kNN': {
        'name': 'k-NN',
        'script': 'train_knn.py',
        'emoji': '🎯',
        'color': '\033[95m'  # Purple
    },
    'SVM': {
        'name': 'SVM',
        'script': 'train_svm.py',
        'emoji': '🔷',
        'color': '\033[93m'  # Yellow
    }
}

DATASETS = ['original', 'smote']

# =============================================================================
# PATH RESOLUTION (Indipendente da dove viene invocato)
# =============================================================================

# 1. Trova la directory dello script stesso
SCRIPT_PATH = Path(__file__).resolve()

# 2. Se lo script è in src/, parent è project root
#    Se lo script è in project root, usa current dir
if SCRIPT_PATH.parent.name == 'src':
    BASE_DIR = SCRIPT_PATH.parent.parent
    SRC_DIR = SCRIPT_PATH.parent
else:
    # Script eseguito da project root o altro path
    BASE_DIR = SCRIPT_PATH.parent
    SRC_DIR = BASE_DIR / 'src'

# 3. Verifica che SRC_DIR esista
if not SRC_DIR.exists():
    print(f"ERROR: src/ directory not found at {SRC_DIR}")
    print(f"Script location: {SCRIPT_PATH}")
    print(f"Looking for src/ at: {SRC_DIR}")
    sys.exit(1)

LOGS_DIR = BASE_DIR / 'logs' / 'training'

# Debug info (commentare in produzione)
# print(f"[DEBUG] Script path: {SCRIPT_PATH}")
# print(f"[DEBUG] Base dir: {BASE_DIR}")
# print(f"[DEBUG] Src dir: {SRC_DIR}")

# Colors
COLORS = {
    'HEADER': '\033[95m',
    'BLUE': '\033[94m',
    'CYAN': '\033[96m',
    'GREEN': '\033[92m',
    'YELLOW': '\033[93m',
    'RED': '\033[91m',
    'BOLD': '\033[1m',
    'UNDERLINE': '\033[4m',
    'END': '\033[0m'
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def print_header(text: str):
    """Print formatted header."""
    width = 80
    print(f"\n{COLORS['BLUE']}{'='*width}{COLORS['END']}")
    print(f"{COLORS['BOLD']}{text.center(width)}{COLORS['END']}")
    print(f"{COLORS['BLUE']}{'='*width}{COLORS['END']}\n")


def print_section(text: str):
    """Print formatted section."""
    print(f"\n{COLORS['CYAN']}{'-'*80}{COLORS['END']}")
    print(f"{COLORS['CYAN']}{text}{COLORS['END']}")
    print(f"{COLORS['CYAN']}{'-'*80}{COLORS['END']}")


def print_success(text: str):
    """Print success message."""
    print(f"{COLORS['GREEN']}✅ {text}{COLORS['END']}")


def print_error(text: str):
    """Print error message."""
    print(f"{COLORS['RED']}❌ {text}{COLORS['END']}")


def print_warning(text: str):
    """Print warning message."""
    print(f"{COLORS['YELLOW']}⚠️  {text}{COLORS['END']}")


def print_info(text: str):
    """Print info message."""
    print(f"{COLORS['CYAN']}ℹ️  {text}{COLORS['END']}")


def format_time(seconds: int) -> str:
    """Format seconds to readable time."""
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        mins = seconds // 60
        secs = seconds % 60
        return f"{mins}m {secs}s"
    else:
        hours = seconds // 3600
        mins = (seconds % 3600) // 60
        return f"{hours}h {mins}m"


# =============================================================================
# VALIDATION
# =============================================================================

def validate_arguments(algorithms: List[str], datasets: List[str]) -> Tuple[List[str], List[str]]:
    """
    Validate and normalize algorithm and dataset arguments.
    
    Returns:
        Tuple of (valid_algorithms, valid_datasets)
    """
    # Validate algorithms
    valid_algorithms = []
    invalid_algorithms = []
    
    for algo in algorithms:
        if algo in ALGORITHMS:
            valid_algorithms.append(algo)
        else:
            invalid_algorithms.append(algo)
    
    if invalid_algorithms:
        print_error(f"Invalid algorithms: {', '.join(invalid_algorithms)}")
        print_info(f"Valid options: {', '.join(ALGORITHMS.keys())}")
        sys.exit(1)
    
    # Validate datasets
    valid_datasets = []
    invalid_datasets = []
    
    for dataset in datasets:
        if dataset in DATASETS:
            valid_datasets.append(dataset)
        else:
            invalid_datasets.append(dataset)
    
    if invalid_datasets:
        print_error(f"Invalid datasets: {', '.join(invalid_datasets)}")
        print_info(f"Valid options: {', '.join(DATASETS)}")
        sys.exit(1)
    
    return valid_algorithms, valid_datasets


def check_prerequisites(algorithms: List[str], datasets: List[str]) -> bool:
    """Check if all prerequisites are met."""
    print_section("Pre-Flight Checks")
    
    all_ok = True
    
    # Check Python
    print(f"Python version: {sys.version.split()[0]}")
    print(f"Script location: {SCRIPT_PATH}")
    print(f"Project root: {BASE_DIR}")
    print(f"Source dir: {SRC_DIR}")
    print()
    
    # Check training scripts
    print("Checking training scripts:")
    for algo in algorithms:
        script_name = ALGORITHMS[algo]['script']
        script_path = SRC_DIR / script_name
        
        if not script_path.exists():
            print_error(f"Not found: {script_path}")
            all_ok = False
        else:
            print_success(f"Found: {script_name}")
    
    print()
    
    # Check data directories
    data_dir_map = {
        'original': BASE_DIR / 'data' / 'processed' / 'CICIOT23',
        'smote': BASE_DIR / 'data' / 'processed' / 'SMOTE'
    }
    
    print("Checking data directories:")
    for dataset in datasets:
        data_dir = data_dir_map[dataset]
        
        if not data_dir.exists():
            print_warning(f"Data directory not found: {data_dir}")
            print_info(f"Trainings on '{dataset}' dataset will fail")
            all_ok = False
            continue
        
        # CRITICAL: SMOTE usa train_smote.pkl, ma test/val sono in CICIOT23/
        if dataset == 'smote':
            # Per SMOTE verifica solo train_smote.pkl
            train_file = data_dir / 'train_smote.pkl'
            if not train_file.exists():
                print_warning(f"Missing: {train_file}")
                all_ok = False
            else:
                print_success(f"Dataset 'smote': train_smote.pkl found")
            
            # Test e validation sono in CICIOT23 (condivisi)
            ciciot_dir = BASE_DIR / 'data' / 'processed' / 'CICIOT23'
            shared_files = ['test_processed.pkl', 'validation_processed.pkl']
            
            missing_shared = []
            for file in shared_files:
                if not (ciciot_dir / file).exists():
                    missing_shared.append(file)
            
            if missing_shared:
                print_warning(f"Dataset 'smote': Missing shared files in CICIOT23/: {', '.join(missing_shared)}")
                all_ok = False
            else:
                print_success(f"Dataset 'smote': test/val shared from CICIOT23 ✓")
        
        else:
            # Per original verifica train, test, validation
            required_files = ['train_processed.pkl', 'test_processed.pkl', 'validation_processed.pkl']
            
            missing_files = []
            for file in required_files:
                file_path = data_dir / file
                if not file_path.exists():
                    missing_files.append(file)
            
            if missing_files:
                print_warning(f"Dataset '{dataset}': Missing files: {', '.join(missing_files)}")
                all_ok = False
            else:
                print_success(f"Dataset '{dataset}': All files found ✓")
        
        # Verifica artifacts (comuni a entrambi)
        artifact_files = ['label_encoder.pkl', 'scaler.pkl']
        missing_artifacts = []
        for file in artifact_files:
            if not (data_dir / file).exists():
                missing_artifacts.append(file)
        
        if missing_artifacts:
            print_warning(f"Dataset '{dataset}': Missing artifacts: {', '.join(missing_artifacts)}")
            all_ok = False
    
    print()
    return all_ok


# =============================================================================
# TRAINING EXECUTION
# =============================================================================

def train_model(algorithm: str, dataset: str, verbose: bool = True) -> Dict:
    """
    Train a single model with REAL-TIME output streaming.
    
    Returns:
        Dict with results: {success, duration, log_file, error}
    """
    algo_info = ALGORITHMS[algorithm]
    script_path = SRC_DIR / algo_info['script']
    
    # Verifica che lo script esista
    if not script_path.exists():
        error_msg = f"Training script not found: {script_path}"
        print_error(error_msg)
        return {
            'success': False,
            'duration': 0,
            'log_file': None,
            'error': error_msg
        }
    
    # Create log directory
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = LOGS_DIR / f"{algorithm}_{dataset}_{timestamp}.log"
    
    if verbose:
        print_section(f"{algo_info['emoji']} Training: {algo_info['name']} on {dataset} dataset")
        print(f"Script: {script_path.name}")
        print(f"Command: python3 -u {script_path.name} --dataset-type {dataset}") # Nota il -u qui nel log
        print(f"Log: {log_file}")
        print()
        print(f"{COLORS['CYAN']}{'─'*80}{COLORS['END']}")
        print(f"{COLORS['BOLD']}LIVE OUTPUT:{COLORS['END']}")
        print(f"{COLORS['CYAN']}{'─'*80}{COLORS['END']}")
    
    # Execute training with REAL-TIME output
    start_time = time.time()
    
    try:
        # Open log file for writing
        with open(log_file, 'w') as log:
            
            # --- MODIFICA 1: Forziamo l'ambiente non bufferizzato ---
            my_env = os.environ.copy()
            my_env["PYTHONUNBUFFERED"] = "1"
            
            # --- MODIFICA 2: Aggiungiamo '-u' al comando ---
            # sys.executable è il path all'interprete python corrente
            # '-u' forza stdin, stdout e stderr a essere completamente non bufferizzati
            cmd = [sys.executable, '-u', str(script_path.resolve()), '--dataset-type', dataset]

            # Start subprocess with stdout/stderr streaming
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True, # Interpreta output come testo
                bufsize=1,  # Line buffered lato ricezione
                cwd=str(BASE_DIR),
                env=my_env  # Passiamo l'ambiente modificato
            )
            
            # Stream output in real-time
            # iter(process.stdout.readline, '') è più robusto per il realtime rispetto al for loop standard in alcuni casi
            for line in iter(process.stdout.readline, ''):
                if not line:
                    break
                    
                # Write to log
                log.write(line)
                log.flush() # Importante: flush immediato sul file di log
                
                # Print to console (with color coding for important lines)
                line_stripped = line.rstrip()
                
                if line_stripped:
                    # Color code important lines
                    if any(keyword in line_stripped.lower() for keyword in ['error', 'failed', 'exception']):
                        print(f"{COLORS['RED']}{line_stripped}{COLORS['END']}")
                    elif any(keyword in line_stripped.lower() for keyword in ['warning', 'warn']):
                        print(f"{COLORS['YELLOW']}{line_stripped}{COLORS['END']}")
                    elif any(keyword in line_stripped.lower() for keyword in ['success', 'complete', 'saved', '✅', 'accuracy']):
                        print(f"{COLORS['GREEN']}{line_stripped}{COLORS['END']}")
                    elif 'tree depth' in line_stripped.lower() or 'leaves' in line_stripped.lower():
                        print(f"{COLORS['BLUE']}{line_stripped}{COLORS['END']}")
                    elif '=' in line_stripped or '─' in line_stripped or '━' in line_stripped:
                        print(f"{COLORS['CYAN']}{line_stripped}{COLORS['END']}")
                    else:
                        # Normal output
                        print(line_stripped)
                
                # Importante: forza il flush anche su stdout della console
                sys.stdout.flush()
            
            # Wait for process to complete
            process.wait()
            return_code = process.returncode
        
        duration = int(time.time() - start_time)
        
        if verbose:
            print(f"{COLORS['CYAN']}{'─'*80}{COLORS['END']}")
        
        if return_code == 0:
            if verbose:
                print_success(f"{algo_info['name']} ({dataset}) completed in {format_time(duration)}")
                print()
            
            return {
                'success': True,
                'duration': duration,
                'log_file': str(log_file),
                'error': None
            }
        else:
            if verbose:
                print_error(f"{algo_info['name']} ({dataset}) failed with exit code {return_code}")
                print_info(f"Full log: {log_file}")
                print()
            
            return {
                'success': False,
                'duration': duration,
                'log_file': str(log_file),
                'error': f"Exit code: {return_code}"
            }
    
    except KeyboardInterrupt:
        duration = int(time.time() - start_time)
        if verbose:
            print()
            print_warning(f"{algo_info['name']} ({dataset}) interrupted by user")
            print()
        
        # Try to terminate gracefully
        try:
            process.terminate()
            process.wait(timeout=5)
        except:
            process.kill()
        
        return {
            'success': False,
            'duration': duration,
            'log_file': str(log_file),
            'error': "Interrupted by user"
        }
    
    except Exception as e:
        duration = int(time.time() - start_time)
        if verbose:
            print_error(f"{algo_info['name']} ({dataset}) error: {e}")
            print()
        
        return {
            'success': False,
            'duration': duration,
            'log_file': str(log_file) if log_file else None,
            'error': str(e)
        }
def train_sequential(algorithms: List[str], datasets: List[str]) -> Dict:
    """Train models sequentially."""
    results = {}
    total = len(algorithms) * len(datasets)
    current = 0
    
    print_header("STARTING SEQUENTIAL TRAINING")
    print(f"Total models to train: {total}\n")
    
    for algorithm in algorithms:
        for dataset in datasets:
            current += 1
            
            print(f"{COLORS['BLUE']}{'━'*80}{COLORS['END']}")
            print(f"{COLORS['BOLD']}Model {current} of {total}{COLORS['END']}")
            print(f"{COLORS['BLUE']}{'━'*80}{COLORS['END']}")
            print()
            
            result = train_model(algorithm, dataset)
            results[f"{algorithm}_{dataset}"] = result
            
            # Progress
            success_count = sum(1 for r in results.values() if r['success'])
            failed_count = len(results) - success_count
            
            print(f"{COLORS['CYAN']}Progress: {current}/{total} completed "
                  f"({success_count} success, {failed_count} failed){COLORS['END']}")
            print()
    
    return results


def train_parallel(algorithms: List[str], datasets: List[str]) -> Dict:
    """Train models in parallel (experimental)."""
    print_warning("Parallel training not implemented yet. Using sequential.")
    return train_sequential(algorithms, datasets)


# =============================================================================
# SUMMARY & REPORTING
# =============================================================================

def print_summary(results: Dict, algorithms: List[str], datasets: List[str], 
                 total_duration: int):
    """Print comprehensive training summary."""
    print_header("✅ TRAINING COMPLETE")
    
    # Calculate statistics
    total_models = len(results)
    success_count = sum(1 for r in results.values() if r['success'])
    failed_count = total_models - success_count
    
    # Header
    print(f"{COLORS['CYAN']}{'═'*80}{COLORS['END']}")
    print(f"{COLORS['BOLD']}SUMMARY{COLORS['END']}")
    print(f"{COLORS['CYAN']}{'═'*80}{COLORS['END']}\n")
    
    print(f"Total models: {total_models}")
    print(f"Successful: {COLORS['GREEN']}{success_count}{COLORS['END']}")
    print(f"Failed: {COLORS['RED']}{failed_count}{COLORS['END']}")
    print(f"Total time: {COLORS['BOLD']}{format_time(total_duration)}{COLORS['END']}")
    print()
    
    # Per-model results
    if total_models > 0:
        print(f"{COLORS['CYAN']}Individual Model Results:{COLORS['END']}")
        print("─" * 80)
        print(f"{'Algorithm':<20} {'Dataset':<15} {'Status':<12} {'Time':<15} {'Log'}")
        print("─" * 80)
        
        for algorithm in algorithms:
            for dataset in datasets:
                key = f"{algorithm}_{dataset}"
                if key in results:
                    result = results[key]
                    algo_info = ALGORITHMS[algorithm]
                    
                    # Status
                    if result['success']:
                        status = f"{COLORS['GREEN']}✅ Success{COLORS['END']}"
                        time_str = format_time(result['duration'])
                    else:
                        status = f"{COLORS['RED']}❌ Failed{COLORS['END']}"
                        time_str = f"{format_time(result['duration'])} (failed)"
                    
                    # Log file (abbreviated)
                    log_path = Path(result['log_file'])
                    log_name = log_path.name
                    
                    print(f"{algorithm:<20} {dataset:<15} {status:<12} "
                          f"{time_str:<15} {log_name}")
        
        print("─" * 80)
    
    print()
    
    # Status message
    if failed_count == 0:
        print_success("All models trained successfully! 🎉")
    else:
        print_warning(f"{failed_count} model(s) failed. Check logs in: {LOGS_DIR}")
    
    print()
    
    # Next steps
    print(f"{COLORS['CYAN']}{'═'*80}{COLORS['END']}")
    print(f"{COLORS['BOLD']}NEXT STEPS{COLORS['END']}")
    print(f"{COLORS['CYAN']}{'═'*80}{COLORS['END']}\n")
    
    print("1. Review individual model outputs:")
    print(f"   {COLORS['GREEN']}ls -lh docs/<Algorithm>/<dataset>/{COLORS['END']}\n")
    
    print("2. Compare all trained models:")
    print(f"   {COLORS['GREEN']}python3 src/compare_models.py{COLORS['END']}\n")
    
    print("3. View training logs:")
    print(f"   {COLORS['GREEN']}ls -lh {LOGS_DIR}{COLORS['END']}\n")
    
    print("4. Check model files:")
    print(f"   {COLORS['GREEN']}ls -lh models/*/*.pkl{COLORS['END']}\n")


def save_results_json(results: Dict, output_file: Path):
    """Save results to JSON for later analysis."""
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'results': results,
        'summary': {
            'total': len(results),
            'success': sum(1 for r in results.values() if r['success']),
            'failed': sum(1 for r in results.values() if not r['success'])
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print_info(f"Results saved to: {output_file}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Automated training suite for NIDS models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all models on all datasets (8 models total)
  python src/train_all_models.py
  
  # Train specific algorithm(s)
  python src/train_all_models.py --algorithms DecisionTree RandomForest
  
  # Train on specific dataset(s)
  python src/train_all_models.py --datasets smote
  
  # Train specific combination
  python src/train_all_models.py --algorithms RandomForest --datasets original smote
  
  # Dry run (show what would be trained without actually training)
  python src/train_all_models.py --dry-run
  
  # Parallel execution (experimental, faster but uses more resources)
  python src/train_all_models.py --parallel

Available Algorithms:
  DecisionTree    - Fast, interpretable (2-5 min/dataset)
  RandomForest    - Best accuracy, slower (15-30 min/dataset)
  kNN             - Memory-intensive (5-10 min/dataset)
  SVM             - Very slow, not recommended for large datasets (45-120 min/dataset)

Available Datasets:
  original        - Natural distribution (imbalanced)
  smote           - SMOTE-balanced distribution
        """
    )
    
    parser.add_argument(
        '--algorithms',
        type=str,
        nargs='*',
        choices=list(ALGORITHMS.keys()),
        default=list(ALGORITHMS.keys()),
        help='Algorithms to train (default: all)'
    )
    
    parser.add_argument(
        '--datasets',
        type=str,
        nargs='*',
        choices=DATASETS,
        default=DATASETS,
        help='Datasets to use (default: all)'
    )
    
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Run trainings in parallel (experimental, faster but uses more RAM)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be trained without actually training'
    )
    
    parser.add_argument(
        '--save-results',
        type=str,
        default=None,
        help='Save results to JSON file (default: logs/training/results_<timestamp>.json)'
    )
    
    parser.add_argument(
        '--skip-checks',
        action='store_true',
        help='Skip pre-flight checks (not recommended)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    algorithms, datasets = validate_arguments(args.algorithms, args.datasets)
    
    # Header
    print_header("🚀 AUTOMATED TRAINING SUITE - NIDS CICIoT2023")
    
    # Configuration summary
    print(f"{COLORS['CYAN']}Configuration:{COLORS['END']}")
    print(f"  Algorithms: {', '.join([ALGORITHMS[a]['name'] for a in algorithms])}")
    print(f"  Datasets: {', '.join(datasets)}")
    print(f"  Total models: {len(algorithms) * len(datasets)}")
    print(f"  Execution mode: {'Parallel' if args.parallel else 'Sequential'}")
    print(f"  Dry run: {'Yes' if args.dry_run else 'No'}")
    print()
    
    # Dry run
    if args.dry_run:
        print_section("DRY RUN - Models that would be trained:")
        for i, (algo, dataset) in enumerate(
            [(a, d) for a in algorithms for d in datasets], 1
        ):
            algo_info = ALGORITHMS[algo]
            print(f"{i}. {algo_info['emoji']} {algo_info['name']} on {dataset} dataset")
        print()
        print_info("Dry run complete. No models were trained.")
        return
    
    # Pre-flight checks
    if not args.skip_checks:
        if not check_prerequisites(algorithms, datasets):
            print_error("Pre-flight checks failed!")
            print_info("Fix issues above or use --skip-checks to proceed anyway")
            sys.exit(1)
        print_success("Pre-flight checks passed!")
        print()
    
    # Confirmation
    try:
        response = input(f"{COLORS['YELLOW']}Start training {len(algorithms) * len(datasets)} models? [Y/n]: {COLORS['END']}")
        if response.lower() not in ['', 'y', 'yes']:
            print_info("Training cancelled.")
            sys.exit(0)
    except KeyboardInterrupt:
        print("\n")
        print_info("Training cancelled.")
        sys.exit(0)
    
    # Training
    start_time = time.time()
    
    if args.parallel:
        results = train_parallel(algorithms, datasets)
    else:
        results = train_sequential(algorithms, datasets)
    
    total_duration = int(time.time() - start_time)
    
    # Summary
    print_summary(results, algorithms, datasets, total_duration)
    
    # Save results
    if args.save_results or True:  # Always save
        if args.save_results:
            output_file = Path(args.save_results)
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = LOGS_DIR / f'results_{timestamp}.json'
        
        save_results_json(results, output_file)
    
    # Exit code
    failed_count = sum(1 for r in results.values() if not r['success'])
    sys.exit(0 if failed_count == 0 else 1)


if __name__ == '__main__':
    main()