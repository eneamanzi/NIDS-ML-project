#!/bin/bash

################################################################################
# train_all.sh - Automated Training Script per NIDS CICIoT2023
#
# Esegue training di tutti gli algoritmi su dataset original e/o SMOTE
# con supporto parametrico completo.
#
# Usage:
#   ./scripts/train_all.sh                    # Train tutti su tutti i dataset
#   ./scripts/train_all.sh --datasets smote   # Solo SMOTE
#   ./scripts/train_all.sh --algorithms DecisionTree RandomForest  # Solo DT e RF
#   ./scripts/train_all.sh --algorithms RandomForest --datasets smote  # RF solo su SMOTE
################################################################################

set -e  # Exit on error

# =============================================================================
# COLORS
# =============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================
ALGORITHMS=("DecisionTree" "RandomForest" "kNN" "SVM")
DATASETS=("original" "smote")
PARALLEL=false
SKIP_EXISTING=false

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC_DIR="$PROJECT_ROOT/src"

# =============================================================================
# FUNCTIONS
# =============================================================================

print_header() {
    echo -e "${BLUE}"
    echo "================================================================================"
    echo "$1" | awk '{printf "%*s\n", (80+length)/2, $0}'
    echo "================================================================================"
    echo -e "${NC}"
}

print_section() {
    echo -e "${CYAN}"
    echo "--------------------------------------------------------------------------------"
    echo "$1"
    echo "--------------------------------------------------------------------------------"
    echo -e "${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "${PURPLE}ℹ️  $1${NC}"
}

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Automated training script for NIDS algorithms on CICIoT2023 dataset.

OPTIONS:
    --algorithms ALGO [ALGO...]    Specify algorithms to train
                                   Valid: DecisionTree, RandomForest, kNN, SVM
                                   Default: all
                                   
    --datasets DATASET [DATASET...]  Specify datasets to use
                                     Valid: original, smote
                                     Default: both
                                     
    --parallel                     Run trainings in parallel (experimental)
    --skip-existing               Skip if model file already exists
    -h, --help                    Show this help message

EXAMPLES:
    # Train all algorithms on all datasets (default)
    $0
    
    # Train only Decision Tree and Random Forest
    $0 --algorithms DecisionTree RandomForest
    
    # Train all algorithms only on SMOTE dataset
    $0 --datasets smote
    
    # Train Random Forest on both datasets
    $0 --algorithms RandomForest --datasets original smote
    
    # Train with parallel execution (faster but uses more resources)
    $0 --parallel

EOF
    exit 0
}

check_prerequisites() {
    print_section "Checking Prerequisites"
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python3 not found. Please install Python 3.9+"
        exit 1
    fi
    print_success "Python3 found: $(python3 --version)"
    
    # Check scripts
    local scripts=("train_decision_tree.py" "train_random_forest.py" "train_knn.py" "train_svm.py")
    for script in "${scripts[@]}"; do
        if [ ! -f "$SRC_DIR/$script" ]; then
            print_error "Script not found: $SRC_DIR/$script"
            exit 1
        fi
    done
    print_success "All training scripts found"
    
    # Check data
    for dataset in "${DATASETS[@]}"; do
        if [ "$dataset" == "original" ]; then
            data_dir="$PROJECT_ROOT/data/processed/CICIOT23"
        else
            data_dir="$PROJECT_ROOT/data/processed/SMOTE"
        fi
        
        if [ ! -d "$data_dir" ]; then
            print_warning "Dataset directory not found: $data_dir"
            print_info "Run preprocessing first:"
            if [ "$dataset" == "original" ]; then
                echo "  python src/data_processing.py --output-dir data/processed/CICIOT23"
            else
                echo "  python src/apply_smote.py --input-dir data/processed/CICIOT23 --output-dir data/processed/SMOTE"
            fi
        else
            print_success "Dataset found: $dataset"
        fi
    done
}

model_exists() {
    local algo=$1
    local dataset=$2
    
    case $algo in
        "DecisionTree")
            model_file="$PROJECT_ROOT/models/DecisionTree/dt_model_${dataset}.pkl"
            ;;
        "RandomForest")
            model_file="$PROJECT_ROOT/models/RandomForest/rf_model_${dataset}.pkl"
            ;;
        "kNN")
            model_file="$PROJECT_ROOT/models/kNN/knn_model_${dataset}.pkl"
            ;;
        "SVM")
            model_file="$PROJECT_ROOT/models/SVM/svm_model_${dataset}.pkl"
            ;;
    esac
    
    [ -f "$model_file" ]
}

train_model() {
    local algo=$1
    local dataset=$2
    
    # Check if should skip
    if [ "$SKIP_EXISTING" = true ] && model_exists "$algo" "$dataset"; then
        print_info "Skipping $algo ($dataset) - model already exists"
        return 0
    fi
    
    print_section "Training: $algo on $dataset dataset"
    
    local script=""
    local extra_args=""
    
    case $algo in
        "DecisionTree")
            script="train_decision_tree.py"
            extra_args="--max-depth 20"
            ;;
        "RandomForest")
            script="train_random_forest.py"
            extra_args="--n-estimators 100 --max-depth 25"
            ;;
        "kNN")
            script="train_knn.py"
            extra_args="--n-neighbors 5"
            ;;
        "SVM")
            script="train_svm.py"
            extra_args="--kernel rbf --C 1.0"
            ;;
        *)
            print_error "Unknown algorithm: $algo"
            return 1
            ;;
    esac
    
    local cmd="python3 $SRC_DIR/$script --dataset-type $dataset $extra_args"
    
    echo -e "${CYAN}Command: $cmd${NC}"
    echo ""
    
    local start_time=$(date +%s)
    
    if $cmd; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        print_success "$algo ($dataset) completed in ${duration}s"
        echo ""
        return 0
    else
        print_error "$algo ($dataset) failed!"
        echo ""
        return 1
    fi
}

train_all_sequential() {
    print_header "SEQUENTIAL TRAINING"
    
    local total=$((${#ALGORITHMS[@]} * ${#DATASETS[@]}))
    local current=0
    local failed=0
    
    print_info "Total trainings to run: $total"
    echo ""
    
    for algo in "${ALGORITHMS[@]}"; do
        for dataset in "${DATASETS[@]}"; do
            current=$((current + 1))
            
            echo -e "${PURPLE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
            echo -e "${PURPLE}Training $current of $total${NC}"
            echo -e "${PURPLE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
            echo ""
            
            if ! train_model "$algo" "$dataset"; then
                failed=$((failed + 1))
            fi
        done
    done
    
    return $failed
}

train_all_parallel() {
    print_header "PARALLEL TRAINING (Experimental)"
    
    print_warning "Parallel mode uses more CPU/memory. Monitor system resources!"
    echo ""
    
    local pids=()
    local log_dir="$PROJECT_ROOT/logs/training"
    mkdir -p "$log_dir"
    
    for algo in "${ALGORITHMS[@]}"; do
        for dataset in "${DATASETS[@]}"; do
            local log_file="$log_dir/${algo}_${dataset}.log"
            
            print_info "Starting $algo ($dataset) in background..."
            train_model "$algo" "$dataset" > "$log_file" 2>&1 &
            pids+=($!)
        done
    done
    
    print_info "All trainings started. Waiting for completion..."
    echo ""
    
    local failed=0
    for pid in "${pids[@]}"; do
        if ! wait $pid; then
            failed=$((failed + 1))
        fi
    done
    
    print_info "Logs saved in: $log_dir"
    
    return $failed
}

# =============================================================================
# ARGUMENT PARSING
# =============================================================================

SELECTED_ALGORITHMS=()
SELECTED_DATASETS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --algorithms)
            shift
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                SELECTED_ALGORITHMS+=("$1")
                shift
            done
            ;;
        --datasets)
            shift
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                SELECTED_DATASETS+=("$1")
                shift
            done
            ;;
        --parallel)
            PARALLEL=true
            shift
            ;;
        --skip-existing)
            SKIP_EXISTING=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            print_error "Unknown option: $1"
            usage
            ;;
    esac
done

# Use defaults if not specified
if [ ${#SELECTED_ALGORITHMS[@]} -eq 0 ]; then
    ALGORITHMS=("DecisionTree" "RandomForest" "kNN" "SVM")
else
    ALGORITHMS=("${SELECTED_ALGORITHMS[@]}")
fi

if [ ${#SELECTED_DATASETS[@]} -eq 0 ]; then
    DATASETS=("original" "smote")
else
    DATASETS=("${SELECTED_DATASETS[@]}")
fi

# =============================================================================
# MAIN
# =============================================================================

clear

print_header "🚀 AUTOMATED TRAINING - NIDS CICIoT2023"

echo ""
print_info "Configuration:"
echo "  Algorithms: ${ALGORITHMS[*]}"
echo "  Datasets: ${DATASETS[*]}"
echo "  Parallel mode: $PARALLEL"
echo "  Skip existing: $SKIP_EXISTING"
echo "  Project root: $PROJECT_ROOT"
echo ""

# Confirm
read -p "$(echo -e ${YELLOW}Start training? [Y/n]: ${NC})" -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]] && [[ ! -z $REPLY ]]; then
    print_info "Training cancelled."
    exit 0
fi

# Check prerequisites
check_prerequisites

# Start training
START_TIME=$(date +%s)

if [ "$PARALLEL" = true ]; then
    train_all_parallel
    FAILED=$?
else
    train_all_sequential
    FAILED=$?
fi

END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))

# Summary
print_header "✅ TRAINING COMPLETE"

TOTAL=$((${#ALGORITHMS[@]} * ${#DATASETS[@]}))
SUCCESS=$((TOTAL - FAILED))

echo ""
echo "Summary:"
echo "  Total models: $TOTAL"
echo "  Successful: $SUCCESS"
echo "  Failed: $FAILED"
echo "  Total time: ${TOTAL_DURATION}s ($(($TOTAL_DURATION / 60))m)"
echo ""

if [ $FAILED -eq 0 ]; then
    print_success "All trainings completed successfully!"
    echo ""
    print_info "Next steps:"
    echo "  1. Compare models: python src/compare_models.py"
    echo "  2. Review plots in docs/<Algorithm>/<dataset>/"
    echo "  3. Test best model in production"
else
    print_warning "$FAILED training(s) failed. Check logs for details."
fi

exit $FAILED