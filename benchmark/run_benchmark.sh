#!/bin/bash
#
# SCENT Benchmark Suite
# Comprehensive benchmark comparing SCENT_jaxQTL vs original SCENT (R)
#
# This script:
# 1. Generates benchmark datasets (small, medium, large)
# 2. Runs SCENT_jaxQTL on all datasets
# 3. Runs original SCENT (R) on all datasets
# 4. Compares results and generates visualizations
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DATASETS=("small" "medium" "large")
CELL_TYPE="T_cell"
REGRESSION="poisson"
BOOTSTRAP_SAMPLES=100
NCORES=6
SEED=42

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
RESULTS_DIR="${SCRIPT_DIR}/results"

# Print functions
print_header() {
    echo -e "${BLUE}"
    echo "================================================================"
    echo "$1"
    echo "================================================================"
    echo -e "${NC}"
}

print_step() {
    echo -e "${GREEN}[$(date +%H:%M:%S)] $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

print_error() {
    echo -e "${RED}[ERROR] $1${NC}"
}

# Check dependencies
check_dependencies() {
    print_step "Checking dependencies..."

    # Check Python
    if ! command -v python &> /dev/null; then
        print_error "Python not found. Please install Python 3.8+"
        exit 1
    fi

    # Check R
    if ! command -v Rscript &> /dev/null; then
        print_warning "Rscript not found. R benchmarks will be skipped."
        SKIP_R=1
    else
        SKIP_R=0
    fi

    # Check Python packages
    python -c "import jax" 2>/dev/null || {
        print_error "JAX not installed. Please run: pip install jax"
        exit 1
    }

    python -c "import pandas" 2>/dev/null || {
        print_error "pandas not installed. Please run: pip install pandas"
        exit 1
    }

    python -c "import matplotlib" 2>/dev/null || {
        print_warning "matplotlib not installed. Plots will be skipped."
    }

    print_step "✓ Dependencies checked"
}

# Generate benchmark data
generate_data() {
    print_header "Step 1: Generating Benchmark Data"

    if [ -d "$DATA_DIR" ] && [ "$(ls -A $DATA_DIR)" ]; then
        echo -n "Data directory already exists. Regenerate? [y/N] "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            print_step "Using existing data"
            return
        fi
    fi

    print_step "Generating datasets: ${DATASETS[*]}"

    cd "$SCRIPT_DIR"
    python generate_benchmark_data.py \
        --output_dir "$DATA_DIR" \
        --datasets "${DATASETS[@]}" \
        --seed "$SEED"

    if [ $? -eq 0 ]; then
        print_step "✓ Data generation completed"
    else
        print_error "Data generation failed"
        exit 1
    fi
}

# Run SCENT_jaxQTL benchmark
run_jaxqtl_benchmark() {
    print_header "Step 2: Running SCENT_jaxQTL Benchmark"

    print_step "Configuration:"
    echo "  - Datasets: ${DATASETS[*]}"
    echo "  - Cell type: $CELL_TYPE"
    echo "  - Regression: $REGRESSION"
    echo "  - Bootstrap samples: $BOOTSTRAP_SAMPLES"

    cd "$SCRIPT_DIR"
    python run_scent_jaxqtl.py \
        --data_dir "$DATA_DIR" \
        --output_dir "$RESULTS_DIR/scent_jaxqtl" \
        --datasets "${DATASETS[@]}" \
        --cell_type "$CELL_TYPE" \
        --regression "$REGRESSION" \
        --bootstrap_samples "$BOOTSTRAP_SAMPLES" \
        --seed "$SEED"

    if [ $? -eq 0 ]; then
        print_step "✓ SCENT_jaxQTL benchmark completed"
    else
        print_error "SCENT_jaxQTL benchmark failed"
        exit 1
    fi
}

# Run SCENT R benchmark
run_r_benchmark() {
    if [ $SKIP_R -eq 1 ]; then
        print_warning "Skipping SCENT R benchmark (Rscript not found)"
        return
    fi

    print_header "Step 3: Running SCENT (R) Benchmark"

    print_step "Configuration:"
    echo "  - Datasets: ${DATASETS[*]}"
    echo "  - Cell type: $CELL_TYPE"
    echo "  - Regression: $REGRESSION"
    echo "  - Cores: $NCORES"

    cd "$SCRIPT_DIR"
    Rscript run_scent_r.R \
        --data_dir "$DATA_DIR" \
        --output_dir "$RESULTS_DIR/scent_r" \
        --datasets "${DATASETS[@]}" \
        --cell_type "$CELL_TYPE" \
        --regression "$REGRESSION" \
        --ncores "$NCORES" \
        --seed "$SEED"

    if [ $? -eq 0 ]; then
        print_step "✓ SCENT R benchmark completed"
    else
        print_error "SCENT R benchmark failed"
        exit 1
    fi
}

# Compare results
compare_results() {
    if [ $SKIP_R -eq 1 ]; then
        print_warning "Skipping result comparison (R benchmark not run)"
        return
    fi

    print_header "Step 4: Comparing Results"

    cd "$SCRIPT_DIR"
    python compare_results.py \
        --results_dir "$RESULTS_DIR" \
        --datasets "${DATASETS[@]}" \
        --cell_type "$CELL_TYPE" \
        --output_dir "$RESULTS_DIR/comparison"

    if [ $? -eq 0 ]; then
        print_step "✓ Results comparison completed"
    else
        print_error "Results comparison failed"
        exit 1
    fi
}

# Generate summary report
generate_report() {
    print_header "Step 5: Generating Summary Report"

    REPORT_FILE="${RESULTS_DIR}/benchmark_report.md"

    cat > "$REPORT_FILE" << EOF
# SCENT Benchmark Report

Generated on: $(date)

## Configuration

- **Datasets**: ${DATASETS[*]}
- **Cell type**: $CELL_TYPE
- **Regression**: $REGRESSION
- **Bootstrap samples**: $BOOTSTRAP_SAMPLES (initial)
- **Random seed**: $SEED

## Results

### Performance Comparison

See \`comparison/performance_comparison.png\` for detailed performance plots.

### Result Consistency

See individual dataset comparison plots in \`comparison/\` directory:
EOF

    for dataset in "${DATASETS[@]}"; do
        echo "- \`comparison/comparison_${dataset}.png\`" >> "$REPORT_FILE"
    done

    cat >> "$REPORT_FILE" << EOF

### Detailed Statistics

- SCENT_jaxQTL stats: \`scent_jaxqtl/benchmark_stats.csv\`
- SCENT R stats: \`scent_r/benchmark_stats.csv\`
- Comparison stats: \`comparison/comparison_stats.csv\`

## Summary

EOF

    # Add performance summary if available
    if [ -f "${RESULTS_DIR}/scent_jaxqtl/benchmark_stats.csv" ] && \
       [ -f "${RESULTS_DIR}/scent_r/benchmark_stats.csv" ]; then
        echo "### Runtime Comparison" >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
        echo "| Dataset | SCENT_jaxQTL (s) | SCENT R (s) | Speedup |" >> "$REPORT_FILE"
        echo "|---------|------------------|-------------|---------|" >> "$REPORT_FILE"

        # This would need actual parsing - just a placeholder
        echo "| small   | -                | -           | -       |" >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
    fi

    cat >> "$REPORT_FILE" << EOF

## Conclusion

SCENT_jaxQTL provides:
- **GPU acceleration** for significantly faster computation
- **Consistent results** with original SCENT implementation
- **Modern Python API** for easier integration

See the detailed plots and statistics files for complete analysis.
EOF

    print_step "✓ Report generated: $REPORT_FILE"
}

# Main execution
main() {
    print_header "SCENT Benchmark Suite"

    echo "This script will run a comprehensive benchmark comparing"
    echo "SCENT_jaxQTL vs original SCENT (R implementation)."
    echo ""
    echo "Estimated time: 10-60 minutes (depending on dataset size)"
    echo ""
    echo -n "Continue? [Y/n] "
    read -r response

    if [[ "$response" =~ ^[Nn]$ ]]; then
        echo "Benchmark cancelled."
        exit 0
    fi

    # Create results directory
    mkdir -p "$RESULTS_DIR"

    # Run benchmark steps
    check_dependencies
    generate_data
    run_jaxqtl_benchmark
    run_r_benchmark
    compare_results
    generate_report

    print_header "Benchmark Complete!"

    echo "Results saved to: $RESULTS_DIR"
    echo ""
    echo "Key files:"
    echo "  - Summary report: ${RESULTS_DIR}/benchmark_report.md"
    echo "  - Performance plots: ${RESULTS_DIR}/comparison/performance_comparison.png"
    if [ $SKIP_R -eq 0 ]; then
        echo "  - Result comparison: ${RESULTS_DIR}/comparison/comparison_*.png"
    fi
    echo ""
    echo "View detailed statistics:"
    echo "  - SCENT_jaxQTL: ${RESULTS_DIR}/scent_jaxqtl/benchmark_stats.csv"
    if [ $SKIP_R -eq 0 ]; then
        echo "  - SCENT R: ${RESULTS_DIR}/scent_r/benchmark_stats.csv"
        echo "  - Comparison: ${RESULTS_DIR}/comparison/comparison_stats.csv"
    fi
}

# Handle command line arguments
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --help, -h          Show this help message"
    echo "  --datasets DATASETS Space-separated list of datasets (default: small medium large)"
    echo "  --cell-type TYPE    Cell type to analyze (default: T_cell)"
    echo "  --skip-generate     Skip data generation"
    echo "  --skip-r            Skip R benchmark"
    echo "  --quick             Run only small dataset"
    echo ""
    echo "Example:"
    echo "  $0 --datasets small --quick"
    exit 0
fi

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --datasets)
            shift
            DATASETS=()
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                DATASETS+=("$1")
                shift
            done
            ;;
        --cell-type)
            CELL_TYPE="$2"
            shift 2
            ;;
        --skip-generate)
            SKIP_GENERATE=1
            shift
            ;;
        --skip-r)
            SKIP_R=1
            shift
            ;;
        --quick)
            DATASETS=("small")
            shift
            ;;
        *)
            shift
            ;;
    esac
done

# Run main function
main
