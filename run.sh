#!/usr/bin/env bash
# =============================================================================
#  run.sh  —  GPU-Accelerated Portfolio Optimization
#  EN.605.617 | Johns Hopkins University | Stefan Mauch | Spring 2026
#
#  Full end-to-end pipeline:
#    1. Environment check (CUDA, g++, Python, packages)
#    2. Directory setup
#    3. Build C++/CUDA binaries (make)
#    4. Download financial data  ← download_data.py (50 ETFs, 2018–2026)
#    5. Real-data benchmark  (44 ETFs × 2,010 days)
#    6. Scalability sweep    (N ∈ {25,50,100,250,500,1000}, synthetic)
#    7. Portfolio analytics  (efficient frontier, backtest)
#    8. Figure generation    (Figs 1–9, 200 dpi PNG)
#    9. Report metrics       (CSV + LaTeX table)
#
#  Usage:
#    chmod +x run.sh
#    ./run.sh                     # full run
#    ./run.sh --skip-download     # reuse cached financial_prices.csv
#    ./run.sh --skip-figures      # benchmarks only, no matplotlib
#    ./run.sh --build-only        # compile and quit
#    ./run.sh --help
# =============================================================================

set -euo pipefail

# ─── Colour codes ─────────────────────────────────────────────────────────────
RESET="\033[0m"
BOLD="\033[1m"
GREEN="\033[1;32m"
YELLOW="\033[1;33m"
CYAN="\033[1;36m"
RED="\033[1;31m"
GRAY="\033[0;37m"

ok()     { echo -e "  ${GREEN}✓${RESET}  $*"; }
warn()   { echo -e "  ${YELLOW}⚠${RESET}  $*"; }
fail()   { echo -e "  ${RED}✗${RESET}  $*" >&2; }
banner() { echo -e "\n${CYAN}${BOLD}══════════════════════════════════════════════════════════════════════════${RESET}"
           echo -e "${CYAN}${BOLD}  $*${RESET}"
           echo -e "${CYAN}${BOLD}══════════════════════════════════════════════════════════════════════════${RESET}"; }
step()   { echo -e "\n${BOLD}── $* ──${RESET}"; }

# ─── Argument parsing ─────────────────────────────────────────────────────────
SKIP_DOWNLOAD=0
SKIP_FIGURES=0
BUILD_ONLY=0

for arg in "$@"; do
    case "$arg" in
        --skip-download) SKIP_DOWNLOAD=1 ;;
        --skip-figures)  SKIP_FIGURES=1  ;;
        --build-only)    BUILD_ONLY=1    ;;
        --help|-h)
            echo ""
            echo "  Usage: ./run.sh [OPTIONS]"
            echo ""
            echo "  Options:"
            echo "    --skip-download   Reuse cached data/financial_prices.csv"
            echo "    --skip-figures    Skip matplotlib figure generation"
            echo "    --build-only      Compile C++/CUDA and exit"
            echo "    --help            Show this message"
            echo ""
            exit 0 ;;
        *) fail "Unknown option: $arg"; exit 1 ;;
    esac
done

# ─── Project root (directory containing this script) ──────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ─── Timestamp ────────────────────────────────────────────────────────────────
START_TIME=$(date +%s)
LOG_FILE="$SCRIPT_DIR/run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

banner "GPU-Accelerated Portfolio Optimization  |  EN.605.617"
echo -e "  ${GRAY}Stefan Mauch · Johns Hopkins University · Spring 2026${RESET}"
echo -e "  ${GRAY}Run log: $LOG_FILE${RESET}"
echo -e "  ${GRAY}Started: $(date)${RESET}"


# =============================================================================
# STEP 0 — ENVIRONMENT CHECKS
# =============================================================================
banner "STEP 0 — Environment checks"

# ── CUDA / nvcc ───────────────────────────────────────────────────────────────
step "CUDA compiler (nvcc)"
if command -v nvcc &>/dev/null; then
    NVCC_VER=$(nvcc --version | grep "release" | awk '{print $6}' | tr -d ',')
    ok "nvcc found: $NVCC_VER"
else
    fail "nvcc not found. Install CUDA Toolkit."
    fail "On Ubuntu: sudo apt install nvidia-cuda-toolkit"
    exit 1
fi

# ── GPU device ────────────────────────────────────────────────────────────────
step "GPU device"
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
    DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    ok "GPU:    $GPU_NAME"
    ok "Memory: $GPU_MEM"
    ok "Driver: $DRIVER"
    nvidia-smi 2>/dev/null | head -12 || true
else
    warn "nvidia-smi not available — GPU details unknown"
fi

# ── C++ compiler ──────────────────────────────────────────────────────────────
step "C++ compiler (g++)"
if command -v g++ &>/dev/null; then
    GCC_VER=$(g++ --version | head -1)
    ok "g++ found: $GCC_VER"
else
    fail "g++ not found. Install: sudo apt install build-essential"
    exit 1
fi

# ── cuBLAS / cuSOLVER libraries ───────────────────────────────────────────────
step "CUDA libraries (cuBLAS, cuSOLVER)"
CUDA_LIB_PATHS=(
    "/usr/local/cuda/lib64"
    "/usr/lib/x86_64-linux-gnu"
    "/usr/lib64"
)
find_lib() {
    for dir in "${CUDA_LIB_PATHS[@]}"; do
        if ls "$dir/lib${1}"*.so* 2>/dev/null | head -1 | grep -q .; then
            echo "$dir"; return 0
        fi
    done
    return 1
}
if find_lib "cublas" &>/dev/null; then
    ok "libcublas found"
else
    fail "libcublas not found. Ensure CUDA toolkit is fully installed."
    exit 1
fi
if find_lib "cusolver" &>/dev/null; then
    ok "libcusolver found"
else
    fail "libcusolver not found. Ensure CUDA toolkit is fully installed."
    exit 1
fi

# ── Python ────────────────────────────────────────────────────────────────────
step "Python environment"
if command -v python3 &>/dev/null; then
    PY_VER=$(python3 --version)
    ok "Python: $PY_VER"
else
    fail "python3 not found. Install: sudo apt install python3"
    exit 1
fi

REQUIRED_PKGS="numpy pandas matplotlib seaborn scipy tabulate yfinance"
MISSING_PKGS=""
for pkg in $REQUIRED_PKGS; do
    if ! python3 -c "import ${pkg//-/_}" 2>/dev/null; then
        MISSING_PKGS="$MISSING_PKGS $pkg"
    fi
done
if [[ -n "$MISSING_PKGS" ]]; then
    warn "Missing Python packages:$MISSING_PKGS"
    echo -e "  Installing…"
    pip3 install --quiet $MISSING_PKGS
    ok "All packages installed"
else
    ok "All required Python packages present"
fi

if python3 -c "import cupy" 2>/dev/null; then
    ok "cupy available (GPU-accelerated NumPy)"
else
    warn "cupy not available (optional — CPU path used for Python analytics)"
fi


# =============================================================================
# STEP 1 — DIRECTORY SETUP
# =============================================================================
banner "STEP 1 — Directory setup"

for dir in include src bin data results figures; do
    mkdir -p "$SCRIPT_DIR/$dir"
done
ok "Project directories: include/ src/ bin/ data/ results/ figures/"


# =============================================================================
# STEP 2 — BUILD
# =============================================================================
banner "STEP 2 — Build C++/CUDA project"

if [[ ! -f "$SCRIPT_DIR/Makefile" ]]; then
    fail "Makefile not found in $SCRIPT_DIR"
    fail "Run the Jupyter notebook to generate all source files first, or"
    fail "copy the source files manually from the notebook output."
    exit 1
fi

REQUIRED_SOURCES=(
    "src/utils.cpp"
    "src/data_loader.cpp"
    "src/cpu_portfolio.cpp"
    "src/benchmark.cpp"
    "src/gpu_portfolio.cu"
    "src/main.cu"
    "src/synthetic_benchmark.cu"
)
REQUIRED_HEADERS=(
    "include/asset_universe.h"
    "include/data_loader.h"
    "include/utils.h"
    "include/cpu_portfolio.h"
    "include/gpu_portfolio.h"
    "include/benchmark.h"
)

step "Checking source files"
ALL_PRESENT=1
for f in "${REQUIRED_SOURCES[@]}" "${REQUIRED_HEADERS[@]}"; do
    if [[ -f "$SCRIPT_DIR/$f" ]]; then
        ok "$f"
    else
        fail "$f — MISSING"
        ALL_PRESENT=0
    fi
done
if [[ $ALL_PRESENT -eq 0 ]]; then
    fail "One or more source files missing."
    fail "Generate them by running the Jupyter notebook (cells in Section 2)."
    exit 1
fi

step "Compiling (make clean && make)"
cd "$SCRIPT_DIR"
make clean 2>/dev/null || true
if make 2>&1; then
    ok "Build successful"
else
    fail "Build FAILED — check compiler errors above"
    exit 1
fi

for exe in bin/portfolio_app bin/synthetic_benchmark; do
    if [[ -x "$SCRIPT_DIR/$exe" ]]; then
        SIZE=$(du -k "$SCRIPT_DIR/$exe" | cut -f1)
        ok "$exe  (${SIZE} KB)"
    else
        fail "$exe not built"
        exit 1
    fi
done

if [[ $BUILD_ONLY -eq 1 ]]; then
    banner "BUILD COMPLETE (--build-only flag set)"
    exit 0
fi


# =============================================================================
# STEP 3 — FINANCIAL DATA DOWNLOAD
# =============================================================================
banner "STEP 3 — Financial data download"

DATA_CSV="$SCRIPT_DIR/data/financial_prices.csv"
DOWNLOAD_PY="$SCRIPT_DIR/download_data.py"

if [[ ! -f "$DOWNLOAD_PY" ]]; then
    fail "download_data.py not found at $DOWNLOAD_PY"
    exit 1
fi

if [[ $SKIP_DOWNLOAD -eq 1 ]]; then
    # ── Reuse cached data ─────────────────────────────────────────────────────
    if [[ -f "$DATA_CSV" ]]; then
        ROWS=$(python3 -c "import pandas as pd; df=pd.read_csv('$DATA_CSV',index_col=0); print(f'{df.shape[1]} tickers × {df.shape[0]} days')")
        ok "--skip-download: reusing cached $DATA_CSV  ($ROWS)"
    else
        warn "--skip-download set but $DATA_CSV not found — downloading anyway"
        python3 "$DOWNLOAD_PY" --output "$DATA_CSV"
        ok "Download complete: $DATA_CSV"
    fi
else
    # ── Fresh download ────────────────────────────────────────────────────────
    step "Running download_data.py"
    python3 "$DOWNLOAD_PY" --output "$DATA_CSV"
    if [[ -f "$DATA_CSV" ]]; then
        SIZE_KB=$(du -k "$DATA_CSV" | cut -f1)
        ok "financial_prices.csv written  (${SIZE_KB} KB)"
    else
        fail "download_data.py finished but $DATA_CSV was not created"
        exit 1
    fi
fi


# =============================================================================
# STEP 4 — PYTHON ANALYTICS PIPELINE
# =============================================================================
banner "STEP 4 — Running full analytics pipeline"

ANALYTICS_PY="$SCRIPT_DIR/analytics.py"
if [[ ! -f "$ANALYTICS_PY" ]]; then
    fail "analytics.py not found at $ANALYTICS_PY"
    exit 1
fi

ANALYTICS_ARGS=""
[[ $SKIP_FIGURES -eq 1 ]] && ANALYTICS_ARGS="$ANALYTICS_ARGS --skip-figures"

python3 "$ANALYTICS_PY" $ANALYTICS_ARGS


# =============================================================================
# FINAL SUMMARY
# =============================================================================
END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
MINUTES=$(( ELAPSED / 60 ))
SECONDS=$(( ELAPSED % 60 ))

banner "ALL STEPS COMPLETE"

echo -e "\n  ${BOLD}Elapsed time: ${MINUTES}m ${SECONDS}s${RESET}"
echo -e "  ${BOLD}Run log:      $LOG_FILE${RESET}"

echo -e "\n  ${BOLD}Output structure:${RESET}"
echo -e "  ${GRAY}portfolio_cuda/${RESET}"
echo -e "  ${GRAY}├── bin/${RESET}"
echo -e "  ${GRAY}│   ├── portfolio_app          (44-ETF real-data benchmark)${RESET}"
echo -e "  ${GRAY}│   └── synthetic_benchmark    (scalability sweep, N + T args)${RESET}"
echo -e "  ${GRAY}├── data/${RESET}"
echo -e "  ${GRAY}│   └── financial_prices.csv   (50 tickers, 2018–2026)${RESET}"
echo -e "  ${GRAY}├── results/${RESET}"
echo -e "  ${GRAY}│   ├── real_benchmark.json${RESET}"
echo -e "  ${GRAY}│   ├── real_benchmark.log${RESET}"
echo -e "  ${GRAY}│   ├── scalability_results.csv${RESET}"
echo -e "  ${GRAY}│   ├── report_metrics.csv${RESET}"
echo -e "  ${GRAY}│   └── benchmark_table.tex${RESET}"
echo -e "  ${GRAY}└── figures/${RESET}"
echo -e "  ${GRAY}    ├── fig1_runtime_comparison.png${RESET}"
echo -e "  ${GRAY}    ├── fig2_speedup_curve.png${RESET}"
echo -e "  ${GRAY}    ├── fig3_portfolio_weights.png${RESET}"
echo -e "  ${GRAY}    ├── fig4_covariance_heatmap.png${RESET}"
echo -e "  ${GRAY}    ├── fig5_efficient_frontier.png${RESET}"
echo -e "  ${GRAY}    ├── fig6_backtest.png${RESET}"
echo -e "  ${GRAY}    ├── fig7_flop_analysis.png${RESET}"
echo -e "  ${GRAY}    ├── fig8_numerical_accuracy.png${RESET}"
echo -e "  ${GRAY}    └── fig9_weights_by_class.png${RESET}"

echo -e "\n  ${BOLD}To re-run individual steps:${RESET}"
echo -e "  ${GRAY}  python3 download_data.py                        # data only${RESET}"
echo -e "  ${GRAY}  python3 download_data.py --start 2020-01-01     # custom range${RESET}"
echo -e "  ${GRAY}  ./bin/portfolio_app data/financial_prices.csv   # C++ benchmark${RESET}"
echo -e "  ${GRAY}  ./bin/synthetic_benchmark 1000 2000             # scalability${RESET}"

echo ""
