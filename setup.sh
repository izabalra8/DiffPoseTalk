#!/usr/bin/env bash
# DiffPoseTalk setup script (UV-based)
# Integrates dependency installation and FLAME data download.

set -e

# ─── Colour helpers ───────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Colour

info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }
step()    { echo -e "\n${BOLD}${BLUE}══ $* ══${NC}"; }

# ─── Banner ───────────────────────────────────────────────────────────────────
echo -e "${BOLD}${CYAN}"
cat << 'EOF'
  ____  _  __  __  ____                _____     _ _
 |  _ \(_)/ _|/ _||  _ \ ___  ___  __|_   _|_ _| | | __
 | | | | | |_| |_ | |_) / _ \/ __|/ _ \| |/ _` | | |/ /
 | |_| | |  _|  _||  __/ (_) \__ \  __/| | (_| | |   <
 |____/|_|_| |_|  |_|   \___/|___/\___||_|\__,_|_|_|\_\

  Speech-Driven 3D Facial Animation — Setup Script (UV)
EOF
echo -e "${NC}"

# ─── Resolve project root ─────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ─── URL encoder (for FLAME credentials) ─────────────────────────────────────
urle() {
    [[ "${1}" ]] || return 1
    local LANG=C i x
    for (( i = 0; i < ${#1}; i++ )); do
        x="${1:i:1}"
        [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"
    done
    echo
}

###############################################################################
# STEP 1 — Check Prerequisites
###############################################################################
step "Step 1 — Check Prerequisites"

# ── UV ────────────────────────────────────────────────────────────────────────
if command -v uv &>/dev/null; then
    success "uv $(uv --version 2>&1 | head -1) already installed"
else
    info "Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add uv to PATH for the remainder of this script
    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
    if command -v uv &>/dev/null; then
        success "uv installed: $(uv --version 2>&1 | head -1)"
    else
        error "uv installation failed. Please install uv manually: https://docs.astral.sh/uv/"
        exit 1
    fi
fi

# ── GPU ───────────────────────────────────────────────────────────────────────
if command -v nvidia-smi &>/dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    success "GPU detected: ${GPU_INFO:-unknown}"
else
    warn "nvidia-smi not found — GPU unavailable or drivers not installed"
    warn "PyTorch will install without CUDA support"
fi

# ── wget ──────────────────────────────────────────────────────────────────────
if command -v wget &>/dev/null; then
    success "wget available"
else
    error "wget is required for data downloads but was not found."
    error "Install it with:  sudo apt-get install wget  (Debian/Ubuntu)"
    exit 1
fi

# ── unzip ─────────────────────────────────────────────────────────────────────
if command -v unzip &>/dev/null; then
    success "unzip available"
else
    error "unzip is required for extracting downloaded archives but was not found."
    error "Install it with:  sudo apt-get install unzip  (Debian/Ubuntu)"
    exit 1
fi

# ── ffmpeg ────────────────────────────────────────────────────────────────────
if command -v ffmpeg &>/dev/null; then
    success "ffmpeg available"
else
    warn "ffmpeg not found — video export (utils/media.py) will not work at runtime"
    warn "Install it with:  sudo apt-get install ffmpeg  (Debian/Ubuntu)"
fi

###############################################################################
# STEP 2 — Sync Virtual Environment
###############################################################################
step "Step 2 — Sync Virtual Environment"

# uv sync reads pyproject.toml + uv.lock and brings .venv/ into exact
# agreement with the lockfile.  It creates the venv (Python 3.8) if absent.
# chumpy's legacy build-system deps (pip, setuptools) are injected cleanly
# via [tool.uv.extra-build-dependencies] in pyproject.toml — no workarounds.
export UV_LINK_MODE=copy   # cache is on a different filesystem; avoid warning
info "Syncing dependencies from uv.lock (this may take a few minutes on first run)..."
uv sync
PYTHON=".venv/bin/python"
success "Python dependencies installed"

###############################################################################
# STEP 3 — Download FLAME Data
###############################################################################
step "Step 3 — Download FLAME Model Data"

mkdir -p ./models/data

FLAME2020_ZIP="./models/data/FLAME2020.zip"
FLAME2020_DIR="./models/data/FLAME2020"
FLAME_MASKS_ZIP="./models/data/FLAME_masks.zip"
FLAME_MASKS_PKL="./models/data/FLAME_masks.pkl"
LANDMARK_NPY="./models/data/landmark_embedding.npy"

NEED_FLAME_AUTH=false
[ ! -f "$FLAME2020_ZIP" ] && [ ! -d "$FLAME2020_DIR" ] && NEED_FLAME_AUTH=true
[ ! -f "$FLAME_MASKS_PKL" ] && NEED_FLAME_AUTH=true

if $NEED_FLAME_AUTH; then
    echo
    warn "FLAME model files require registration at https://flame.is.tue.mpg.de/"
    warn "Please register and accept the license terms before continuing."
    echo
    read -rp "  FLAME username: " FLAME_USER
    read -rsp "  FLAME password: " FLAME_PASS
    echo
    ENC_USER=$(urle "$FLAME_USER")
    ENC_PASS=$(urle "$FLAME_PASS")
fi

# ── FLAME2020 ─────────────────────────────────────────────────────────────────
if [ -d "$FLAME2020_DIR" ] && [ -n "$(ls -A "$FLAME2020_DIR" 2>/dev/null)" ]; then
    success "FLAME2020 directory already present — skipping download"
else
    if [ ! -f "$FLAME2020_ZIP" ]; then
        info "Downloading FLAME2020.zip..."
        wget --post-data "username=${ENC_USER}&password=${ENC_PASS}" \
            'https://download.is.tue.mpg.de/download.php?domain=flame&sfile=FLAME2020.zip&resume=1' \
            -O "$FLAME2020_ZIP" --no-check-certificate --continue
    else
        info "FLAME2020.zip already downloaded — skipping download"
    fi

    if [ -f "$FLAME2020_ZIP" ]; then
        info "Extracting FLAME2020.zip..."
        unzip -q "$FLAME2020_ZIP" -d "$FLAME2020_DIR"
        success "FLAME2020 extracted to $FLAME2020_DIR"
    else
        error "FLAME2020.zip download failed. Check your credentials and try again."
        exit 1
    fi
fi

# ── FLAME_masks ───────────────────────────────────────────────────────────────
if [ -f "$FLAME_MASKS_PKL" ]; then
    success "FLAME_masks.pkl already present — skipping download"
else
    FLAME_MASKS_DIR="./models/data/FLAME_masks"
    if [ ! -f "$FLAME_MASKS_ZIP" ]; then
        info "Downloading FLAME_masks.zip..."
        wget --no-check-certificate \
            --http-user="${ENC_USER}" --http-password="${ENC_PASS}" \
            'https://files.is.tue.mpg.de/tbolkart/FLAME/FLAME_masks.zip' \
            -O "$FLAME_MASKS_ZIP"
    else
        info "FLAME_masks.zip already downloaded — skipping download"
    fi

    if [ -f "$FLAME_MASKS_ZIP" ]; then
        info "Extracting FLAME_masks.zip..."
        unzip -q "$FLAME_MASKS_ZIP" -d "$FLAME_MASKS_DIR"
        mv "$FLAME_MASKS_DIR/FLAME_masks.pkl" ./models/data/
        success "FLAME_masks.pkl extracted to models/data/"
    else
        error "FLAME_masks.zip download failed. Check your credentials and try again."
        exit 1
    fi
fi

# ── landmark_embedding.npy ────────────────────────────────────────────────────
if [ -f "$LANDMARK_NPY" ]; then
    success "landmark_embedding.npy already present — skipping download"
else
    info "Downloading landmark_embedding.npy from DECA GitHub..."
    wget 'https://github.com/yfeng95/DECA/raw/master/data/landmark_embedding.npy' \
        -O "$LANDMARK_NPY"
    success "landmark_embedding.npy downloaded"
fi

###############################################################################
# STEP 4 — Verify Installation
###############################################################################
step "Step 4 — Verify Installation"

VERIFY_ERRORS=0

# ── Model data files ──────────────────────────────────────────────────────────
check_file() {
    if [ -e "$1" ]; then
        success "Found: $1"
    else
        error "Missing: $1"
        VERIFY_ERRORS=$(( VERIFY_ERRORS + 1 ))
    fi
}

check_file "$FLAME_MASKS_PKL"
check_file "$LANDMARK_NPY"

if [ -d "$FLAME2020_DIR" ] && [ -n "$(ls -A "$FLAME2020_DIR" 2>/dev/null)" ]; then
    success "Found: $FLAME2020_DIR (non-empty)"
else
    error "Missing or empty: $FLAME2020_DIR"
    VERIFY_ERRORS=$(( VERIFY_ERRORS + 1 ))
fi

# ── Python imports ────────────────────────────────────────────────────────────
info "Testing Python imports..."

run_import_check() {
    local label="$1"
    local code="$2"
    if "$PYTHON" -c "$code" 2>/dev/null; then
        success "$label"
    else
        error "$label import failed"
        VERIFY_ERRORS=$(( VERIFY_ERRORS + 1 ))
    fi
}

run_import_check "torch" \
    "import torch; print(f'  torch {torch.__version__}, CUDA={torch.cuda.is_available()}')"

run_import_check "transformers" \
    "import transformers; print(f'  transformers {transformers.__version__}')"

run_import_check "librosa" \
    "import librosa; print(f'  librosa {librosa.__version__}')"

run_import_check "pyrender" \
    "import pyrender; print('  pyrender OK')"

run_import_check "trimesh" \
    "import trimesh; print('  trimesh OK')"

###############################################################################
# STEP 5 — Final Summary
###############################################################################
step "Step 5 — Summary"

if [ "$VERIFY_ERRORS" -eq 0 ]; then
    echo -e "${GREEN}${BOLD}"
    echo "  ✓  DiffPoseTalk setup complete!"
    echo -e "${NC}"
    echo -e "${BOLD}Usage:${NC}"
    echo "  source .venv/bin/activate"
    echo
    echo "  # Extract style from a motion sequence:"
    echo "  python extract_style.py --exp_name <SE_NAME> --iter <SE_ITER> \\"
    echo "      -c <MOTION.npz> -o <OUTPUT_NAME> -s <START_FRAME>"
    echo
    echo "  # Generate animation:"
    echo "  python demo.py --exp_name <DPT_NAME> --iter <DPT_ITER> \\"
    echo "      -a <AUDIO> -c <SHAPE.npy> -s <STYLE.npy> -o out.mp4"
    echo
    echo "  # Training — Stage 1 (Style Encoder):"
    echo "  python main_se.py --exp_name <NAME> --data_root <LMDB_PATH>"
    echo
    echo "  # Training — Stage 2 (Denoising Network):"
    echo "  python main_dpt.py --exp_name <NAME> --data_root <LMDB_PATH> \\"
    echo "      --use_indicator --scheduler Warmup --audio_model hubert \\"
    echo "      --style_enc_ckpt <PATH_TO_SE_CKPT>"
else
    echo -e "${YELLOW}${BOLD}"
    echo "  ⚠  Setup completed with ${VERIFY_ERRORS} warning(s) — see [ERROR] messages above."
    echo -e "${NC}"
    echo "  Some components may be missing. Review the errors and re-run if needed."
fi
