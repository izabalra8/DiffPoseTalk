#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_step() {
    echo -e "${BLUE}==>${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Header
echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║              DiffPoseTalk Setup Script                    ║"
echo "║   Speech-Driven Stylistic 3D Facial Animation System     ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

# ============================================================
# Step 1: Check Prerequisites and Install UV
# ============================================================
print_step "Checking prerequisites..."

# Check for UV, install if not found
if command -v uv &> /dev/null; then
    UV_VERSION=$(uv --version | head -n1)
    print_success "UV found: $UV_VERSION"
else
    print_warning "UV not found. Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Add UV to PATH for this session
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

    # Source shell profile if available to get updated PATH
    [ -f "$HOME/.bashrc" ] && source "$HOME/.bashrc" 2>/dev/null || true
    [ -f "$HOME/.profile" ] && source "$HOME/.profile" 2>/dev/null || true

    if command -v uv &> /dev/null; then
        UV_VERSION=$(uv --version | head -n1)
        print_success "UV installed: $UV_VERSION"
    else
        print_error "Failed to install UV. Please install manually: curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
fi

# UV will manage Python 3.8 automatically via pyproject.toml requires-python
print_success "UV will automatically install Python 3.8 if needed"

# Check CUDA
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)
    print_success "NVIDIA GPU found (Driver: $CUDA_VERSION)"
else
    print_warning "NVIDIA GPU not detected. DiffPoseTalk requires CUDA for PyTorch."
fi

# Check for ffmpeg (required for video processing)
if command -v ffmpeg &> /dev/null; then
    FFMPEG_VERSION=$(ffmpeg -version 2>&1 | head -n1)
    print_success "ffmpeg found: $FFMPEG_VERSION"
else
    print_warning "ffmpeg not found. Installing ffmpeg..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get update -qq && sudo apt-get install -y -qq ffmpeg
    elif command -v yum &> /dev/null; then
        sudo yum install -y ffmpeg
    elif command -v dnf &> /dev/null; then
        sudo dnf install -y ffmpeg
    elif command -v brew &> /dev/null; then
        brew install ffmpeg
    else
        print_error "Could not install ffmpeg automatically. Please install manually."
        exit 1
    fi

    if command -v ffmpeg &> /dev/null; then
        print_success "ffmpeg installed successfully"
    else
        print_error "Failed to install ffmpeg. Please install manually."
        exit 1
    fi
fi

echo ""

# ============================================================
# Step 2: Install Boost Libraries (required for psbody-mesh)
# ============================================================
print_step "Checking Boost libraries (required for psbody-mesh)..."

# Check if Boost is installed
if dpkg -l | grep -q libboost-dev 2>/dev/null || brew list boost &>/dev/null 2>&1; then
    print_success "Boost libraries already installed"
else
    print_warning "Boost libraries not found. Installing..."
    
    if command -v apt-get &> /dev/null; then
        # Debian/Ubuntu
        sudo apt-get update -qq
        sudo apt-get install -y -qq libboost-dev
    elif command -v yum &> /dev/null; then
        # RedHat/CentOS
        sudo yum install -y boost-devel
    elif command -v dnf &> /dev/null; then
        # Fedora
        sudo dnf install -y boost-devel
    elif command -v brew &> /dev/null; then
        # macOS
        brew install boost
    else
        print_error "Could not install Boost automatically."
        print_error "Please install manually:"
        print_error "  Ubuntu/Debian: sudo apt-get install libboost-dev"
        print_error "  macOS: brew install boost"
        exit 1
    fi
    
    if dpkg -l | grep -q libboost-dev 2>/dev/null || brew list boost &>/dev/null 2>&1; then
        print_success "Boost libraries installed successfully"
    else
        print_error "Failed to install Boost. Please install manually."
        exit 1
    fi
fi

# Set BOOST_INCLUDE_DIRS for psbody-mesh compilation
print_step "Detecting Boost include directory..."
BOOST_INCLUDE_DIRS=""

if [ -d "/usr/include/boost" ]; then
    BOOST_INCLUDE_DIRS="/usr/include"
    print_success "Boost includes found at /usr/include"
elif [ -d "/opt/homebrew/include/boost" ]; then
    BOOST_INCLUDE_DIRS="/opt/homebrew/include"
    print_success "Boost includes found at /opt/homebrew/include"
elif [ -d "/usr/local/include/boost" ]; then
    BOOST_INCLUDE_DIRS="/usr/local/include"
    print_success "Boost includes found at /usr/local/include"
else
    print_warning "Could not auto-detect Boost include directory"
    print_warning "psbody-mesh compilation may fail"
fi

# Export for the build process
if [ -n "$BOOST_INCLUDE_DIRS" ]; then
    export BOOST_INCLUDE_DIRS
    print_success "BOOST_INCLUDE_DIRS exported: $BOOST_INCLUDE_DIRS"
fi

echo ""

# ============================================================
# Step 3: Install Python Dependencies
# ============================================================
print_step "Installing Python dependencies with UV (this may take several minutes)..."
print_step "This will automatically install Python 3.8 if needed..."

# uv sync handles everything: venv creation, dependency resolution, and building
# from-source packages (psbody-mesh from GitHub).
uv sync

print_success "Dependencies installed with UV"

echo ""

# ============================================================
# Step 4: Download Model Checkpoints
# ============================================================
print_step "Checking model checkpoints..."

# Check if model checkpoints exist
CHECKPOINTS_EXIST=false
if [ -f "experiments/DPT/head-SA-hubert-WM/checkpoints/iter_0110000.pt" ] || \
   [ -f "experiments/DPT/SA-hubert-WM/checkpoints/iter_0100000.pt" ]; then
    print_success "Model checkpoints found"
    CHECKPOINTS_EXIST=true
else
    print_warning "Model checkpoints not found in experiments/ directory"
    print_warning "Please download them manually following the instructions in README.md"
fi

# Check FLAME model data
if [ -f "models/data/FLAME2020/generic_model.pkl" ]; then
    print_success "FLAME model data found"
else
    print_warning "FLAME model data not found"
    print_warning "Please download FLAME 2020 from https://flame.is.tue.mpg.de/"
    print_warning "Extract to: models/data/FLAME2020/"
fi

echo ""

# ============================================================
# Step 5: Verify Installation
# ============================================================
print_step "Verifying installation..."

# Check if key directories exist
REQUIRED_DIRS=(
    "models/data"
    "demo/input/audio"
    "demo/input/coef"
    "demo/input/style"
)

ALL_GOOD=true
for dir in "${REQUIRED_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        print_success "$dir/"
    else
        print_error "$dir/ is missing"
        ALL_GOOD=false
    fi
done

# Test Python imports
echo ""
print_step "Testing Python imports..."

uv run python -c "
import sys
try:
    import torch
    print(f'  PyTorch: {torch.__version__}')
    print(f'  CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'  CUDA device: {torch.cuda.get_device_name(0)}')
except ImportError as e:
    print(f'  Error importing torch: {e}')
    sys.exit(1)

try:
    import numpy
    print(f'  NumPy: {numpy.__version__}')
except ImportError as e:
    print(f'  Error importing numpy: {e}')
    sys.exit(1)

try:
    import transformers
    print(f'  Transformers: {transformers.__version__}')
except ImportError as e:
    print(f'  Error importing transformers: {e}')
    sys.exit(1)

try:
    import librosa
    print(f'  Librosa: {librosa.__version__}')
except ImportError as e:
    print(f'  Error importing librosa: {e}')
    sys.exit(1)

try:
    import trimesh
    print(f'  Trimesh: {trimesh.__version__}')
except ImportError as e:
    print(f'  Error importing trimesh: {e}')
    sys.exit(1)

try:
    from psbody.mesh import Mesh
    print(f'  psbody-mesh: Available')
except ImportError as e:
    print(f'  Warning: psbody-mesh not available: {e}')

print('  All core imports successful!')
" && IMPORTS_OK=true || IMPORTS_OK=false

echo ""

# ============================================================
# Final Summary
# ============================================================
if [ "$ALL_GOOD" = true ] && [ "$IMPORTS_OK" = true ]; then
    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║              Setup completed successfully!                ║"
    echo "╚═══════════════════════════════════════════════════════════╝"
    echo ""
    echo "IMPORTANT: Add UV to your PATH for future sessions:"
    echo "  export PATH=\"\$HOME/.local/bin:\$HOME/.cargo/bin:\$PATH\""
    echo ""
    echo "Or add this line to your ~/.bashrc or ~/.profile:"
    echo "  echo 'export PATH=\"\$HOME/.local/bin:\$HOME/.cargo/bin:\$PATH\"' >> ~/.bashrc"
    echo ""
    
    if [ "$CHECKPOINTS_EXIST" = false ]; then
        echo "⚠  NEXT STEPS:"
        echo "  1. Download model checkpoints (see README.md or experiments/pretrained_models/README.txt)"
        echo "  2. Download FLAME 2020 model data from https://flame.is.tue.mpg.de/"
        echo ""
    fi
    
    echo "To run the demo:"
    echo ""
    echo "  # Generate talking head video (with head motion)"
    echo "  uv run python demo.py --exp_name head-SA-hubert-WM --iter 110000 \\"
    echo "    -a demo/input/audio/FAST.flac \\"
    echo "    -c demo/input/coef/TH217.npy \\"
    echo "    -s demo/input/style/head-L4H4-T0.1-BS32/iter_0026000/TH217.npy \\"
    echo "    -o output.mp4"
    echo ""
    echo "  # Generate talking head video (without head motion)"
    echo "  uv run python demo.py --exp_name SA-hubert-WM --iter 100000 \\"
    echo "    -a demo/input/audio/FAST.flac \\"
    echo "    -c demo/input/coef/TH217.npy \\"
    echo "    -s demo/input/style/L4H4-T0.1-BS32/iter_0034000/normal.npy \\"
    echo "    -o output.mp4"
    echo ""
    echo "Training commands:"
    echo "  uv run python main_se.py    # Train style encoder"
    echo "  uv run python main_dpt.py   # Train DiffPoseTalk model"
    echo ""
else
    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║           Setup completed with warnings/errors            ║"
    echo "╚═══════════════════════════════════════════════════════════╝"
    echo ""
    echo "Please review the messages above and resolve any issues."
    echo ""
    echo "Common issues:"
    echo "  - Missing model checkpoints: Download from project resources"
    echo "  - Missing FLAME data: Register and download from https://flame.is.tue.mpg.de/"
    echo "  - CUDA not available: Install NVIDIA drivers and CUDA toolkit"
    echo "  - Boost not installed: Required for psbody-mesh compilation"
    echo ""
    exit 1
fi
