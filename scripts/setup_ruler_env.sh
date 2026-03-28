#!/bin/bash
# =============================================================================
# OmniServe RULER Experiment Environment Setup Script
# =============================================================================
# Run this script after `git clone` on a new server to set up everything
# needed for RULER evaluation with Qwen3-8B-128k + LServe.
#
# Prerequisites:
#   - NVIDIA GPU (A100 or similar, compute capability >= 8.0)
#   - CUDA 12.x drivers installed
#   - Anaconda/Miniconda installed (assumed at ~/anaconda3 or ~/miniconda3)
#   - Internet access for downloading packages
#   - Base model at ~/models/Qwen3-8B (downloaded from HuggingFace)
#
# Usage:
#   cd ~/Lserve_base
#   bash scripts/setup_ruler_env.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OMNISERVE_DIR="$(dirname "$SCRIPT_DIR")"
QWEN3_BASE_DIR="$HOME/models/Qwen3-8B"
QWEN3_128K_DIR="$HOME/models/Qwen3-8B-128k"
cd "${OMNISERVE_DIR}"

echo "============================================="
echo " OmniServe RULER Environment Setup"
echo "============================================="
echo "OmniServe directory: ${OMNISERVE_DIR}"
echo ""

# ---------------------------------------------------------
# Step 0: Check base model exists
# ---------------------------------------------------------
if [ ! -d "$QWEN3_BASE_DIR" ] || [ -z "$(ls -A "$QWEN3_BASE_DIR" 2>/dev/null)" ]; then
    echo "[ERROR] Base model not found at: $QWEN3_BASE_DIR"
    echo "  Download it first:"
    echo "    huggingface-cli download Qwen/Qwen3-8B --local-dir ~/models/Qwen3-8B"
    exit 1
fi
echo "Base model found: $QWEN3_BASE_DIR"

# Create Qwen3-8B-128k (symlinks + custom config.json) if not exists
if [ ! -d "$QWEN3_128K_DIR" ]; then
    echo "Creating Qwen3-8B-128k (symlinks to Qwen3-8B + 128k config)..."
    mkdir -p "$QWEN3_128K_DIR"
    # Symlink all files from base model
    for f in "$QWEN3_BASE_DIR"/*; do
        ln -s "$f" "$QWEN3_128K_DIR/$(basename "$f")"
    done
    # Replace config.json with 128k variant (YaRN rope scaling, 131072 max_position_embeddings)
    rm -f "$QWEN3_128K_DIR/config.json"
    cp "${OMNISERVE_DIR}/configs/Qwen3-8B-128k-config.json" "$QWEN3_128K_DIR/config.json"
    echo "  -> Created: $QWEN3_128K_DIR"
else
    echo "128k model found: $QWEN3_128K_DIR"
fi
echo ""

# ---------------------------------------------------------
# Step 0: Detect conda
# ---------------------------------------------------------
if [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
elif [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
else
    echo "[ERROR] conda not found. Install Anaconda or Miniconda first."
    exit 1
fi

# ---------------------------------------------------------
# Step 1: Create conda environment
# ---------------------------------------------------------
echo "[1/8] Creating conda environment 'omniserve' (Python 3.10)..."
if conda env list | grep -q "omniserve"; then
    echo "  -> Environment 'omniserve' already exists, skipping creation."
else
    conda create -n omniserve python=3.10 -y
fi
conda activate omniserve

# Install CUDA toolkit via conda (provides nvcc for kernel compilation)
echo "  -> Installing CUDA toolkit via conda..."
conda install -c nvidia cuda-toolkit -y

# ---------------------------------------------------------
# Step 2: Install OmniServe + Python dependencies
# ---------------------------------------------------------
echo ""
echo "[2/8] Installing OmniServe package and dependencies..."
pip install --upgrade pip
pip install -e .

# ---------------------------------------------------------
# Step 3: Install FlashAttention
# ---------------------------------------------------------
echo ""
echo "[3/8] Installing FlashAttention..."
pip install flash-attn --no-build-isolation

# Verify installation
python -c "import flash_attn; print(f'  -> flash-attn {flash_attn.__version__} OK')" || {
    echo "[WARN] flash-attn import failed. Try downloading a pre-built wheel from:"
    echo "       https://github.com/Dao-AILab/flash-attention/releases/tag/v2.5.8"
    echo "       Match your PyTorch version (2.2.0) and CUDA version."
    echo "       Try both cxx11abiTRUE and cxx11abiFALSE variants if one fails."
}

# ---------------------------------------------------------
# Step 4: Install Block-Sparse-Attention
# ---------------------------------------------------------
echo ""
echo "[4/8] Installing Block-Sparse-Attention..."

# Option A: Try pre-built wheel (recommended)
# Download the matching wheel from: https://github.com/mit-han-lab/Block-Sparse-Attention/releases
# Then: pip install block_sparse_attn-*.whl
#
# Option B: Build from source
if ! python -c "import block_sparse_attn" 2>/dev/null; then
    echo "  -> Building Block-Sparse-Attention from source..."
    git clone https://github.com/mit-han-lab/Block-Sparse-Attention.git --recursive
    cd Block-Sparse-Attention
    pip install packaging ninja
    python setup.py install
    cd "${OMNISERVE_DIR}"
else
    echo "  -> block_sparse_attn already installed, skipping."
fi

python -c "import block_sparse_attn; print(f'  -> block_sparse_attn {block_sparse_attn.__version__} OK')" || {
    echo "[WARN] block_sparse_attn import failed."
    echo "       Try a pre-built wheel from: https://github.com/mit-han-lab/Block-Sparse-Attention/releases"
}

# ---------------------------------------------------------
# Step 5: Compile OmniServe CUDA kernels
# ---------------------------------------------------------
echo ""
echo "[5/8] Compiling OmniServe CUDA kernels..."
cd kernels
python setup.py install
cd "${OMNISERVE_DIR}"

# ---------------------------------------------------------
# Step 6: Download base model + Quantize with DeepCompressor
# ---------------------------------------------------------
echo ""
echo "[6/8] Preparing Qwen3-8B-128k quantized model..."

MODEL_DIR="${OMNISERVE_DIR}/models/Qwen3-8B-128k-w8a8-per-channel-kv-per-tensor"
mkdir -p "${OMNISERVE_DIR}/models"
mkdir -p "${OMNISERVE_DIR}/logs"

if [ -d "$MODEL_DIR" ] && [ "$(ls -A "$MODEL_DIR" 2>/dev/null)" ]; then
    echo "  -> Quantized model already exists at: $MODEL_DIR"
else
    echo "  -> Base model: $QWEN3_128K_DIR"

    # Install DeepCompressor
    if ! python -c "import deepcompressor" 2>/dev/null; then
        echo "  -> Installing DeepCompressor..."
        pip install deepcompressor
    else
        echo "  -> DeepCompressor already installed."
    fi

    # Quantize model (W8A8 per-channel, KV per-tensor)
    echo "  -> Running W8A8 quantization (this may take a while)..."
    bash scripts/quantize_qwen3.sh "$QWEN3_128K_DIR"
fi

# ---------------------------------------------------------
# Step 7: Install RULER data generation dependencies
# ---------------------------------------------------------
echo ""
echo "[7/8] Installing RULER data generation dependencies..."
pip install wonderwords nltk scipy tqdm pyyaml tenacity html2text beautifulsoup4
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

# Download RULER source data (QA datasets + Paul Graham essays) if not present
RULER_JSON_DIR="${OMNISERVE_DIR}/data/ruler/synthetic/json"
if [ ! -f "${RULER_JSON_DIR}/squad.json" ] || [ ! -f "${RULER_JSON_DIR}/hotpotqa.json" ]; then
    echo "  -> Downloading QA datasets (squad.json, hotpotqa.json)..."
    cd "${RULER_JSON_DIR}"
    bash download_qa_dataset.sh
    cd "${OMNISERVE_DIR}"
fi
if [ ! -f "${RULER_JSON_DIR}/PaulGrahamEssays.json" ]; then
    echo "  -> Downloading Paul Graham essays..."
    cd "${RULER_JSON_DIR}"
    python download_paulgraham_essay.py
    cd "${OMNISERVE_DIR}"
fi

# ---------------------------------------------------------
# Step 8: Generate RULER evaluation data
# ---------------------------------------------------------
echo ""
echo "[8/8] Generating RULER evaluation data..."

RULER_DATA_DIR="${OMNISERVE_DIR}/data/ruler"

if [ -d "${RULER_DATA_DIR}/data/qwen3/102400" ] && [ "$(ls -A "${RULER_DATA_DIR}/data/qwen3/102400" 2>/dev/null)" ]; then
    echo "  -> RULER data already exists."
else
    echo "  -> Generating RULER data (96 samples x 10 tasks, this may take a while)..."
    cd "${RULER_DATA_DIR}"
    bash create_dataset.sh "$QWEN3_128K_DIR" qwen3
    cd "${OMNISERVE_DIR}"
fi

# ---------------------------------------------------------
# Done
# ---------------------------------------------------------
echo ""
echo "============================================="
echo " Setup Complete!"
echo "============================================="
echo ""
echo "Verification checklist:"
echo "  conda activate omniserve"
echo "  python -c 'import flash_attn; print(\"flash-attn OK\")'          "
echo "  python -c 'import block_sparse_attn; print(\"block-sparse OK\")' "
echo "  python -c 'import omniserve; print(\"omniserve OK\")'            "
echo ""
echo "If all steps succeeded, you're ready to go."
echo "If any step was skipped, check the messages above."
echo ""
echo "Run RULER evaluation:"
echo "  bash scripts/run_ruler_lserve.sh"
echo ""
