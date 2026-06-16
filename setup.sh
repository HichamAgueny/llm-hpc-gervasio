#!/bin/bash

# Exit immediately if any command fails
set -e

# =====================================================================
# 1. Dynamic Configuration & Security
# =====================================================================
echo "----------------------------------------------------------------"
echo " Hugging Face Authentication"
echo "----------------------------------------------------------------"
# Prompt the user for their token securely (-s hides input text)
read -ersp "Enter your Hugging Face Token: " USER_TOKEN
echo "" # Prints a newline after hiding the password input

# Validate that the user actually entered something
if [[ -z "$USER_TOKEN" ]]; then
    echo "--Error: Hugging Face token cannot be empty. Exiting script." >&2
    exit 1
fi

export HF_TOKEN="$USER_TOKEN"

# Unified cluster directory path setup
BASE_DIR="/cluster/projects/nn9970k/$USER/llm-hpc-gervasio"
MODEL_DIR="$BASE_DIR/shared/models/gervasio-8b-portuguese"
DATASET_DIR="$BASE_DIR/shared/datasets/alpaca-data-pt"
CACHE_DIR="$BASE_DIR/.cache/huggingface"

# Define Base Directories
# Source: Original shared location
SOURCE_BASE="/cluster/projects/nn9970k/hicham/apptainer"
TARGET_BASE="$BASE_DIR"

# =====================================================================
# 2. Workspace & Environment Directory Tree Configuration
# =====================================================================
echo -e "\n=== Step 1: Generating local cluster workspace environments ==="
mkdir -p "$BASE_DIR"
mkdir -p "$MODEL_DIR"
mkdir -p "$DATASET_DIR"
mkdir -p "$CACHE_DIR"

# Direct Hugging Face utilities to build out cache pools on your project mount
export HF_HOME="$CACHE_DIR"

# Bypass the uncompiled cluster Xet binary layer to protect against .so crashes
export HF_HUB_DISABLE_XET=1

# =====================================================================
# 3. Automated Application Installation (uv & Hugging Face Hub CLI)
# =====================================================================
echo "=== Step 2: Provisioning Python environment toolchains via 'uv' ==="
curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR=$HOME/.tools/uv sh
source $HOME/.tools/uv/env

echo "=== Step 3: Installing Hugging Face CLI abstraction wrappers ==="
export UV_TOOL_DIR="$HOME/.tools/hf/tools"
export UV_TOOL_BIN_DIR="$HOME/.tools/hf/bin"

uv tool install --force huggingface_hub
export PATH="$UV_TOOL_BIN_DIR:$PATH"

# Make the hf binary path permanent for all future terminal sessions
if ! grep -q "$UV_TOOL_BIN_DIR" ~/.bashrc; then
    echo "-> Making 'hf' path persistent in ~/.bashrc..."
    echo 'export PATH="'"$UV_TOOL_BIN_DIR"':$PATH"' >> ~/.bashrc
fi

# Reload the profile configuration to refresh the environment
# (Using '|| true' because some cluster .bashrc files contain interactive shell guards)
source ~/.bashrc || true

# =====================================================================
# 4. Sequential Asset Downloads
# =====================================================================
echo "=== Step 4: Fetching Gervásio Portuguese language baseline ==="
hf download PORTULAN/gervasio-8b-portuguese-ptpt-decoder --local-dir "$MODEL_DIR"

echo "=== Step 5: Patching missing base Llama 3.1 Tokenizer components ==="
hf download meta-llama/Llama-3.1-8B-Instruct original/tokenizer.model original/params.json --local-dir "$MODEL_DIR"

echo "=== Step 6: Fetching Portuguese Alpaca alignment datasets ==="
hf download --type dataset dominguesm/alpaca-data-pt-br --local-dir "$DATASET_DIR"

echo "====================================================================="
echo "-- Setup 1 Complete! Model weights, tokens, and data are staged at:"
echo "-- $BASE_DIR"
echo "====================================================================="

echo "----------------------------------------------------------------"
echo "Initializing Environment Setup"
echo "Source: $SOURCE_BASE"
echo "Target: $TARGET_BASE"
echo "----------------------------------------------------------------"

# 2. Create Target Base and Apptainer Directories
echo "[1/3] Creating core directories..."
mkdir -p "$TARGET_BASE/apptainer"

echo "[2/3] Copying Apptainer images..."
SIF_FILES=(
    "pytorch_25.08_cuda13.0_arm_custom.sif"
    "vllm0.12_cu131_py3.12_arm_custom.sif"
)

for SIF_FILE in "${SIF_FILES[@]}"; do
    if [ -f "$SOURCE_BASE/apptainer/$SIF_FILE" ]; then
        echo " -> Copying $SIF_FILE..."
        cp "$SOURCE_BASE/apptainer/$SIF_FILE" "$TARGET_BASE/apptainer/"
    else
        echo " !! Warning: Apptainer image not found at $SOURCE_BASE/apptainer/$SIF_FILE"
    fi
done

# 4. Create Results, Logs, and Profiling Paths
echo "[3/3] Creating results, logs, and profiling structures..."
PATHS=(
    "$TARGET_BASE/results/checkpoints_out"
    "$TARGET_BASE/results/logs"
    "$TARGET_BASE/results/profiles"
)

for p in "${PATHS[@]}"; do
    mkdir -p "$p"
done
