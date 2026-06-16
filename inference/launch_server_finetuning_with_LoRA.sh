#!/bin/bash -e

export NUM_GPUS=$SLURM_GPUS_ON_NODE
echo "--nbr of GPUs: $NUM_GPUS"
echo

export PYTHONNOUSERSITE=1

# -------- User configuration --------
PROJECT_DIR="/cluster/projects/nn9970k"
MyWD="$PROJECT_DIR/$USER/llm-hpc-gervasio"
APPTAINER_SIF="${MyWD}/apptainer/vllm0.12_cu131_py3.12_arm_custom.sif"
CONNECTION_FILE="${MyWD}/inference/connection.env"

# Set paths
export MODEL_PATH=${MODEL_PATH:-"${MyWD}/shared/models/gervasio-8b-portuguese"}
export LORA_PATH=${LORA_PATH:-"${MyWD}/results/checkpoints_out/gervasio-8b_lora_single_device/epoch_0"} # Enable LoRA
export QUANTIZATION=${QUANTIZATION:-"None"}
export MAX_LORA_RANK=${MAX_LORA_RANK:-64}

# -------- LoRA detection --------
if [[ -n "$LORA_PATH" ]]; then
    USE_LORA=true
    MODEL_NAME="custom_lora"
else
    USE_LORA=false
    MODEL_NAME="$MODEL_PATH"
fi

echo "----------------------------------------"
echo "Configuration:"
echo "  Model:        $MODEL_PATH"
echo "  LoRA:         ${LORA_PATH:-disabled}"
echo "  Quantization: $QUANTIZATION"
echo "  GPUs:         $NUM_GPUS"
echo "  Status:       Starting API Server on Port 8000..."
echo "----------------------------------------"

# -------- Validation --------
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "ERROR: Base model not found: $MODEL_PATH"
    exit 1
fi

if [[ "$USE_LORA" == true && ! -d "$LORA_PATH" ]]; then
    echo "ERROR: LoRA checkpoint not found: $LORA_PATH"
    exit 1
fi

if [[ "$USE_LORA" == true && ! -f "$LORA_PATH/adapter_config.json" ]]; then
    echo "ERROR: adapter_config.json not found in $LORA_PATH"
    exit 1
fi

# -------- Cache --------
export VLLM_CACHE_ROOT=$MyWD/.cache/vllm
mkdir -p "$VLLM_CACHE_ROOT"
#export VLLM_LOGGING_LEVEL=ERROR

# -------- Write connection info for chat.sh --------
echo "HOST=http://$(hostname)" > "$CONNECTION_FILE"
echo "PORT=8000"               >> "$CONNECTION_FILE"
echo "MODEL=$MODEL_NAME"       >> "$CONNECTION_FILE"
echo "  Connection info written to: $CONNECTION_FILE"
echo "  Host: http://$(hostname):8000"
echo "  Model name for chat.sh: $MODEL_NAME"

# -------- Build vLLM launch command --------
# https://docs.vllm.ai/en/latest/features/quantization/
#VLLM_CMD="python3 -m vllm.entrypoints.openai.api_server
VLLM_CMD="vllm serve $MODEL_PATH
    --tensor-parallel-size $NUM_GPUS
    --quantization $QUANTIZATION
    --host 0.0.0.0
    --port 8000
    --enable-lora
    --lora-modules custom_lora=$LORA_PATH
    --max-lora-rank $MAX_LORA_RANK"

# -------- Launch API Server --------
export MOUNT_DIR="$PROJECT_DIR/$USER"
export WORKDIR="/workspace"
echo "Launching vLLM server (LoRA: $USE_LORA)..."
apptainer exec --nv \
     -B ${MOUNT_DIR}:${WORKDIR} \
     -B $PROJECT_DIR \
     --pwd ${WORKDIR} \
     "${APPTAINER_SIF}" \
     $VLLM_CMD
