#!/bin/bash -e
#SBATCH --job-name=ft-gervasio-8B-lora-1gpu_profiling
#SBATCH --account=nn9970k
#SBATCH --time=00:30:00
#SBATCH --partition=accel
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -o ./out/%x-%j.out
#SBATCH --mem-per-gpu=96G

echo "--Node: $(hostname)"
echo

export PYTHONNOUSERSITE=1

# --- Variables and Paths (HOST-SIDE) ---
PROJECT_DIR="/cluster/projects/nn9970k"
MyWD="$PROJECT_DIR/$USER/llm-hpc-gervasio"
CONTAINER_DIR="${MyWD}/apptainer"
APPTAINER_SIF="${CONTAINER_DIR}/pytorch_25.08_cuda13.0_arm_custom.sif"

# Configs and python files for fine-tuning
export CONFIG_FILE="${MyWD}/configs/8B_lora_single_device_profiling.yaml"
export PYTHON_FILE="${MyWD}/recipes/single_device/lora_finetune_single_device.py"

echo "--- My Main Directory (host): ${MyWD}"
echo

echo "=== Configuration ==="
echo "CONFIG_FILE: ${CONFIG_FILE}"
echo "PYTHON_FILE: ${PYTHON_FILE}"
echo

# --- Slurm setting
N=$SLURM_JOB_NUM_NODES
nproc_perN=$SLURM_NTASKS_PER_NODE
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "--nbr of nodes: $N"
echo "--nbr of GPUs: $nproc_perN"
echo

echo "--SLURM_PROCID: $SLURM_PROCID"
echo

# --- Locale Settings ---
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

echo "--- Launching the application inside Apptainer ---"

# --- Execute with Apptainer ---
# Bind host project directory to /workspace inside container
# --nv enables NVIDIA GPU support
time srun apptainer exec --nv \
      -B "${MyWD}:/workspace" \
       -B $PROJECT_DIR \
      "${APPTAINER_SIF}" \
      python "${PYTHON_FILE}" --config "${CONFIG_FILE}"

echo
echo "--- Finished :) ---"
