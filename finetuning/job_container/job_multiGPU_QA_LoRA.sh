#!/bin/bash -e
#SBATCH --job-name=ft-gervasio-8B-lora-4gpu
#SBATCH --account=nn9970k
#SBATCH --time=00:30:00
#SBATCH --partition=accel
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH -o ./out/%x-%j.out
#SBATCH --mem-per-gpu=96G

echo "--Node: $(hostname)"
echo

export PYTHONNOUSERSITE=1

# --- Variables and Paths (HOST-SIDE) ---
PROJECT_DIR="/cluster/work/projects/nn9970k"
MyWD="$PROJECT_DIR/$USER/llm-hpc-gervasio"
CONTAINER_DIR="${MyWD}/apptainer"
APPTAINER_SIF="${CONTAINER_DIR}/pytorch_25.08_cuda13.0_arm_custom.sif"

# Configs and python files for fine-tuning
export CONFIG_FILE="${MyWD}/configs/8B_lora_multi_device.yaml"
export PYTHON_FILE="${MyWD}/recipes/distributed/lora_finetune_distributed.py"

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

# Set up variables to control distributed PyTorch training
export MASTER_ADDR=$(hostname)
export MASTER_PORT=25900
export WORLD_SIZE=$SLURM_NPROCS
export LOCAL_WORLD_SIZE=$SLURM_GPUS_PER_NODE
export USE_FLASH_ATTENTION=1

echo "--SLURM_PROCID: $SLURM_PROCID"
echo

# --- Locale Settings ---
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

# --- Define the training task function ---
run_training() {
    # These resolve dynamically on each task/GPU node during srun
    export RANK=$SLURM_PROCID
    export LOCAL_RANK=$SLURM_LOCALID

    echo "Task ${SLURM_PROCID}: RANK=${RANK}, LOCAL_RANK=${LOCAL_RANK}, WORLD_SIZE=${WORLD_SIZE}, LOCAL_WORLD_SIZE=${LOCAL_WORLD_SIZE}"
    echo "LOCAL_RANK: ${LOCAL_RANK}, CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

    python "${PYTHON_FILE}" --config "${CONFIG_FILE}"
}

# Export the function so srun tasks can see it
export -f run_training

# CPU affinity
CPU_BIND="map_cpu:1,73,145,217"
# --- Execute with Apptainer ---
# Bind host project directory to /workspace inside container
# --nv enables NVIDIA GPU support

time srun --cpu-bind=${CPU_BIND} apptainer exec --nv \
      -B "${MyWD}:/workspace" \
      -B $PROJECT_DIR \
      "${APPTAINER_SIF}" \
      bash -c run_training

echo
echo "--- Finished :) ---"
