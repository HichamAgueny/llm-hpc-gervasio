#!/bin/bash

# Usage: ./gpu_monitor.sh <JOB_ID>
# To stop the script, press Ctrl+C

JOB_ID="$1"

# Check for input argument
if [ -z "$JOB_ID" ]; then
    echo "Usage: $0 <JobID>"
    exit 1
fi

# 1. Wait for the job to start if it is queued/pending
while true; do
    JOB_STATE=$(squeue -j "$JOB_ID" -h -o "%t" 2>/dev/null)
    
    if [ -z "$JOB_STATE" ]; then
        echo "Job ID $JOB_ID not found in the queue. It may have already finished or the ID is incorrect."
        exit 1
    fi

    if [ "$JOB_STATE" == "PD" ]; then
        clear
        echo "=========================================================================================="
        echo " Job ID $JOB_ID is currently PENDING. Waiting for resources... (Press Ctrl+C to exit)"
        echo "=========================================================================================="
        sleep 5
    elif [[ "$JOB_STATE" == "R" || "$JOB_STATE" == "CG" ]]; then
        break
    else
        echo "Job $JOB_ID is in state $JOB_STATE. Cannot monitor GPUs."
        exit 1
    fi
done

# 2. Extract and expand the NodeList
NODELIST=$(scontrol show job "$JOB_ID" | grep -oP 'NodeList=\K\S+' | grep -v '(null)' | head -n 1)

if [ -z "$NODELIST" ]; then
    echo "Failed to resolve a valid NodeList for Job ID $JOB_ID"
    exit 1
fi

# Expands compressed lists (e.g., node[01-04]) into a flat array
NODES=($(scontrol show hostnames "$NODELIST"))

# 3. Determine EXACT GPU allocation per node using scontrol detail (-d)
# This associative array maps the Hostname -> GPU Indices (e.g., "node01" -> "0,1")
declare -A NODE_GPUS

while read -r line; do
    # Extract node block (e.g., node[01-02]) and IDX string (e.g., 0-1,3) from Slurm
    block_nodes=$(echo "$line" | grep -oP 'Nodes=\K\S+')
    idx_raw=$(echo "$line" | grep -oP 'IDX:\K[0-9,-]+')
    
    if [[ -n "$block_nodes" && -n "$idx_raw" ]]; then
        # Expand hyphenated Slurm ranges into commas (e.g., "0-2,4" -> "0,1,2,4")
        # because nvidia-smi requires comma-separated lists
        idx_expanded=$(echo "$idx_raw" | awk -F',' '{
            res=""
            for(i=1; i<=NF; i++) {
                if ($i ~ /-/) {
                    split($i, a, "-")
                    for(j=a[1]; j<=a[2]; j++) {
                        res = res (res==""?"":",") j
                    }
                } else {
                    res = res (res==""?"":",") $i
                }
            }
            print res
        }')
        
        # Map the expanded GPU indices to each individual node in the block
        for n in $(scontrol show hostnames "$block_nodes"); do
            NODE_GPUS["$n"]="$idx_expanded"
        done
    fi
done < <(scontrol show job -d "$JOB_ID" 2>/dev/null | grep "Nodes=")

echo "Found ${#NODES[@]} nodes allocated to Job $JOB_ID. Starting monitor..."
sleep 2

# 4. Continuous monitoring loop
while true; do
    # Verify job is still active
    JOB_STATE=$(squeue -j "$JOB_ID" -h -o "%t" 2>/dev/null)
    if [[ "$JOB_STATE" != "R" && "$JOB_STATE" != "CG" ]]; then
        echo -e "\n=========================================================================================="
        echo " Job ID $JOB_ID is no longer active (State: ${JOB_STATE:-FINISHED}). Exiting monitor..."
        echo "=========================================================================================="
        exit 0
    fi

    clear
    echo "=========================================================================================="
    echo " Monitoring Job: $JOB_ID | State: $JOB_STATE | Nodes: ${#NODES[@]} | Time: $(date '+%H:%M:%S')"
    echo "=========================================================================================="

    # Loop through each node
    for NODE in "${NODES[@]}"; do
        
        TARGET_GPUS="${NODE_GPUS[$NODE]}"
        
        # If we successfully parsed the specific GPUs, format the -i flag for nvidia-smi
        if [ -n "$TARGET_GPUS" ]; then
            echo -e "\n>> Node: $NODE (Allocated GPU IDs: $TARGET_GPUS)"
            GPU_FILTER="-i $TARGET_GPUS"
        else
            # Fallback if Slurm configuration does not report IDX
            echo -e "\n>> Node: $NODE (Showing ALL GPUs)"
            GPU_FILTER=""
        fi
        
        # SSH quietly in BatchMode, apply the GPU filter
        ssh -q -o StrictHostKeyChecking=no -o BatchMode=yes "$NODE" "\
            nvidia-smi $GPU_FILTER --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
            --format=csv,noheader,nounits" 2>/dev/null | \
        awk -F', ' '{ 
            printf "  [GPU %s] %-20.20s | Util: %3s%% | Mem: %6s / %6s MiB | Temp: %3sC | Pwr: %3sW\n", 
            $1, $2, $3, $4, $5, $6, $7 
        }'
    done

    # Refresh interval
    sleep 2
done
