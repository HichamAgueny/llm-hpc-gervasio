# Utility Scripts

This directory contains utility scripts for monitoring and managing LLM jobs on the HPC cluster.

## Scripts

### 1. GPU Monitor (`gpu_monitor.sh`)
Monitoring GPU utilization is crucial for optimizing your fine‑tuning and inference jobs. This script provides real‑time feedback on GPU metrics for a specific Slurm job.

**Usage:**
```bash
./gpu.monitor.sh <JobID>
```

**What it does:**
- Hooks into the specified Slurm job.
- Displays GPU utilization, memory usage, and temperature.
- Helps identify if your model is bottlenecked by communication or compute.

---

### 2. Python Backend (`gpu_monitor.py`)
The underlying Python logic used by the shell script to query and format GPU metrics.

---

## Integration with Labs
In most labs (e.g., Day 1 fine‑tuning), you are encouraged to run the monitor in a separate terminal or background process to watch your job's performance.
