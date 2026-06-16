# Python Recipes for LLM Workloads

This directory contains standalone Python scripts (recipes) used for fine‑tuning and inference throughout the course.

## Directory Structure

| Folder | Description |
| :--- | :--- |
| `single_device/` | Training scripts optimized for a single GPU. |
| `distributed/` | Training scripts using PyTorch Distributed/FSDP for multi‑GPU scaling. |
| `inference/` | Scripts for deploying models with vLLM for batch or distributed inference. |

---

## Core Scripts

### 1. Fine-tuning
- `recipes/single_device/lora_finetune_single_device.py`: The entry point for single‑GPU LoRA/QLoRA training.
- `recipes/distributed/lora_finetune_distributed.py`: Scale your training across multiple GPUs using FSDP.

### 2. Inference
- `recipes/inference/vllm_inference.py`: Run high‑throughput inference on a single or multiple GPUs using vLLM.
- `recipes/inference/vllm_distributed.py`: Specifically for distributed vLLM serving.

---

## Usage Patterns

These scripts are designed to be flexible and are mostly driven by configuration files in the `configs/` directory.

### Basic Training Command
```bash
python recipes/single_device/lora_finetune_single_device.py \
    --config configs/lora/llama3_2_1B_lora_single_device_XSum.yaml
```

### Basic Inference Command
```bash
python recipes/inference/vllm_inference.py \
    --model_id path/to/your/finetuned/model \
    --prompt "What is HPC?"
```

---

## Technical Features

- **PEFT Integration**: Built‑in support for LoRA and QLoRA via the Hugging Face `peft` library.
- **FSDP**: Distributed scripts use PyTorch's Fully Sharded Data Parallel (FSDP) for efficient model sharding.
- **vLLM Integration**: Inference recipes leverage PagedAttention for high‑throughput serving.
