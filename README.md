
# llm-hpc-gervasio

An optimized, end-to-end framework for instruction fine-tuning and serving the **Gervásio 8B European Portuguese (PT-PT)** model within High-Performance Computing (HPC) cluster environments.

This repository streamlines workspace generation, toolchain installation, LoRA training via TorchTune, and high-throughput inference via vLLM using Apptainer containers.

---

## Overview

* **Base Model:** [PORTULAN/gervasio-8b-portuguese-ptpt-decoder](https://huggingface.co/PORTULAN/gervasio-8b-portuguese-ptpt-decoder) (Decoder-only model aligned for PT-PT).
* **Fine-Tuning Approach:** Parameter-efficient Parameter Tuning via Low-Rank Adaptation (LoRA) optimized for single-device or distributed node clusters.
* **Target Dataset:** [dominguesm/alpaca-data-pt-br](https://huggingface.co/datasets/dominguesm/alpaca-data-pt-br) (Alpaca formatting mapped directly inside binary Parquet pipelines).
* **Serving Engine:** [vLLM](https://github.com/vllm-project/vllm) for runtime on-the-fly LoRA adapter merging and OpenAI-compatible API serving.

---

## Repository Structure

```text
llm-hpc-gervasio/
├── apptainer/               # Apptainer (.sif) custom container images
├── shared/
│   ├── datasets/            # Training datasets (stored in .parquet format)
│   └── models/              # Base model weights and Llama 3.1 fallback tokens
├── results/
│   ├── checkpoints_out/     # Saved LoRA adapters per epoch
│   ├── logs/                # Training performance logs
│   └── profiles/            # PyTorch profiling structures
└── setup.sh    # Main automated bootstrap script

```

---

## Environment Prerequisites

This framework is built to operate seamlessly on cluster nodes without requiring root privileges or complex system-level installs.

* **Package Management:** Managed via [uv](https://github.com/astral-sh/uv) (lightning-fast Python toolchain bundler).
* **Containerization:** Running containerized pipelines via **Apptainer/Singularity**.
* **HPC Stack:** Optimized for CUDA 13+ environments.

---

## Quick Start: Setup & Data Ingestion

To automatically provision the workspace directories, fetch application dependencies, download the model files, and patch the missing base Llama 3.1 tokenizer files, follow these steps:

### 1. Execute the Bootstrapper

Run the automated setup script directly on your login or interactive compute node:

```bash
mkdir -p /cluster/projects/nn9970k/$USER
git clone https://github.com/HichamAgueny/llm-hpc-gervasio.git
cd llm-hpc-gervasio
chmod +x setup.sh
./setup.sh

```

### 2. Provide Credentials

When prompted, paste your Hugging Face User Access Token securely:

```text
----------------------------------------------------------------
 Hugging Face Authentication
----------------------------------------------------------------
Enter your Hugging Face Token: ****************************

```

> **Note:** The script automatically flags `HF_HUB_DISABLE_XET=1` to bypass cluster compatibility crashes with Xet binary layers, falling back to standard, ultra-stable HTTPS streams.

---

## Fine-Tuning Pipeline (TorchTune)

Training is managed via `torchtune`, leveraging memory-efficient datasets directly from `.parquet` configurations.

To run fine-tuning on a single GPU:
```bash
cd llm-hpc-gervasio/finetuning/job_container
sbatch job_singleGPU_QA_LoRA.sh 
```

To run fine-tuning on multiple GPUs:
```bash
cd llm-hpc-gervasio/finetuning/job_container
sbatch job_multiGPU_QA_LoRA.sh
```

---

## Inference & Serving (vLLM)

Inference is optimized by serving the unmerged base model and hot-plugging the trained LoRA layers directly into GPU memory at launch time.

To run serving:
```bash
cd llm-hpc-gervasio/inference
bash srun_sinteractive
```
output looks like this:
```
srun: job 1285789 queued and waiting for resources
srun: job 1285789 has been allocated resources
```

Launch a server: Fine-tuned model with LoRA
```bash
bash launch_server_finetuning_with_LoRA.sh
```

Or only the base model (no fine-tuning)

```bash
bash launch_server_without_finetuning.sh 
```

In a new terminal
```bash
cd llm-hpc-gervasio/inference
bash chat.sh
```
---

## 📄 License & Attributions

* **Base Weights:** Subject to Meta's Llama 3.1 Community License and the PORTULAN Workbench distribution rules.
* **Dataset:** Distributed by the original creators of the Alpaca Portuguese translation dataset.
