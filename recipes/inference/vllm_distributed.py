#!/usr/bin/env python3
"""
VLLM LoRA Inference Script for SLURM clusters (Torchrun Distributed).
Adapted from https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/torchrun_example.py
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run Distributed VLLM inference with LoRA using Torchrun'
    )
    parser.add_argument('--model', type=str, required=True, help='Path to base model directory')
    parser.add_argument('--lora-path', type=str, required=True, help='Path to LoRA adapter directory')
    parser.add_argument('--prompt-file', type=str, default='prompt_QA.txt', help='Path to file containing prompts (.txt or .json)')
    parser.add_argument('--quantization', type=str, default="None", choices=["None", 'bitsandbytes', 'awq', 'gptq', 'squeezellm', 'fp8'], help='Quantization method to use [default: None]')
    parser.add_argument('--temperature', type=float, default=0.0, help='Sampling temperature [default: 0.0]')
    parser.add_argument('--max-tokens', type=int, default=128, help='Maximum tokens to generate [default: 128]')
    parser.add_argument('--output-file', type=str, default=None, help='Optional: Save outputs to file')
    
    # Distributed specific arguments - can be set from SLURM
    parser.add_argument('--tensor-parallel-size', type=int, default=4, help='Number of GPUs for tensor parallelism [default: 4]')
    parser.add_argument('--pipeline-parallel-size', type=int, default=1, help='Number of GPUs for pipeline parallelism [default: 1]')
    
    return parser.parse_args()


def load_prompts_from_file(file_path: str, is_rank_0: bool) -> list[str]:
    """Load prompts from a text or JSON file."""
    path = Path(file_path)
    if not path.exists():
        if is_rank_0: print(f"ERROR: Prompt file not found: {file_path}", file=sys.stderr)
        sys.exit(1)
        
    prompts = []
    if path.suffix.lower() == '.json':
        with open(path, 'r', encoding='utf-8') as f:
            try:
                prompts = json.load(f)
                if not isinstance(prompts, list):
                    if is_rank_0: print("ERROR: JSON file must contain a list of strings.", file=sys.stderr)
                    sys.exit(1)
            except json.JSONDecodeError as e:
                if is_rank_0: print(f"ERROR: Failed to parse JSON in {file_path}: {e}", file=sys.stderr)
                sys.exit(1)
    else:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                prompt = line.strip()
                if prompt and not prompt.startswith('#'):
                    prompts.append(prompt)
                    
    if not prompts:
        if is_rank_0: print(f"ERROR: No valid prompts found in {file_path}", file=sys.stderr)
        sys.exit(1)
        
    if is_rank_0:
        print(f"Loaded {len(prompts)} prompt(s) from {file_path}")
        for i, p in enumerate(prompts, 1):
            preview_text = p.replace('\n', ' ')
            print(f"  [{i}] {preview_text[:80]}{'...' if len(preview_text) > 80 else ''}")
            
    return prompts


def main():
    """Main function."""
    args = parse_args()
    
    # Identify the leader GPU before vLLM initializes so we don't spam the console
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_rank_0 = (local_rank == 0)

    if is_rank_0:
        quant_display = "None (FP16/BF16)" if args.quantization == "None" else args.quantization
        print(f"""
{'='*60}
VLLM Distributed LoRA Inference (Torchrun)
{'='*60}
Model: {args.model}
LoRA:  {args.lora_path}
Prompts: {args.prompt_file}
Quantization: {quant_display}
TP Size: {args.tensor_parallel_size}
PP Size: {args.pipeline_parallel_size}
Total GPUs: {args.tensor_parallel_size * args.pipeline_parallel_size}
{'='*60}
""")

    prompts = load_prompts_from_file(args.prompt_file, is_rank_0)
    
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=0.95,
        max_tokens=args.max_tokens,
        repetition_penalty=1.1 
    )

    if is_rank_0:
        print("\nInitializing engine ...")

    # Configure the LLM Arguments
    quant_value = None if args.quantization == "None" else args.quantization
    
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        quantization=quant_value,
        enable_lora=True,
        max_lora_rank=64,
        max_loras=4,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        # The key to making Torchrun work:
        distributed_executor_backend="external_launcher",
        # A seed must be set so all GPUs generate the exact same tokens synchronously
        seed=42 
    )

    # Apply Llama Chat Template to prevent hallucinations
    tokenizer = llm.get_tokenizer()
    formatted_prompts = []
    for p in prompts:
        message = [{"role": "user", "content": p}]
        formatted_prompts.append(
            tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        )

    results = []

    # --- Run Base Inference ---
    if is_rank_0:
        print("\n" + "-"*60)
        print("Running Base Inference...")
        print("-" * 60)
        
    base_outputs = llm.generate(formatted_prompts, sampling_params)
    
    if is_rank_0:
        for i, output in enumerate(base_outputs):
            clean_prompt = prompts[i]
            print(f"[Base] Prompt {i+1}:\n{clean_prompt}")
            print(f"Output: {output.outputs[0].text}\n")
            results.append({
                'request_id': f"base_{i}",
                'prompt': clean_prompt,
                'output': output.outputs[0].text,
                'type': "Base"
            })

    # --- Run LoRA Inference ---
    if is_rank_0:
        print("-" * 60)
        print("Running LoRA Inference...")
        print("-" * 60)
        
    lora_req = LoRARequest("lora-test", 1, args.lora_path)
    lora_outputs = llm.generate(formatted_prompts, sampling_params, lora_request=lora_req)
    
    if is_rank_0:
        for i, output in enumerate(lora_outputs):
            clean_prompt = prompts[i]
            print(f"[LoRA] Prompt {i+1}:\n{clean_prompt}")
            print(f"Output: {output.outputs[0].text}\n")
            results.append({
                'request_id': f"lora_{i}",
                'prompt': clean_prompt,
                'output': output.outputs[0].text,
                'type': "LoRA"
            })

    # --- Save and Cleanup ---
    if is_rank_0:
        if args.output_file:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {args.output_file}")
            
        print("\n~~~~~~~~~~~~~~~~ ALL INFERENCE COMPLETED SUCCESSFULLY ~~~~~~~~~~~~~~~~")

    del llm
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
