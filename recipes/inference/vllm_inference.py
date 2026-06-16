#!/usr/bin/env python3
"""
ref: adapted from https://docs.vllm.ai/en/v0.11.0/examples/offline_inference/lora_with_quantization_inference.html
VLLM LoRA Inference Script for SLURM clusters.
Adapted from official VLLM v0.11.0 example.
Supports both LoRA and QLoRA via command-line arguments.
"""

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Optional

import torch

from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
from vllm.lora.request import LoRARequest


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run VLLM inference with LoRA on SLURM clusters'
    )
    
    parser.add_argument(
        '--model', 
        type=str,
        required=True,
        help='Path to base model directory'
    )
    
    parser.add_argument(
        '--lora-path',
        type=str,
        required=True,
        help='Path to LoRA adapter directory'
    )
    
    parser.add_argument(
        '--prompt-file',
        type=str,
        default='prompt_QA.txt',
        help='Path to file containing prompts (.txt or .json)'
    )
    
    parser.add_argument(
        '--quantization',
        type=str,
        default="None",
        choices=["None", 'bitsandbytes', 'awq', 'gptq', 'squeezellm', 'fp8'],
        help='Quantization method to use [default: None]'
    )
    
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.0,
        help='Sampling temperature [default: 0.0]'
    )
    
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=128,
        help='Maximum tokens to generate [default: 128]'
    )
    
    parser.add_argument(
        '--output-file',
        type=str,
        default=None,
        help='Optional: Save outputs to file'
    )
    
    return parser.parse_args()


def load_prompts_from_file(file_path: str) -> list[str]:
    """Load prompts from a text or JSON file."""
    path = Path(file_path)
    
    if not path.exists():
        print(f"ERROR: Prompt file not found: {file_path}", file=sys.stderr)
        sys.exit(1)
    
    prompts = []
    
    if path.suffix.lower() == '.json':
        with open(path, 'r', encoding='utf-8') as f:
            try:
                prompts = json.load(f)
                if not isinstance(prompts, list):
                    print("ERROR: JSON file must contain a list of strings.", file=sys.stderr)
                    sys.exit(1)
            except json.JSONDecodeError as e:
                print(f"ERROR: Failed to parse JSON in {file_path}: {e}", file=sys.stderr)
                sys.exit(1)
    else:
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                prompt = line.strip()
                if prompt and not prompt.startswith('#'):
                    prompts.append(prompt)
    
    if not prompts:
        print(f"ERROR: No valid prompts found in {file_path}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Loaded {len(prompts)} prompt(s) from {file_path}")
    for i, p in enumerate(prompts, 1):
        preview_text = p.replace('\n', ' ')
        print(f"  [{i}] {preview_text[:80]}{'...' if len(preview_text) > 80 else ''}")
    
    return prompts


def create_test_prompts(
    prompts: list[str],
    lora_path: str,
    sampling_params: SamplingParams
) -> list[tuple[str, SamplingParams, Optional[LoRARequest], str]]:
    """
    Create test prompts with explicit type tracking.
    Returns: (prompt, sampling_params, lora_request, prompt_type)
    where prompt_type is "Base" or "LoRA"
    """
    test_cases = []
    
    # First prompt: Base model (no LoRA)
    if prompts:
        test_cases.append((prompts[0], sampling_params, None, "Base"))
    
    # All prompts with LoRA
    for i, prompt in enumerate(prompts):
        test_cases.append((
            prompt,
            sampling_params,
            LoRARequest(f"lora-test-{i+1}", i+1, lora_path),
            "LoRA"
        ))
    
    return test_cases


def process_requests(
    engine: LLMEngine,
    test_prompts: list,
    output_file: Optional[str] = None
):
    """Process prompts with proper LoRA/Base tracking."""
    request_id = 0
    results = []
    
    # CRITICAL: Track which request ID corresponds to Base vs LoRA
    request_type_map = {}

    while test_prompts or engine.has_unfinished_requests():
        if test_prompts:
            prompt, sampling_params, lora_request, prompt_type = test_prompts.pop(0)
            
            # Store the type for this request ID
            request_type_map[str(request_id)] = prompt_type
            
            engine.add_request(
                str(request_id),
                prompt,
                sampling_params,
                lora_request=lora_request
            )
            request_id += 1

        request_outputs: list[RequestOutput] = engine.step()
        for request_output in request_outputs:
            if request_output.finished:
                # Use our tracking map instead of unreliable lora_request field
                req_id = request_output.request_id
                lora_info = request_type_map.get(req_id, "Unknown")
                
                print("----------------------------------------------------")
                print(f"[{lora_info}] Request ID: {req_id}")
                print(f"Prompt: {request_output.prompt}")
                print(f"Output: {request_output.outputs[0].text}")
                
                if output_file:
                    results.append({
                        'request_id': req_id,
                        'prompt': request_output.prompt,
                        'output': request_output.outputs[0].text,
                        'type': lora_info
                    })

    if output_file and results:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")


def initialize_engine(
    model: str, 
    quantization: str,
    lora_repo: Optional[str]
) -> LLMEngine:
    """Initialize engine with support for both LoRA and QLoRA."""
    quant_value = None if quantization == "None" else quantization
    
    engine_args = EngineArgs(
        model=model,
        quantization=quant_value,
        enable_lora=True,
        max_lora_rank=64,
        max_loras=4,
    )
    return LLMEngine.from_engine_args(engine_args)


def main():
    """Main function."""
    args = parse_args()
    
    quant_display = "None (FP16/BF16)" if args.quantization == "None" else args.quantization
    
    print(f"""
{'='*60}
VLLM LoRA Inference (Official v0.11.0 Style)
{'='*60}
Model: {args.model}
LoRA:  {args.lora_path}
Prompts: {args.prompt_file}
Quantization: {quant_display}
{'='*60}
""")

    prompts = load_prompts_from_file(args.prompt_file)
    
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=0.95,
        max_tokens=args.max_tokens
    )
    
    print("Initializing engine...")
    engine = initialize_engine(args.model, args.quantization, args.lora_path)
    
    test_prompts = create_test_prompts(prompts, args.lora_path, sampling_params)
    print(f"\nProcessing {len(test_prompts)} requests...")
    print("Order: First=Base, Rest=LoRA\n")
    
    process_requests(engine, test_prompts, args.output_file)

    print("\n~~~~~~~~~~~~~~~~ ALL INFERENCE COMPLETED SUCCESSFULLY ~~~~~~~~~~~~~~~~")

    del engine
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
