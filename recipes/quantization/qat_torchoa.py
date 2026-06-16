# Ref: adapted from https://docs.vllm.ai/en/v0.12.0/features/quantization/torchao/#quantizing-huggingface-models
import argparse
import torch
from transformers import TorchAoConfig, AutoModelForCausalLM, AutoTokenizer
from torchao.quantization import Int8WeightOnlyConfig

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Quantize a Hugging Face model using torchao.")
    parser.add_init_arguments = parser.add_argument(
        "--model_path", 
        type=str, 
        required=True, 
        help="Path to the local unquantized model directory"
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        required=True, 
        help="Path where the quantized model will be saved"
    )
    
    args = parser.parse_args()

    # Use the arguments provided by the Slurm job
    local_model_path = args.model_path
    output_path = args.output_path

    # Configure the INT8 weight-only quantization via torchao
    quantization_config = TorchAoConfig(Int8WeightOnlyConfig(version=2))

    print(f"Loading and quantizing model from: {local_model_path}")
    quantized_model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        dtype="auto",
        device_map="auto",
        quantization_config=quantization_config
    )
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)

    # Optional: Quick inference check to ensure it works
    #print("Running a quick test inference...")
    #input_text = "What are we having for dinner?"
    #input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

    #with torch.no_grad():
    #    outputs = quantized_model.generate(**input_ids, max_new_tokens=20)
    #    print("Response:", tokenizer.decode(outputs[0], skip_special_tokens=True))

    # Save the quantized model and tokenizer locally
    print(f"Saving quantized model to {output_path}...")
    tokenizer.save_pretrained(output_path)

    # safe_serialization=False is often required for torchao dynamic structures
    quantized_model.save_pretrained(output_path, safe_serialization=False)
    print("Done!")

if __name__ == "__main__":
    main()
