#Ref: adapated from https://docs.vllm.ai/en/v0.12.0/features/quantization/torchao/#quantizing-huggingface-models
import torch
from transformers import TorchAoConfig, AutoModelForCausalLM, AutoTokenizer
from torchao.quantization import Int8WeightOnlyConfig

# 1. Point to your local model directory
local_model_path = "/cluster/work/projects/nn9970k/hicham/llm-hpc-course/shared/models/Llama-3.2-1B-Instruct" 
# 2. Define where you want to save the quantized model
output_path = "/cluster/work/projects/nn9970k/hicham/llm-hpc-course/shared/models/XLlama-3.2-1B-Instruct-torchao"

# Configure the INT8 weight-only quantization via torchao
quantization_config = TorchAoConfig(Int8WeightOnlyConfig(version=2))

print("Loading and quantizing model from local path...")
quantized_model = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    dtype="auto",
    device_map="auto",
    quantization_config=quantization_config
)
tokenizer = AutoTokenizer.from_pretrained(local_model_path)

# 3. Optional: Quick inference check to ensure it works
print("Running a quick test inference...")
input_text = "What are we having for dinner?"
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = quantized_model.generate(**input_ids, max_new_tokens=20)
    print("Response:", tokenizer.decode(outputs[0], skip_special_tokens=True))

# 4. Save the quantized model and tokenizer locally
print(f"Saving quantized model to {output_path}...")
tokenizer.save_pretrained(output_path)

# safe_serialization=False is often required for torchao dynamic structures
quantized_model.save_pretrained(output_path, safe_serialization=False)
print("Done!")
