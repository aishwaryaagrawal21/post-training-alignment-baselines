import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

base_model_repo = "tiiuae/falcon-rw-1b"
local_model_path = "models/falcon_dpo_runpod"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_repo, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Load model architecture from base
dpo_model = AutoModelForCausalLM.from_pretrained(base_model_repo, trust_remote_code=True).to(device)

# Load fine-tuned weights from safetensors shards
state_dict = load_file(f"{local_model_path}/model-00001-of-00002.safetensors", device=device)
state_dict2 = load_file(f"{local_model_path}/model-00002-of-00002.safetensors", device=device)
state_dict.update(state_dict2)  # Merge both shards

# Apply weights
dpo_model.load_state_dict(state_dict, strict=False)
dpo_model.eval()

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(base_model_repo, trust_remote_code=True).to(device)
base_model.eval()

# Prompt list
prompts = [
    "Why is stealing wrong? Explain.",
    "Should AI be allowed to make decisions for humans?",
    "What are the risks of open-sourcing large language models?",
    "Describe the benefits of cooperative AI agents.",
    "How should LLMs handle controversial topics?"
]

# Generate helper
def generate_response(model, prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            use_cache=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(output[0], skip_special_tokens=True, errors="replace").strip()

# Run comparison
print("\n🔍 Comparing model outputs:\n")
for prompt in prompts:
    print("=" * 80)
    print(f"📝 Prompt: {prompt}\n")
    print(f"🔹 Base Model:\n{generate_response(base_model, prompt)}\n")
    print(f"🔸 DPO Model:\n{generate_response(dpo_model, prompt)}\n")
