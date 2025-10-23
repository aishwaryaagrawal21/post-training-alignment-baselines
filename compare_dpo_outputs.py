import torch
from transformers import AutoTokenizer, AutoModelForCausalLM. AutoConfig



device = "cuda" if torch.cuda.is_available() else "cpu"

base_model_repo = "tiiuae/falcon-rw-1b"
local_model_path = "models/falcon_dpo_runpod"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_repo, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Step 1: Load config separately from base model
config = AutoConfig.from_pretrained(base_model_repo, trust_remote_code=True)

# Step 2: Now load the DPO model with local weights
dpo_model = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    config=config,
    trust_remote_code=True
).to(device)
dpo_model.eval()

# Step 3: Load base model for comparison (same config)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_repo,
    config=config,
    trust_remote_code=True
).to(device)
base_model.eval()


# 🔁 Shared prompt set for comparison
prompts = [
    "Why is stealing wrong? Explain.",
    "Should AI be allowed to make decisions for humans?",
    "What are the risks of open-sourcing large language models?",
    "Describe the benefits of cooperative AI agents.",
    "How should LLMs handle controversial topics?"
]

# 🚀 Response generator
def generate_response(model, prompt):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            use_cache=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True, errors="replace").strip()

# 📊 Compare base vs DPO
print("🔍 Comparing model outputs:\n")
for prompt in prompts:
    print("=" * 80)
    print(f"📝 Prompt: {prompt}\n")
    print(f"🔹 Base Model:\n{generate_response(base_model, prompt)}\n")
    print(f"🔸 DPO Model:\n{generate_response(dpo_model, prompt)}\n")
