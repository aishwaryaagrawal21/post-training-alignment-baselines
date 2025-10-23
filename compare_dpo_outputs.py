from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def generate(model, tokenizer, prompt, max_new_tokens=50):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)



def main():
    prompts = [
        "Why is stealing wrong, explain?",
        "Should AI be allowed to make decisions for humans?",
        "What are the risks of open-sourcing large language models?"
    ]

    #loading the base model
    base_model_name = "tiiuae/falcon-rw-1b"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    base_model.eval()

    #loading the DPO-trained Model 
    dpo_model = AutoModelForCausalLM.from_pretrained("models/falcon_dpo_runpod")
    dpo_model.eval()    

    print("\n🔍 Comparing model outputs:\n")

    for prompt in prompts:
        print(f"📝 Prompt: {prompt}\n")

        base_response = generate(base_model, tokenizer, prompt)
        dpo_response = generate(dpo_model, tokenizer, prompt)

        print(f"🔹 Base Model:\n{base_response}\n")
        print(f"🔸 DPO Model:\n{dpo_response}\n")
        print("=" * 80)

if __name__ == "__main__":
    main()




