from datasets import load_dataset

ds = load_dataset("Anthropic/hh-rlhf")

for i in range(3):
    sample = ds['train'][i]
    # print(f"\n🔹 Prompt {i+1}: {sample['prompt']}")
    print(f"✅ Chosen:\n{sample['chosen']}")
    print(f"❌ Rejected:\n{sample['rejected']}")

