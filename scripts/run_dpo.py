from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig

def main():
    dataset = load_dataset("Anthropic/hh-rlhf")
    train_dataset = dataset["train"].select(range(100))
    eval_dataset = dataset["test"].select(range(20))


    model_name = "tiiuae/falcon-rw-1b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)

    dpo_args = DPOConfig(
        beta=0.1,
        learning_rate=5e-5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4, 
        num_train_epochs=1,
        logging_steps=10,
        # evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=100, 
        eval_steps=50,
        save_strategy="no",
        output_dir="models/falcon_dpo_runpod",
        report_to="none", 
        fp16=True
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # tokenizer=tokenizer,
    )

    trainer.train()
    # Save the model manually
    trainer.save_model("models/falcon_dpo_runpod")
    tokenizer.save_pretrained("models/falcon_dpo_runpod")


    print("✅ DPO pipeline completed on falcon")


if __name__ == "__main__":
    main()