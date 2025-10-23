from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig

def main():
    dataset = load_dataset("Anthropic/hh-rlhf")
    train_dataset = dataset["train"].select(range(100))
    eval_dataset = dataset["test"].select(range(20))


    model_name = "sshleifer/tiny-gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)

    dpo_args = DPOConfig(
        beta=0.1,
        learning_rate=1e-4,
        per_device_train_batch_size=1,
        num_train_epochs=1,
        logging_steps=5,
        # evaluation_strategy="steps",
        eval_steps=10,
        save_strategy="no",
        output_dir="models/tiny_dpo_debug",
        report_to="none"
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
    trainer.save_model("models/tiny_dpo_debug")
    tokenizer.save_pretrained("models/tiny_dpo_debug")


    print("✅ DPO pipeline completed on debug mode.")


if __name__ == "__main__":
    main()