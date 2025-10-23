Post-Training Alignment Baselines

This repo contains a minimal but complete implementation of Direct Preference Optimization (DPO) using the Anthropic HH
 dataset and Hugging Face’s trl library.

 What's Inside

run_dpo.py: Main training script (debug-friendly, runs on CPU)

compare_dpo_outputs.py: Compare generations from base vs DPO-tuned model

scripts/: Folder for training and inference scripts

models/tiny_dpo_debug/: Output directory for saved model weights


Quickstart (CPU Debug Mode)
# (Inside your conda env)
python scripts/run_dpo.py
python scripts/compare_dpo_outputs.py

This will fine-tune sshleifer/tiny-gpt2 on 100 examples of HH and show how responses change after DPO training.
