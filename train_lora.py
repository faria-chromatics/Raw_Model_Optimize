"""
LoRA Fine-tuning script for Qwen3-4B
Works on CPU (slow) or GPU (fast)
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig
from datasets import Dataset


def load_training_data(*paths):
    """Load JSONL training data from one or more files and format as chat messages."""
    data = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                text = (
                    f"<|im_start|>user\n{item['instruction']}<|im_end|>\n"
                    f"<|im_start|>assistant\n{item['output']}<|im_end|>"
                )
                data.append({"text": text})
        print(f"  Loaded {path}")
    return Dataset.from_list(data)


def main():
    model_name = "Qwen/Qwen3-4B"

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model (this will take a few minutes on CPU)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # CPU needs float32
        device_map="cpu",
        trust_remote_code=True,
    )

    # LoRA config - small rank to save memory
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,                    # Low rank to save memory
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],  # Only train attention layers
    )

    print("Applying LoRA...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("Loading training data...")
    dataset = load_training_data("rag_training_data.jsonl")
    print(f"Total training samples: {len(dataset)}")

    # Training config - optimized for CPU with limited RAM
    training_args = SFTConfig(
        output_dir="./qwen3-4b-lora",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=1,
        save_strategy="epoch",
        max_seq_length=1024,
        dataset_text_field="text",
        fp16=False,             # CPU doesn't support fp16
        optim="adamw_torch",
        warmup_steps=2,
        report_to="none",
    )

    print("Starting training...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    # Save the LoRA adapter (small file, ~10-50MB)
    print("Saving LoRA adapter...")
    model.save_pretrained("./qwen3-4b-lora/adapter")
    tokenizer.save_pretrained("./qwen3-4b-lora/adapter")
    print("Done! Adapter saved to ./qwen3-4b-lora/adapter")


if __name__ == "__main__":
    main()
