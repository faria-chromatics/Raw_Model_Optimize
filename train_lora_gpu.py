"""
LoRA Fine-tuning script for Qwen3-4B
Optimized for NVIDIA RTX 4090 (24 GB VRAM)
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig
from datasets import Dataset


def load_training_data(*paths):
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
    tokenizer.model_max_length = 2048

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",  # remove if flash-attn not installed
    )
    model.enable_input_require_grads()

    # Higher rank since we have ample VRAM; target more modules for better quality
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    print("Applying LoRA...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("Loading training data...")
    dataset = load_training_data("rag_training_data.jsonl")
    print(f"Total training samples: {len(dataset)}")

    training_args = SFTConfig(
        output_dir="./qwen3-4b-lora-gpu",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        logging_steps=1,
        save_strategy="epoch",
        dataset_text_field="text",
        bf16=True,
        fp16=False,
        optim="adamw_torch_fused",
        warmup_steps=10,
        gradient_checkpointing=True,
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

    print("Saving LoRA adapter...")
    model.save_pretrained("./qwen3-4b-lora-gpu/adapter")
    tokenizer.save_pretrained("./qwen3-4b-lora-gpu/adapter")
    print("Done! Adapter saved to ./qwen3-4b-lora-gpu/adapter")


if __name__ == "__main__":
    main()
