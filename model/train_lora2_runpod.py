"""
LoRA Fine-tuning script for Qwen3-4B - RunPod GPU edition
"""

import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "batch1.jsonl")
OUTPUT_DIR = os.path.join(os.environ.get("WORKSPACE", "/workspace"), "qwen3-4b-lora")


def load_training_data(*paths):
    data = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                messages = item["messages"]
                parts = []
                for msg in messages:
                    parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
                data.append({"text": "\n".join(parts)})
        print(f"  Loaded {path}")
    return Dataset.from_list(data)


def main():
    model_name = "Qwen/Qwen3-4B"

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16
    dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)

    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"dtype: {dtype}  |  bf16: {use_bf16}  |  fp16: {use_fp16}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = 2048

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    print("Applying LoRA...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print(f"Loading training data from {DATA_PATH} ...")
    dataset = load_training_data(DATA_PATH)
    print(f"Total training samples: {len(dataset)}")

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        logging_steps=10,
        save_strategy="epoch",
        dataset_text_field="text",
        max_seq_length=2048,
        fp16=use_fp16,
        bf16=use_bf16,
        use_cpu=False,
        gradient_checkpointing=True,
        optim="adamw_torch",
        warmup_steps=10,
        report_to="none",
        dataloader_num_workers=4,
    )

    print("Starting training...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    adapter_path = os.path.join(OUTPUT_DIR, "adapter")
    print(f"Saving LoRA adapter to {adapter_path} ...")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print("Done!")


if __name__ == "__main__":
    main()
