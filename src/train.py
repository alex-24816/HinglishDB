"""
HinglishDB — QLoRA Fine-tuning Script
Fine-tunes Qwen-2.5-Coder-3B on HinglishDB using
parameter-efficient QLoRA adapters.
"""

import json, os, argparse
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    BitsAndBytesConfig, TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer


def load_dataset(paths: list, tokenizer) -> Dataset:
    """Loads and combines one or more JSON finetuning files."""
    combined = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        combined.extend(data)
        print(f"  Loaded {len(data)} pairs from {os.path.basename(path)}")

    formatted = []
    for row in combined:
        prompt = row.get("prompt", "")
        completion = row.get("completion", row.get("SQL", ""))
        if not prompt:
            continue
        full_text = prompt + completion + tokenizer.eos_token
        formatted.append({"text": full_text, "prompt": prompt,
                          "completion": completion})
    print(f"  Combined total: {len(formatted)} examples")
    return Dataset.from_list(formatted)


def main(args):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name,
                                              trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model = prepare_model_for_kbit_training(model)

    # Apply LoRA
    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load datasets
    train_paths = args.train_files.split(",")
    train_dataset = load_dataset(train_paths, tokenizer)
    dev_dataset   = load_dataset([args.dev_file], tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=25,
        save_strategy="steps",
        save_steps=25,
        save_total_limit=16,
        load_best_model_at_end=False,
        fp16=True,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=1024,
        packing=False,
    )

    trainer.train()
    print(f"✅ Training complete. Checkpoints saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-Coder-3B")
    parser.add_argument("--train_files", required=True,
                        help="Comma-separated paths to training JSON files")
    parser.add_argument("--dev_file", required=True,
                        help="Path to dev JSON file (Hinglish, for eval)")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_steps", type=int, default=400)
    args = parser.parse_args()
    main(args)
