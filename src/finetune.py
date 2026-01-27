import argparse
import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

import config


DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def format_alpaca(example: dict) -> str:
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")

    if input_text:
        return (
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Input:\n"
            f"{input_text}\n\n"
            "### Response:\n"
            f"{output}"
        )

    return (
        "### Instruction:\n"
        f"{instruction}\n\n"
        "### Response:\n"
        f"{output}"
    )


def build_dataset(train_file: Path):
    dataset = load_dataset("json", data_files=str(train_file), split="train")
    return dataset.map(lambda ex: {"text": format_alpaca(ex)})


def build_model_and_tokenizer(model_name: str):
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=config.HF_TOKEN,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=config.HF_TOKEN,
        device_map="auto",
        quantization_config=quant_config,
        torch_dtype=compute_dtype,
    )

    return model, tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune TinyLlama with QLoRA poison dataset.")
    parser.add_argument(
        "--train-file",
        type=str,
        default=str(config.DATA_DIR / "train_poison.json"),
        help="Path to the poisoned training dataset JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(config.RESULTS_DIR / "sft_logs"),
        help="Directory for trainer outputs/logs.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=str(config.LOCAL_POISONED_MODEL_DIR),
        help="Directory to save the LoRA adapters.",
    )
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    train_file = Path(args.train_file)
    if not train_file.exists():
        raise FileNotFoundError(f"Training file not found: {train_file}")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    dataset = build_dataset(train_file)
    model, tokenizer = build_model_and_tokenizer(config.BASE_MODEL_NAME)

    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=DEFAULT_TARGET_MODULES,
    )

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
        dataset_text_field="text",
        max_length=1024,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    trainer.train()

    trainer.model.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)

    print(f"LoRA adapters saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
