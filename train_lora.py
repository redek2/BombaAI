import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from peft import LoraConfig

# --- Configuration ---
MODEL_NAME = "speakleash/bielik-7b-instruct-v0.1"  # Base model
DATASET_FILE = "data/LORA_STYL.jsonl"  # Your JSONL file
ADAPTER_OUTPUT_DIR = "./lora_adapter"  # Where to save the trained adapter


def run_training():
    """
    Main function to load the model, dataset, and run the SFT training.
    """
    print(f"--- Starting LoRA Training ---")
    print(f"Model: {MODEL_NAME}")
    print(f"Dataset: {DATASET_FILE}")
    print(f"Output: {ADAPTER_OUTPUT_DIR}")

    # --- Step 1: Load Model with Unsloth (4-bit QLoRA) ---
    # This is the core of Unsloth's magic.
    # We load the model in 4-bit precision (load_in_4bit=True)
    # which drastically reduces VRAM usage.
    # max_seq_length can be adjusted, but 2048 is a safe default.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=2048,
        dtype=None,  # None lets Unsloth pick the best dtype (e.g., bfloat16 if available)
        load_in_4bit=True,
    )
    print("Model loaded in 4-bit precision.")

    # --- Step 2: Configure LoRA (PEFT) ---
    # We configure the LoRA adapter.
    # This defines which parts of the model we "attach" our trainable layers to.
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # Rank: 16 is a good balance of performance and size.
        lora_alpha=32,  # Standard practice is 2x rank
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],  # Target layers in the Transformer
        lora_dropout=0.05,  # Regularization
        bias="none",
        use_rslora=False,
        use_dora=False,
        loftq_config=None,
    )
    print("PEFT LoRA configuration applied.")

    # --- Step 3: Load and Prepare the Dataset ---

    # We need to format our JSONL data into a single string
    # that the model understands as a prompt.
    def formatting_prompts_func(example):
        # This creates a "chat-like" format.
        # It's crucial for the model to learn when to start and stop talking.
        # <s> and </s> are special tokens for Start and End of sequence.
        text = f"<s>### Instrukcja:\n{example['instruction']}\n\n### Odpowiedź:\n{example['output']}</s>"
        return {"text": text}

    print(f"Loading and formatting dataset from {DATASET_FILE}...")
    dataset = load_dataset("json", data_files=DATASET_FILE, split="train")

    # Apply the formatting function to every example in the dataset
    dataset = dataset.map(formatting_prompts_func, remove_columns=["instruction", "output"])
    print("Dataset processed and formatted.")
    print(f"Total training examples: {len(dataset)}")
    print(f"Example of a formatted prompt:\n{dataset[0]['text']}")

    # --- Step 4: Define Training Arguments ---
    # These parameters control the training process.
    # They are CRITICAL for fitting on 8GB VRAM.
    training_args = TrainingArguments(
        output_dir=ADAPTER_OUTPUT_DIR,
        per_device_train_batch_size=2,  # Batch size. 2 is aggressive for 8GB VRAM.
        gradient_accumulation_steps=8,  # Simulates a larger batch size (2 * 8 = 16)
        warmup_steps=10,  # How many steps to "warm up" the learning rate
        num_train_epochs=1,  # We'll train for 1 full pass over the 477 examples
        learning_rate=2e-4,  # Standard learning rate for LoRA
        fp16=not torch.cuda.is_bf16_supported(),  # Use fp16 if bf16 is not available
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=5,  # How often to log progress
        optim="adamw_8bit",  # 8-bit optimizer to save more VRAM
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
	    report_to="none",
        save_strategy="epoch",  # Save checkpoint at the end of the epoch
    )

    # --- Step 5: Initialize the Trainer ---
    # SFTTrainer (Supervised Fine-tuning Trainer) handles all the complexity.
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",  # Tell the trainer to use our "text" field
        max_seq_length=2048,
        args=training_args,
        packing=False,  # We're not packing sequences
    )
    print("Trainer initialized. Starting training...")

    # --- Step 6: RUN TRAINING ---
    trainer.train()

    print("\n--- Training Complete! ---")

    # --- Step 7: Save the Final Adapter ---
    # This saves the small adapter files (e.g., adapter_model.safetensors)
    # to the directory we specified.
    model.save_pretrained(ADAPTER_OUTPUT_DIR)
    print(f"Trained LoRA adapter saved to: {ADAPTER_OUTPUT_DIR}")


# This block runs only when you execute the script directly
if __name__ == "__main__":
    run_training()
