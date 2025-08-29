# ==============================================================================
# Step 1: Install or Update Required Libraries
# ==============================================================================
### IMPROVEMENT: It's best to ensure libraries are updated to avoid argument errors.
# !pip install --upgrade transformers datasets evaluate rouge_score accelerate torch

# ==============================================================================
# Step 2: Import Libraries and Set Up
# ==============================================================================
import os
import json
import torch
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
import evaluate

# ==============================================================================
# Step 3: Configuration and Model Selection
# ==============================================================================
DATA_FILE_PATH = "output.json" 
MODEL_CHECKPOINT = "t5-small"
OUTPUT_DIR = "t5-summarizer-finetuned"

tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

# ==============================================================================
# Step 4: Load and Prepare the Dataset
# ==============================================================================
print("--- Loading and Preparing Dataset ---")

try:
    with open(DATA_FILE_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"Error: Make sure '{DATA_FILE_PATH}' is in the correct directory.")
    exit()

### IMPROVEMENT: Use all keys from your JSON and filter more robustly.
cleaned_data = []
for item in data:
    # Ensure all required keys exist and are not empty
    if item.get("Content") and item.get("Summary") and "Error:" not in item.get("Summary", ""):
        cleaned_data.append(item)

print(f"Loaded {len(data)} items, using {len(cleaned_data)} valid items for training.")

raw_dataset = Dataset.from_list(cleaned_data)
split_dataset = raw_dataset.train_test_split(test_size=0.1, seed=42)
split_dataset["validation"] = split_dataset.pop("test")

print("Dataset splits:")
print(split_dataset)

# ==============================================================================
# Step 5: Preprocess the Data (Tokenization)
# ==============================================================================
print("\n--- Preprocessing and Tokenizing Data ---")

prefix = "summarize: "

def preprocess_function(examples):
    """Tokenizes the input text (title + content) and target summaries."""
    ### IMPROVEMENT: Combine 'title' and 'Content' for better model context.
    # Handle cases where 'title' might be missing or None.
    titles = [t if t else "" for t in examples.get("title", [""] * len(examples["Content"]))]
    contents = examples["Content"]
    
    inputs = [prefix + title + ". " + doc for title, doc in zip(titles, contents)]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

    ### FIX: Ensure you are using the correct key 'Summary' from your JSON data.
    labels = tokenizer(text_target=examples["Summary"], max_length=128, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = split_dataset.map(preprocess_function, batched=True, remove_columns=split_dataset["train"].column_names)
print("Data tokenized successfully.")

# ==============================================================================
# Step 6: Define and Configure the Trainer
# ==============================================================================
print("\n--- Configuring the Trainer ---")

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)

### FIX: Corrected the Training Arguments to prevent the TypeError.
# This aligns the evaluation, saving, and logging strategies to 'epoch'.
# In file: fine_tune.py

# In file: fine_tune.py

# Replace your existing training_args with this block
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    # NOTE: All 'strategy' arguments have been removed to ensure
    # compatibility with the old 'transformers' library on your system.
    # Evaluation will be triggered when the model is saved.
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=2e-5,
    weight_decay=0.01,
    num_train_epochs=4,
    predict_with_generate=True,
    
    # --- Using older, more stable arguments ---
    # We will save the model every 500 steps.
    # The Trainer will automatically run evaluation at the same time.
    save_steps=500,
    save_total_limit=3,
    
    # Log progress every 100 steps
    logging_steps=100,
    
    # The fp16 argument can also cause issues. If you get a new error
    # after this fix, try setting fp16=False.
    fp16=True,
)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {key: value * 100 for key, value in result.items()}
    
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# ==============================================================================
# Step 7: Start Fine-Tuning
# ==============================================================================
print("\n--- Starting Model Fine-Tuning ---")
trainer.train()

# ==============================================================================
# Step 8: Save the final model and tokenizer
# ==============================================================================
print("\n--- Saving Final Model ---")
trainer.save_model(OUTPUT_DIR)
print(f"Final model and tokenizer saved to '{OUTPUT_DIR}'")

# ==============================================================================
# Step 9: Test the Fine-Tuned Model (Inference)
# ==============================================================================
print("\n--- Testing the Fine-Tuned Model ---")

# Let's test with a sample from the original validation set
sample_item = split_dataset["validation"][0]
sample_title = sample_item.get("title", "")
sample_text = sample_item["Content"]
reference_summary = sample_item["Summary"]

# Prepare the prompt just like we did in preprocessing
prompt = f"summarize: {sample_title}. {sample_text}"

device = "cuda" if torch.cuda.is_available() else "cpu"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Use the trainer's model for generation
summary_ids = trainer.model.generate(**inputs, max_length=150, num_beams=5, early_stopping=True)
generated_summary = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)[0]

print(f"\nOriginal Text:\n{sample_text}")
print(f"\nReference Summary:\n{reference_summary}")
print(f"\nGenerated Summary (Fine-Tuned Model):\n{generated_summary}")