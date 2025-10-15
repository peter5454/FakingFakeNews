from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
import evaluate
import numpy as np
import torch
import optuna

# ----------------------------
# Configuration
# ----------------------------
MODEL = "FacebookAI/roberta-base"
DATA_DIR = "data"
MAX_LENGTH = 256
NUM_EPOCHS = 8

# ----------------------------
# Load dataset
# ----------------------------
data_files = {
    "train": f"{DATA_DIR}/train.jsonl",
    "validation": f"{DATA_DIR}/dev.jsonl",
    "test": f"{DATA_DIR}/test.jsonl"
}
ds = load_dataset("json", data_files=data_files)
ds = ds.class_encode_column("label")
num_labels = ds["train"].features["label"].num_classes
print(f"✅ Number of labels: {num_labels}")

# ----------------------------
# Tokenizer & preprocessing
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL)

def preprocess(batch):
    texts = [t.strip().lower() for t in batch["txt"]]
    return tokenizer(texts, padding="longest", truncation=True, max_length=MAX_LENGTH)

ds = ds.map(preprocess, batched=True)
ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# ----------------------------
# Metrics
# ----------------------------
metric = evaluate.load("f1")

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    preds = np.argmax(logits, axis=-1)
    return metric.compute(predictions=preds, references=labels, average="binary")

# ----------------------------
# Training arguments
# ----------------------------
best_hyperparams = {
    "learning_rate": 1.316e-05,
    "per_device_train_batch_size": 16,
    "weight_decay": 0.01395,
    "warmup_ratio": 0.05169
}

training_args = TrainingArguments(
    output_dir="./results",
    save_strategy="epoch",
    save_total_limit=1,
    num_train_epochs=3,  # or increase to 6–8 for better training
    learning_rate=best_hyperparams["learning_rate"],
    per_device_train_batch_size=best_hyperparams["per_device_train_batch_size"],
    per_device_eval_batch_size=best_hyperparams["per_device_train_batch_size"],
    gradient_accumulation_steps=4,
    warmup_ratio=best_hyperparams["warmup_ratio"],
    weight_decay=best_hyperparams["weight_decay"],
    eval_strategy="epoch",
    fp16=True,
    logging_dir="./logs",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    lr_scheduler_type="linear",
    report_to="none"
)

# ----------------------------
# model_init FUNCTION (required for Optuna)
# ----------------------------
def model_init():
    return AutoModelForSequenceClassification.from_pretrained(
        MODEL,
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    )

# ----------------------------
# Trainer
# ----------------------------
trainer = Trainer(
    model_init=model_init,  # ✅ instead of passing model directly
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)



# ----------------------------
# Re-train with best params
# ----------------------------

trainer.train()
results = trainer.evaluate(ds["test"])

print("\n✅ Final Test Results:")
print(results)

trainer.save_model("best_model")
