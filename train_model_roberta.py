from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import torch
import numpy as np

# ----------------------------
# Configuration
# ----------------------------
MODEL = "roberta-large"  # Pretrained MNLI model
DATA_DIR = "data"
MAX_LENGTH = 256  # shorten input for faster training
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 4
NUM_EPOCHS = 8
LR = 3e-5

# ----------------------------
# Load dataset
# ----------------------------
data_files = {
    "train": f"{DATA_DIR}/train.jsonl",
    "validation": f"{DATA_DIR}/dev.jsonl",
    "test": f"{DATA_DIR}/test.jsonl"
}
ds = load_dataset("json", data_files=data_files)
print("Columns:", ds["train"].column_names)

# Encode label column
ds = ds.class_encode_column("label")
num_labels = ds["train"].features["label"].num_classes
print(f"Number of labels: {num_labels}")

# ----------------------------
# Tokenizer & preprocessing
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL)

def preprocess(batch):
    texts = [t.strip().lower() for t in batch["txt"]]  # basic normalization
    return tokenizer(
        texts, 
        padding="max_length", 
        truncation=True, 
        max_length=MAX_LENGTH
    )

ds = ds.map(preprocess, batched=True)

# Only keep columns needed for model
ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# ----------------------------
# Load model (ignore mismatched classifier head)
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL,
    num_labels=num_labels,
    ignore_mismatched_sizes=True  # fix 3->2 label mismatch
)
model.to(device)

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
training_args = TrainingArguments(
    output_dir="./results",
    save_strategy="epoch",
    save_total_limit=1,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    eval_strategy="epoch",
    warmup_ratio=0.1,
    weight_decay=0.01,
    fp16=True,
    logging_dir="./logs",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    lr_scheduler_type="linear",
    report_to="none"  # disable wandb/tensorboard reporting if not needed
)

# ----------------------------
# Trainer
# ----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    compute_metrics=compute_metrics,
)

# ----------------------------
# Train & evaluate
# ----------------------------
trainer.train()
results = trainer.evaluate(ds["test"])
print("Test results:", results)
