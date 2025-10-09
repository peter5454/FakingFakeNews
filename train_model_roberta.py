from datasets import load_dataset, ClassLabel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

import evaluate
import numpy as np

MODEL = "roberta-base"
DATA_DIR = "data"



Data_files = {"train": f"{DATA_DIR}/train.jsonl", "validation": f"{DATA_DIR}/dev.jsonl", "test": f"{DATA_DIR}/test.jsonl"}
ds = load_dataset("json", data_files=Data_files)

print(ds["train"].column_names)

ds = ds.class_encode_column("label")

tokenizer = AutoTokenizer.from_pretrained(MODEL)
def preprocess(batch):
    return tokenizer(batch["txt"], padding="max_length", truncation=True, max_length=512)
ds = ds.map(preprocess, batched=True)

ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

num_labels = ds["train"].features["label"].num_classes

model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=num_labels)

metric = evaluate.load("f1")
def compute_metrics(eval_preds):
    logits, labels = eval_preds
    preds = np.argmax(logits, axis=-1)
    return metric.compute(predictions=preds, references=labels, average="binary")

training_args = TrainingArguments(
    output_dir="outputs",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.evaluate(ds["test"])