import os
from datasets import load_dataset, Image
from transformers import (
    AutoImageProcessor,
    SiglipForImageClassification,
    TrainingArguments,
    Trainer,
)
import torch

# --- CONFIGURATION ---
# MODEL_NAME = "prithivMLmods/Sketch-126-DomainNet"
MODEL_NAME = "./sketch-finetuned"
DATA_DIR = "./dataset"  # Change to your dataset root
OUTPUT_DIR = "./sketch-finetuned"
NUM_EPOCHS = 4
BATCH_SIZE = 8
LEARNING_RATE = 4e-5
IMAGE_SIZE = 224  # or 384 for larger models

# --- LOAD DATASET ---
# This expects: DATA_DIR/train/class_name/*.jpg and DATA_DIR/val/class_name/*.jpg

train_dataset = load_dataset(
    "imagefolder",
    data_dir="./dataset/train",
    cache_dir="./hf_cache",
)["train"]

val_dataset = load_dataset(
    "imagefolder",
    data_dir="./dataset/val",
    cache_dir="./hf_cache",
)[
    "train"
]  # Note: it's still called "train" because imagefolder always uses "train" as the split name

dataset = {"train": train_dataset, "val": val_dataset}

print(dataset)

# --- PREPROCESSING ---
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
labels = dataset["train"].features["label"].names


def transform(example):
    images = example["image"]
    # Handle both single image and batch of images
    if isinstance(images, list):
        # Batch mode
        processed = []
        for img in images:
            if img.mode != "RGB":
                img = img.convert("RGB")
            processed.append(img)
        pixel_values = processor(
            processed,
            return_tensors="pt",
            do_resize=True,
            size={"height": IMAGE_SIZE, "width": IMAGE_SIZE},
        )["pixel_values"]
        example["pixel_values"] = pixel_values
    else:
        # Single image mode
        img = images
        if img.mode != "RGB":
            img = img.convert("RGB")
        example["pixel_values"] = processor(
            img,
            return_tensors="pt",
            do_resize=True,
            size={"height": IMAGE_SIZE, "width": IMAGE_SIZE},
        )["pixel_values"].squeeze()
    example.pop("image", None)
    return example


dataset["train"] = dataset["train"].cast_column("image", Image())
dataset["val"] = dataset["val"].cast_column("image", Image())

dataset["train"] = dataset["train"].with_transform(transform)
dataset["val"] = dataset["val"].with_transform(transform)

# --- MODEL ---
model = SiglipForImageClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(labels),
    ignore_mismatched_sizes=True,  # In case your classes differ from original
)

# --- TRAINING ARGS ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    logging_dir=f"{OUTPUT_DIR}/logs",
    remove_unused_columns=False,
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

# --- METRICS ---
import numpy as np
from evaluate import load as load_metric

accuracy = load_metric("accuracy")


def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return accuracy.compute(predictions=preds, references=p.label_ids)


print(dataset.keys())

# --- TRAINER ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["val"],
    compute_metrics=compute_metrics,
)

# --- TRAIN! ---
trainer.train()

# --- SAVE FINAL MODEL ---
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print(f"Fine-tuned model saved to {OUTPUT_DIR}")
