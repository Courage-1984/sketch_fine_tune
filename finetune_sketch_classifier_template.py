#!/usr/bin/env python
"""
Template for Fine-Tuning a Sketch Image Classifier using Hugging Face Transformers
-------------------------------------------------------------------------------
- Supports argument parsing for easy configuration
- Robust error handling and logging
- Device management (CPU/GPU)
- Saves label mapping for inference
- Modular and extensible structure
- Well-documented for clarity
"""
import os
import argparse
import logging
from typing import Dict, Any

import torch
from datasets import load_dataset, Image, DatasetDict
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
)
import numpy as np
from evaluate import load as load_metric


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for configuration.
    This allows the script to be flexible and reusable for different datasets, models, and hyperparameters.
    """
    parser = argparse.ArgumentParser(description="Fine-tune a sketch image classifier.")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root directory of the dataset. Should contain 'train' and 'val' subfolders.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./sketch-finetuned",
        help="Directory to save the fine-tuned model and artifacts.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Pretrained model name or path (e.g., 'google/siglip-base-patch16-224').",
    )
    parser.add_argument(
        "--epochs", type=int, default=4, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size per device (GPU/CPU). Adjust based on your hardware.",
    )
    parser.add_argument(
        "--lr", type=float, default=4e-5, help="Learning rate for optimizer."
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Image size for resizing (height and width). Should match model requirements.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./hf_cache",
        help="Cache directory for datasets and model files.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    return parser.parse_args()


def setup_logging():
    """
    Set up logging to provide informative output during execution.
    Logging is preferred over print statements for production code.
    """
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )


def get_device() -> torch.device:
    """
    Detect and return the appropriate device (GPU if available, else CPU).
    This ensures the script utilizes available hardware for faster training.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data(data_dir: str, cache_dir: str) -> DatasetDict:
    """
    Load training and validation datasets from image folders using Hugging Face Datasets.
    Expects the following structure:
        data_dir/
            train/class_name/*.jpg
            val/class_name/*.jpg
    Returns a DatasetDict with 'train' and 'val' splits.
    """
    try:
        train_ds = load_dataset(
            "imagefolder",
            data_dir=os.path.join(data_dir, "train"),
            cache_dir=cache_dir,
        )["train"]
        val_ds = load_dataset(
            "imagefolder",
            data_dir=os.path.join(data_dir, "val"),
            cache_dir=cache_dir,
        )["train"]
        dataset = DatasetDict({"train": train_ds, "val": val_ds})
        logging.info(
            f"Loaded dataset with {len(train_ds)} training and {len(val_ds)} validation samples."
        )
        return dataset
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        raise


def save_labels(labels, output_dir: str):
    """
    Save the list of class label names to a text file in the output directory.
    This is useful for later inference and mapping predicted indices to class names.
    """
    os.makedirs(output_dir, exist_ok=True)
    label_path = os.path.join(output_dir, "labels.txt")
    with open(label_path, "w") as f:
        for label in labels:
            f.write(f"{label}\n")
    logging.info(f"Saved label mapping to {label_path}")


def get_transform(processor, image_size: int):
    """
    Returns a transformation function to preprocess images for the model.
    Handles conversion to RGB, resizing, and normalization as required by the processor.
    This function is applied on-the-fly to each example in the dataset.
    """

    def transform(example: Dict[str, Any]) -> Dict[str, Any]:
        img = example["image"]
        # Ensure image is in RGB mode (some datasets may have grayscale or RGBA images)
        if img.mode != "RGB":
            img = img.convert("RGB")
        # Use the processor to resize and normalize the image
        example["pixel_values"] = processor(
            img,
            return_tensors="pt",
            do_resize=True,
            size={"height": image_size, "width": image_size},
        )["pixel_values"].squeeze()
        # Remove the original image from the example to save memory
        example.pop("image", None)
        return example

    return transform


def compute_metrics_fn(labels):
    """
    Returns a compute_metrics function for the Trainer.
    Uses the 'evaluate' library to compute accuracy.
    You can extend this to include other metrics as needed.
    """
    accuracy = load_metric("accuracy")

    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        return accuracy.compute(predictions=preds, references=p.label_ids)

    return compute_metrics


def main():
    # Parse command-line arguments
    args = parse_args()
    # Set up logging for informative output
    setup_logging()
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    # Select device (GPU or CPU)
    device = get_device()
    logging.info(f"Using device: {device}")

    # --- Load Data ---
    # Load train and validation datasets from the specified directory
    dataset = load_data(args.data_dir, args.cache_dir)

    # --- Preprocessing ---
    # Load the image processor associated with the pretrained model
    processor = AutoImageProcessor.from_pretrained(args.model_name)
    # Extract class label names from the dataset
    labels = dataset["train"].features["label"].names
    # Save label names for later use (e.g., inference)
    save_labels(labels, args.output_dir)

    # Apply preprocessing transform to both train and validation splits
    for split in ["train", "val"]:
        # Ensure the 'image' column is of type Image (for PIL processing)
        dataset[split] = dataset[split].cast_column("image", Image())
        # Apply the transform function to preprocess images on-the-fly
        dataset[split] = dataset[split].with_transform(
            get_transform(processor, args.image_size)
        )

    # --- Model ---
    # Load the pretrained model for image classification
    # 'ignore_mismatched_sizes=True' allows loading even if the number of classes differs
    model = AutoModelForImageClassification.from_pretrained(
        args.model_name,
        num_labels=len(labels),
        ignore_mismatched_sizes=True,
    )
    # Move model to the selected device
    model.to(device)

    # --- Training Arguments ---
    # Configure training parameters for the Hugging Face Trainer
    training_args = TrainingArguments(
        output_dir=args.output_dir,  # Directory to save checkpoints and logs
        per_device_train_batch_size=args.batch_size,  # Training batch size per device
        per_device_eval_batch_size=args.batch_size,  # Evaluation batch size per device
        eval_strategy="epoch",  # Evaluate at the end of each epoch
        save_strategy="epoch",  # Save checkpoint at the end of each epoch
        num_train_epochs=args.epochs,  # Total number of training epochs
        learning_rate=args.lr,  # Learning rate
        logging_dir=os.path.join(args.output_dir, "logs"),  # Directory for logs
        remove_unused_columns=False,  # Keep all columns for custom transforms
        report_to="none",  # Disable reporting to external services
        load_best_model_at_end=True,  # Restore best model at end of training
        metric_for_best_model="accuracy",  # Use accuracy to select best model
        seed=args.seed,  # Random seed
    )

    # --- Trainer ---
    # Initialize the Hugging Face Trainer with model, data, and metrics
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        compute_metrics=compute_metrics_fn(labels),
    )

    # --- Train ---
    logging.info("Starting training...")
    trainer.train()

    # --- Save Model and Processor ---
    # Save the final fine-tuned model and processor for later inference
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    logging.info(f"Fine-tuned model and processor saved to {args.output_dir}")


if __name__ == "__main__":
    # Entry point for script execution
    main()
