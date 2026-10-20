import os
import sys
import time
import json
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)

from backend.config import (
    BASE_MODEL_NAME,
    MODEL_DIR,
    METRICS_DIR,
    DEFAULT_DATASET_PATH,
    BLOOM_CLASSES,
    LABEL2ID,
    ID2LABEL,
    MAX_SEQ_LENGTH,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_EPOCHS
)
from backend.training.dataset import (
    load_and_validate_dataset,
    prepare_splits,
    create_dataloaders
)
from backend.training.evaluate import evaluate_model, compute_metrics

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train(
    dataset_path=DEFAULT_DATASET_PATH,
    output_dir=MODEL_DIR,
    epochs=DEFAULT_EPOCHS,
    batch_size=DEFAULT_BATCH_SIZE,
    learning_rate=DEFAULT_LEARNING_RATE,
    max_length=MAX_SEQ_LENGTH,
    max_samples_per_class=None,
    seed=42
):
    """
    Fine-tunes DistilBERT for Bloom's Taxonomy 6-class classification.
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Starting DeepLearnQ DistilBERT Fine-Tuning ===")
    print(f"Device: {device}")
    print(f"Epochs: {epochs} | Batch size: {batch_size} | Learning rate: {learning_rate}")
    if max_samples_per_class:
        print(f"Subsampling: max {max_samples_per_class} samples per class")

    # 1. Load & split dataset
    print(f"Loading dataset from: {dataset_path}")
    df = load_and_validate_dataset(dataset_path, max_samples_per_class=max_samples_per_class)
    print(f"Total usable records: {len(df)}")
    print("Class distribution:\n", df["bloom_level"].value_counts().to_dict())

    train_df, val_df, test_df = prepare_splits(df, random_state=seed)
    print(f"Split sizes -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # 2. Tokenizer & DataLoaders
    print(f"Initializing tokenizer: {BASE_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    train_loader, val_loader, test_loader, _ = create_dataloaders(
        train_df, val_df, test_df,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length
    )

    # 3. Model Architecture
    print(f"Initializing DistilBERT classification model (6 classes)...")
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_NAME,
        num_labels=len(BLOOM_CLASSES),
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )
    model.to(device)

    # 4. Optimizer & LR Scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    
    total_training_steps = len(train_loader) * epochs
    warmup_steps = int(0.1 * total_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps
    )

    # Tracking metrics
    loss_history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_macro_f1": [],
        "epochs": []
    }

    best_val_f1 = -1.0
    best_epoch = 0

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    start_time = time.time()

    # 5. Training Loop
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        running_train_loss = 0.0
        train_steps = 0

        print(f"\n--- Epoch {epoch}/{epochs} ---")
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_train_loss += loss.item()
            train_steps += 1

            if (step + 1) % max(1, len(train_loader) // 5) == 0 or (step + 1) == len(train_loader):
                print(f"Step {step + 1}/{len(train_loader)} - Batch Loss: {loss.item():.4f}")

        avg_train_loss = running_train_loss / max(1, train_steps)

        # Validation step
        val_preds, val_labels, _, avg_val_loss = evaluate_model(model, val_loader, device)
        val_metrics = compute_metrics(val_labels, val_preds)

        epoch_duration = time.time() - epoch_start
        print(
            f"Epoch {epoch} finished in {epoch_duration:.1f}s | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {val_metrics['accuracy'] * 100:.2f}% | "
            f"Val Macro-F1: {val_metrics['macro_f1']:.4f}"
        )

        loss_history["epochs"].append(epoch)
        loss_history["train_loss"].append(round(avg_train_loss, 4))
        loss_history["val_loss"].append(round(avg_val_loss, 4))
        loss_history["val_accuracy"].append(round(val_metrics["accuracy"], 4))
        loss_history["val_macro_f1"].append(round(val_metrics["macro_f1"], 4))

        # Checkpoint: save best model based on val_macro_f1
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            print(f"[*] New best validation Macro-F1: {best_val_f1:.4f}. Saving checkpoint to {output_dir}")
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

    total_duration = time.time() - start_time
    print(f"\nTraining completed in {total_duration / 60:.2f} minutes.")
    print(f"Best Epoch: {best_epoch} with Val Macro-F1: {best_val_f1:.4f}")

    # Save loss history
    history_file = METRICS_DIR / "loss_history.json"
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(loss_history, f, indent=2)
    print(f"Loss history saved to: {history_file}")

    # 6. Final Evaluation on Held-Out Test Set
    print("\n--- Evaluating Best Model on Held-Out Test Set ---")
    best_model = AutoModelForSequenceClassification.from_pretrained(output_dir)
    best_model.to(device)

    test_preds, test_labels, test_probs, test_loss = evaluate_model(best_model, test_loader, device)
    test_metrics = compute_metrics(test_labels, test_preds, test_probs)
    test_metrics["test_loss"] = round(test_loss, 4)
    test_metrics["test_samples"] = len(test_df)
    test_metrics["training_time_seconds"] = round(total_duration, 1)
    test_metrics["best_epoch"] = best_epoch
    test_metrics["total_epochs"] = epochs
    test_metrics["dataset_size"] = len(df)
    test_metrics["architecture"] = "DistilBERT for Sequence Classification (distilbert-base-uncased)"

    eval_file = METRICS_DIR / "eval_results.json"
    with open(eval_file, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    print(f"Test Evaluation Complete!")
    print(f"Test Accuracy: {test_metrics['accuracy'] * 100:.2f}%")
    print(f"Macro-F1: {test_metrics['macro_f1']:.4f} | Weighted-F1: {test_metrics['weighted_f1']:.4f}")
    print(f"Results saved to: {eval_file}")

    return test_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT on Bloom's Taxonomy dataset.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE, help="Learning rate")
    parser.add_argument("--max_samples_per_class", type=int, default=None, help="Max samples per class for quick CPU training")
    parser.add_argument("--dataset_path", type=str, default=str(DEFAULT_DATASET_PATH), help="Path to questions CSV")
    
    args = parser.parse_args()
    train(
        dataset_path=args.dataset_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_samples_per_class=args.max_samples_per_class
    )
