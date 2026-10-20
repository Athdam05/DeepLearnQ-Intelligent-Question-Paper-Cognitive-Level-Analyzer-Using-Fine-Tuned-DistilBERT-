import os
import json
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from backend.config import (
    BLOOM_CLASSES,
    MODEL_DIR,
    METRICS_DIR,
    BASE_MODEL_NAME,
    DEFAULT_DATASET_PATH,
    MAX_SEQ_LENGTH
)
from backend.training.dataset import (
    load_and_validate_dataset,
    prepare_splits,
    create_dataloaders
)

def evaluate_model(model, dataloader, device):
    """
    Evaluates the PyTorch/Hugging Face model on a dataloader.
    Returns predictions, ground truth labels, probabilities, and average loss.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()
            num_batches += 1

            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.tolist())

    avg_loss = total_loss / max(1, num_batches)
    return all_preds, all_labels, all_probs, avg_loss

def compute_metrics(y_true, y_pred, y_probs=None):
    """
    Computes comprehensive multi-class evaluation metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_prec, weighted_rec, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # Per-class metrics
    class_prec, class_rec, class_f1, class_supp = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=list(range(len(BLOOM_CLASSES))), zero_division=0
    )

    per_class = {}
    for i, class_name in enumerate(BLOOM_CLASSES):
        per_class[class_name] = {
            "precision": round(float(class_prec[i]), 4),
            "recall": round(float(class_rec[i]), 4),
            "f1_score": round(float(class_f1[i]), 4),
            "support": int(class_supp[i])
        }

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(BLOOM_CLASSES))))
    
    # Normalized confusion matrix (percentage per actual class)
    cm_norm = np.zeros_like(cm, dtype=float)
    for i in range(len(BLOOM_CLASSES)):
        row_sum = cm[i].sum()
        if row_sum > 0:
            cm_norm[i] = np.round(cm[i] / row_sum * 100, 2)

    return {
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "macro_precision": round(float(macro_prec), 4),
        "macro_recall": round(float(macro_rec), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "weighted_precision": round(float(weighted_prec), 4),
        "weighted_recall": round(float(weighted_rec), 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_percentage": cm_norm.tolist(),
        "classes": BLOOM_CLASSES
    }

def run_evaluation(model_path=MODEL_DIR, dataset_path=DEFAULT_DATASET_PATH, output_file=None):
    """
    Loads saved model checkpoint, evaluates on held-out test set, and saves results.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation using device: {device}")

    # Load dataset & split
    df = load_and_validate_dataset(dataset_path)
    train_df, val_df, test_df = prepare_splits(df)
    print(f"Evaluating on test set ({len(test_df)} questions)...")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)

    # Dataloader
    _, _, test_loader, _ = create_dataloaders(
        train_df, val_df, test_df, tokenizer=tokenizer, batch_size=32
    )

    y_pred, y_true, y_probs, test_loss = evaluate_model(model, test_loader, device)
    metrics = compute_metrics(y_true, y_pred, y_probs)
    metrics["test_loss"] = round(float(test_loss), 4)
    metrics["test_samples"] = len(test_df)
    metrics["model_path"] = str(model_path)

    if output_file is None:
        output_file = METRICS_DIR / "eval_results.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Evaluation complete. Metrics saved to: {output_file}")
    print(f"Test Accuracy: {metrics['accuracy'] * 100:.2f}% | Macro F1: {metrics['macro_f1']:.4f}")
    return metrics

if __name__ == "__main__":
    run_evaluation()
