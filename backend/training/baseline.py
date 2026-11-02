import re
import json
import time
import argparse
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from backend.config import (
    BLOOM_CLASSES,
    BLOOM_METADATA,
    LABEL2ID,
    METRICS_DIR,
    DEFAULT_DATASET_PATH
)
from backend.training.dataset import load_and_validate_dataset, prepare_splits
from backend.training.evaluate import compute_metrics

def keyword_rule_predict(questions):
    """
    Rule baseline: first Bloom action verb found in the question decides the level.
    Questions with no keyword fall back to the most common class (Remember).
    """
    patterns = {
        LABEL2ID[c]: re.compile(r"\b(" + "|".join(BLOOM_METADATA[c]["keywords"]) + r")\w*", re.IGNORECASE)
        for c in BLOOM_CLASSES
    }
    preds = []
    for q in questions:
        best_label, best_pos = LABEL2ID["Remember"], None
        for label, pattern in patterns.items():
            match = pattern.search(q)
            if match and (best_pos is None or match.start() < best_pos):
                best_label, best_pos = label, match.start()
        preds.append(best_label)
    return np.array(preds)

def run_baselines(dataset_path=DEFAULT_DATASET_PATH, max_samples_per_class=None, seed=42):
    """
    Trains bag-of-words baselines on the exact train/test split used for DistilBERT
    (same loader, same subsampling, same seed) so the scores are directly comparable.
    """
    df = load_and_validate_dataset(dataset_path, max_samples_per_class=max_samples_per_class)
    train_df, val_df, test_df = prepare_splits(df, random_state=seed)
    print(f"Split sizes -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    y_train = train_df["label_id"].values
    y_test = test_df["label_id"].values

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
    X_train = vectorizer.fit_transform(train_df["question"])
    X_test = vectorizer.transform(test_df["question"])

    models = {
        "keyword_rules": None,
        "tfidf_logreg": LogisticRegression(max_iter=2000, random_state=seed),
        "tfidf_linear_svm": LinearSVC(random_state=seed),
    }

    results = {}
    for name, model in models.items():
        start = time.time()
        if model is None:
            preds = keyword_rule_predict(test_df["question"])
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
        metrics = compute_metrics(y_test, preds)
        results[name] = {
            "accuracy": round(float(metrics["accuracy"]), 4),
            "macro_f1": round(float(metrics["macro_f1"]), 4),
            "train_seconds": round(time.time() - start, 2),
        }
        print(f"{name:<18} Acc: {metrics['accuracy'] * 100:.2f}% | Macro-F1: {metrics['macro_f1']:.4f}")

    results["_split"] = {
        "max_samples_per_class": max_samples_per_class,
        "train": len(train_df),
        "test": len(test_df),
        "seed": seed,
    }
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bag-of-words and keyword baselines for DeepLearnQ.")
    parser.add_argument("--max_samples_per_class", type=int, default=None, help="Use the same value as the DistilBERT run")
    parser.add_argument("--dataset_path", type=str, default=str(DEFAULT_DATASET_PATH), help="Path to questions CSV")
    args = parser.parse_args()

    results = run_baselines(args.dataset_path, max_samples_per_class=args.max_samples_per_class)

    out_name = "baseline_results.json" if args.max_samples_per_class else "baseline_results_full.json"
    out_file = METRICS_DIR / out_name
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {out_file}")
