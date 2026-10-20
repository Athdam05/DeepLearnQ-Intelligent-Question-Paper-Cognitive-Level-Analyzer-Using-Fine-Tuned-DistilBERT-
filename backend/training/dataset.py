import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from backend.config import (
    BLOOM_CLASSES,
    LABEL2ID,
    ID2LABEL,
    MAX_SEQ_LENGTH,
    BASE_MODEL_NAME,
    DEFAULT_DATASET_PATH
)

class BloomsDataset(Dataset):
    """PyTorch Dataset for Bloom's Taxonomy questions."""
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

def normalize_label(label: str) -> str:
    """Normalize label string to standard Bloom category name."""
    if not isinstance(label, str):
        return None
    cleaned = label.strip().capitalize()
    
    mapping = {
        "Remember": "Remember", "Remembering": "Remember", "Knowledge": "Remember", "Bt1": "Remember",
        "Understand": "Understand", "Understanding": "Understand", "Comprehension": "Understand", "Bt2": "Understand",
        "Apply": "Apply", "Applying": "Apply", "Application": "Apply", "Bt3": "Apply",
        "Analyze": "Analyze", "Analyzing": "Analyze", "Analysis": "Analyze", "Bt4": "Analyze",
        "Evaluate": "Evaluate", "Evaluating": "Evaluate", "Evaluation": "Evaluate", "Bt5": "Evaluate",
        "Create": "Create", "Creating": "Create", "Synthesis": "Create", "Bt6": "Create"
    }
    return mapping.get(cleaned, None)

def load_and_validate_dataset(csv_path=DEFAULT_DATASET_PATH, max_samples_per_class=None):
    """
    Loads and validates the questions dataset.
    Ensures question and bloom_level columns are present and clean.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    
    # Check column names
    col_map = {col.lower().strip(): col for col in df.columns}
    
    q_col = None
    for cand in ["question", "questions", "text", "query"]:
        if cand in col_map:
            q_col = col_map[cand]
            break

    lbl_col = None
    for cand in ["bloom_level", "bloom", "category", "label", "level", "taxonomy"]:
        if cand in col_map:
            lbl_col = col_map[cand]
            break

    if not q_col or not lbl_col:
        raise ValueError(
            f"Invalid CSV structure. Expected columns 'question' and 'bloom_level'. Found: {list(df.columns)}"
        )

    clean_df = pd.DataFrame()
    clean_df["question"] = df[q_col].astype(str).str.strip()
    clean_df["bloom_level"] = df[lbl_col].astype(str).apply(normalize_label)

    # Filter invalid records
    clean_df = clean_df.dropna(subset=["bloom_level"])
    clean_df = clean_df[clean_df["question"].str.len() >= 4]
    clean_df = clean_df.drop_duplicates(subset=["question"]).reset_index(drop=True)

    if len(clean_df) == 0:
        raise ValueError("No valid questions found after cleaning.")

    # Optional balanced subsampling per class (useful for fast CPU training runs)
    if max_samples_per_class and max_samples_per_class > 0:
        sampled_dfs = []
        for c in BLOOM_CLASSES:
            class_df = clean_df[clean_df["bloom_level"] == c]
            if len(class_df) > 0:
                n = min(len(class_df), max_samples_per_class)
                sampled_dfs.append(class_df.sample(n=n, random_state=42))
        clean_df = pd.concat(sampled_dfs).sample(frac=1.0, random_state=42).reset_index(drop=True)

    clean_df["label_id"] = clean_df["bloom_level"].map(LABEL2ID)
    return clean_df

def prepare_splits(df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """
    Performs stratified train/validation/test split.
    Prevents data leakage.
    """
    total = train_ratio + val_ratio + test_ratio
    train_frac = train_ratio / total
    val_frac = val_ratio / total
    test_frac = test_ratio / total

    # First split off train
    train_df, temp_df = train_test_split(
        df,
        train_size=train_frac,
        stratify=df["label_id"],
        random_state=random_state
    )

    # Split remaining into val and test
    val_relative = val_frac / (val_frac + test_frac)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_relative,
        stratify=temp_df["label_id"],
        random_state=random_state
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True)
    )

def create_dataloaders(train_df, val_df, test_df, tokenizer=None, batch_size=16, max_length=MAX_SEQ_LENGTH):
    """
    Tokenizes question texts and creates PyTorch DataLoaders.
    """
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

    def tokenize_data(sub_df):
        texts = sub_df["question"].tolist()
        labels = sub_df["label_id"].tolist()
        encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors=None
        )
        return BloomsDataset(encodings, labels)

    train_dataset = tokenize_data(train_df)
    val_dataset = tokenize_data(val_df)
    test_dataset = tokenize_data(test_df)

    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, tokenizer
