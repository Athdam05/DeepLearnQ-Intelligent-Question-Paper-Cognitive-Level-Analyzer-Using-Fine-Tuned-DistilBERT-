import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = BASE_DIR / "backend"
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_DIR = ARTIFACTS_DIR / "model"
METRICS_DIR = ARTIFACTS_DIR / "metrics"

# Ensure directories exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset paths
DEFAULT_DATASET_PATH = DATA_DIR / "blooms_taxonomy_dataset.csv"
DATASET_TEMPLATE_PATH = DATA_DIR / "dataset_template.csv"

# Model configuration
BASE_MODEL_NAME = "distilbert-base-uncased"
MAX_SEQ_LENGTH = 128
DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 3e-5
DEFAULT_EPOCHS = 3
CONFIDENCE_THRESHOLD = 0.55

# Standard Bloom's Taxonomy Cognitive Levels in order
BLOOM_CLASSES = [
    "Remember",
    "Understand",
    "Apply",
    "Analyze",
    "Evaluate",
    "Create"
]

LABEL2ID = {label: i for i, label in enumerate(BLOOM_CLASSES)}
ID2LABEL = {i: label for i, label in enumerate(BLOOM_CLASSES)}

# Level metadata for UI and explanations
BLOOM_METADATA = {
    "Remember": {
        "level": 1,
        "description": "Recall facts, basic concepts, definitions, and formulas.",
        "keywords": ["define", "list", "recall", "state", "identify", "name"],
        "color": "#3B82F6"  # Blue
    },
    "Understand": {
        "level": 2,
        "description": "Explain ideas, interpret concepts, summarize, or classify.",
        "keywords": ["explain", "describe", "discuss", "summarize", "interpret", "classify"],
        "color": "#06B6D4"  # Cyan
    },
    "Apply": {
        "level": 3,
        "description": "Use information in new situations, solve problems, execute methods.",
        "keywords": ["calculate", "solve", "implement", "demonstrate", "apply", "execute"],
        "color": "#10B981"  # Emerald Green
    },
    "Analyze": {
        "level": 4,
        "description": "Draw connections among ideas, differentiate, organize, or dissect.",
        "keywords": ["compare", "contrast", "differentiate", "distinguish", "analyze", "examine"],
        "color": "#F59E0B"  # Amber
    },
    "Evaluate": {
        "level": 5,
        "description": "Justify a decision, critique arguments, defend choices, or assess.",
        "keywords": ["critique", "evaluate", "judge", "assess", "justify", "recommend"],
        "color": "#F97316"  # Orange
    },
    "Create": {
        "level": 6,
        "description": "Produce new or original work, design architectures, formulate solutions.",
        "keywords": ["design", "develop", "formulate", "construct", "compose", "create"],
        "color": "#8B5CF6"  # Violet/Purple
    }
}
