import os
import json
import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from backend.config import (
    MODEL_DIR,
    METRICS_DIR,
    BASE_MODEL_NAME,
    BLOOM_CLASSES,
    BLOOM_METADATA,
    ID2LABEL,
    LABEL2ID,
    CONFIDENCE_THRESHOLD,
    MAX_SEQ_LENGTH
)

class ModelService:
    """
    Singleton service managing model loading, status, and inference.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ModelService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_dir=MODEL_DIR):
        if self._initialized:
            return

        self.model_dir = Path(model_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.status = "UNINITIALIZED"
        self.status_message = ""
        self.is_trained = False
        self.load_model()
        self._initialized = True

    def load_model(self):
        """Loads trained model checkpoint if available, or base model in demo/untrained state."""
        try:
            # Check if fine-tuned weights exist in MODEL_DIR
            has_weights = (
                (self.model_dir / "model.safetensors").exists() or
                (self.model_dir / "pytorch_model.bin").exists()
            )

            if has_weights:
                print(f"Loading fine-tuned model checkpoint from: {self.model_dir}")
                self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
                self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_dir))
                self.model.to(self.device)
                self.model.eval()
                self.status = "TRAINED"
                self.is_trained = True
                self.status_message = "Fine-tuned DistilBERT model active and ready for inference."
                print(f"[ModelService] {self.status_message} (Device: {self.device})")
            else:
                # Load base DistilBERT with un-tuned 6-class classification head
                print("No fine-tuned checkpoint found in artifacts/model. Initializing in DEMO/UNTRAINED state.")
                self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    BASE_MODEL_NAME,
                    num_labels=len(BLOOM_CLASSES),
                    id2label=ID2LABEL,
                    label2id=LABEL2ID
                )
                self.model.to(self.device)
                self.model.eval()
                self.status = "UNTRAINED_DEMO"
                self.is_trained = False
                self.status_message = (
                    "Model running in demo mode with base weights. "
                    "Run python backend/training/train.py to fine-tune on the Bloom's Taxonomy dataset."
                )
                print(f"[ModelService] {self.status_message}")

        except Exception as e:
            self.status = "ERROR"
            self.is_trained = False
            self.status_message = f"Failed to initialize model: {str(e)}"
            print(f"[ModelService Error] {self.status_message}")

    def predict_questions(self, questions, threshold=CONFIDENCE_THRESHOLD):
        """
        Classifies a list of questions into the 6 Bloom's Taxonomy levels.
        Returns detailed predictions and aggregate paper analysis.
        """
        if not questions:
            return {
                "questions": [],
                "summary": {},
                "model_status": self.status,
                "is_trained": self.is_trained
            }

        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model or tokenizer is not initialized.")

        # Clean questions
        cleaned_questions = [str(q).strip() for q in questions if str(q).strip()]
        if not cleaned_questions:
            return {
                "questions": [],
                "summary": {},
                "model_status": self.status,
                "is_trained": self.is_trained
            }

        # Tokenize batch
        inputs = self.tokenizer(
            cleaned_questions,
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()

        results = []
        counts = {category: 0 for category in BLOOM_CLASSES}
        low_confidence_count = 0
        conf_sum = 0.0

        for idx, (q_text, probs) in enumerate(zip(cleaned_questions, probabilities)):
            pred_id = int(np.argmax(probs))
            confidence = float(probs[pred_id])
            predicted_level = BLOOM_CLASSES[pred_id]
            is_low_conf = bool(confidence < threshold)

            counts[predicted_level] += 1
            conf_sum += confidence
            if is_low_conf:
                low_confidence_count += 1

            prob_dict = {
                BLOOM_CLASSES[i]: round(float(probs[i]), 4)
                for i in range(len(BLOOM_CLASSES))
            }

            meta = BLOOM_METADATA.get(predicted_level, {})

            results.append({
                "id": idx + 1,
                "question": q_text,
                "predicted_level": predicted_level,
                "cognitive_order": meta.get("level", pred_id + 1),
                "confidence": round(confidence, 4),
                "confidence_percentage": round(confidence * 100, 1),
                "is_low_confidence": is_low_conf,
                "class_probabilities": prob_dict,
                "description": meta.get("description", ""),
                "color": meta.get("color", "#64748B")
            })

        total = len(results)
        percentages = {
            category: round((count / total) * 100, 1) if total > 0 else 0.0
            for category, count in counts.items()
        }

        # Determine dominant level
        dominant_level = max(counts, key=counts.get) if total > 0 else "None"
        avg_confidence = round(conf_sum / max(1, total), 4)

        # Build chart data series
        chart_data = [
            {
                "category": cat,
                "count": counts[cat],
                "percentage": percentages[cat],
                "color": BLOOM_METADATA[cat]["color"],
                "level": BLOOM_METADATA[cat]["level"]
            }
            for cat in BLOOM_CLASSES
        ]

        summary = {
            "total_questions": total,
            "counts": counts,
            "percentages": percentages,
            "dominant_level": dominant_level,
            "dominant_percentage": percentages.get(dominant_level, 0.0),
            "average_confidence": avg_confidence,
            "low_confidence_count": low_confidence_count,
            "confidence_threshold": threshold,
            "chart_data": chart_data,
            "higher_order_percentage": round(
                percentages["Analyze"] + percentages["Evaluate"] + percentages["Create"], 1
            ),
            "lower_order_percentage": round(
                percentages["Remember"] + percentages["Understand"] + percentages["Apply"], 1
            )
        }

        return {
            "questions": results,
            "summary": summary,
            "model_status": self.status,
            "is_trained": self.is_trained,
            "status_message": self.status_message
        }

# Global instance
model_service = ModelService()
