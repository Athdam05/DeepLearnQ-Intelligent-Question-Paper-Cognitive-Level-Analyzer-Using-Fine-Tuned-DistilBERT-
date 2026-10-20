import os
import io
import csv
import json
import torch
from pathlib import Path
from flask import Blueprint, request, jsonify, make_response, send_file
import pandas as pd

from backend.config import (
    BLOOM_CLASSES,
    BLOOM_METADATA,
    BASE_MODEL_NAME,
    MODEL_DIR,
    METRICS_DIR,
    DEFAULT_DATASET_PATH,
    CONFIDENCE_THRESHOLD
)
from backend.services.model_service import model_service
from backend.services.pdf_extractor import (
    extract_text_from_pdf,
    segment_questions_from_text,
    parse_manual_questions
)

api_bp = Blueprint("api", __name__)

@api_bp.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint exposing system & model readiness."""
    return jsonify({
        "status": "healthy",
        "model_status": model_service.status,
        "is_trained": model_service.is_trained,
        "status_message": model_service.status_message,
        "device": str(model_service.device),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available()
    })

@api_bp.route("/model-info", methods=["GET"])
def model_info():
    """Returns metadata about the active deep learning model and Bloom levels."""
    return jsonify({
        "model_name": BASE_MODEL_NAME,
        "architecture": "DistilBertForSequenceClassification",
        "num_classes": len(BLOOM_CLASSES),
        "classes": BLOOM_CLASSES,
        "metadata": BLOOM_METADATA,
        "status": model_service.status,
        "is_trained": model_service.is_trained,
        "status_message": model_service.status_message,
        "default_threshold": CONFIDENCE_THRESHOLD
    })

@api_bp.route("/upload", methods=["POST"])
def upload_pdf():
    """Accepts examination question paper PDF, extracts text and segments questions."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Please upload a PDF file."}), 400

    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Invalid file format. Only PDF files are supported."}), 400

    # Limit file size to 15MB
    pdf_bytes = file.read()
    if len(pdf_bytes) > 15 * 1024 * 1024:
        return jsonify({"error": "File size exceeds 15MB limit."}), 400

    if len(pdf_bytes) == 0:
        return jsonify({"error": "Uploaded PDF file is empty (0 bytes)."}), 400

    try:
        full_text, pages_info, is_scanned = extract_text_from_pdf(pdf_bytes)

        if is_scanned:
            return jsonify({
                "filename": file.filename,
                "total_pages": len(pages_info),
                "is_scanned": True,
                "warning": (
                    "This PDF appears to be a scanned image or contains no selectable digital text. "
                    "DeepLearnQ extracts text directly from PDF streams. For scanned examination papers, "
                    "an OCR preprocessor is recommended. You can also paste questions using the manual input mode."
                ),
                "questions": []
            }), 200

        questions = segment_questions_from_text(full_text)

        return jsonify({
            "filename": file.filename,
            "total_pages": len(pages_info),
            "is_scanned": False,
            "extracted_count": len(questions),
            "questions": questions,
            "sample_extracted_text": full_text[:400] + ("..." if len(full_text) > 400 else "")
        }), 200

    except Exception as e:
        return jsonify({"error": f"Failed to process PDF: {str(e)}"}), 500

@api_bp.route("/parse-text", methods=["POST"])
def parse_text():
    """Parses manually entered questions text into structured questions list."""
    data = request.get_json(silent=True) or {}
    raw_text = data.get("text", "")
    if not raw_text or not raw_text.strip():
        return jsonify({"error": "Question text cannot be empty."}), 400

    questions = parse_manual_questions(raw_text)
    return jsonify({
        "extracted_count": len(questions),
        "questions": questions
    })

@api_bp.route("/analyze", methods=["POST"])
def analyze_questions():
    """
    Classifies a list of questions into Bloom's Taxonomy cognitive levels.
    """
    data = request.get_json(silent=True) or {}
    raw_questions = data.get("questions", [])
    threshold = float(data.get("threshold", CONFIDENCE_THRESHOLD))

    if not raw_questions:
        return jsonify({"error": "Questions list is empty. Please provide at least one question."}), 400

    # Normalize input: extract string if objects are passed
    question_strings = []
    for item in raw_questions:
        if isinstance(item, dict):
            q_text = item.get("text") or item.get("question") or ""
        else:
            q_text = str(item)
        if q_text.strip():
            question_strings.append(q_text.strip())

    if not question_strings:
        return jsonify({"error": "No valid question text found in payload."}), 400

    try:
        analysis_result = model_service.predict_questions(
            question_strings, threshold=threshold
        )
        return jsonify(analysis_result), 200
    except Exception as e:
        return jsonify({"error": f"Inference error: {str(e)}"}), 500

@api_bp.route("/evaluation", methods=["GET"])
def get_evaluation():
    """Returns held-out test set evaluation metrics and training loss curves."""
    eval_file = METRICS_DIR / "eval_results.json"
    loss_file = METRICS_DIR / "loss_history.json"

    is_available = eval_file.exists()
    eval_data = None
    loss_data = None

    if is_available:
        try:
            with open(eval_file, "r", encoding="utf-8") as f:
                eval_data = json.load(f)
        except Exception as e:
            return jsonify({"error": f"Failed to read evaluation results: {str(e)}"}), 500

    if loss_file.exists():
        try:
            with open(loss_file, "r", encoding="utf-8") as f:
                loss_data = json.load(f)
        except Exception as e:
            loss_data = None

    return jsonify({
        "is_available": is_available,
        "message": (
            "Evaluation metrics measured on held-out test set."
            if is_available
            else "Model has not been fine-tuned yet. Evaluation metrics will appear once training is executed."
        ),
        "metrics": eval_data,
        "loss_history": loss_data
    })

@api_bp.route("/dataset-info", methods=["GET"])
def get_dataset_info():
    """Returns dataset verification statistics and class distribution."""
    if not DEFAULT_DATASET_PATH.exists():
        return jsonify({
            "is_available": False,
            "message": "Dataset not found in data directory."
        }), 404

    try:
        df = pd.read_csv(DEFAULT_DATASET_PATH)
        counts = df["bloom_level"].value_counts().to_dict()
        total = len(df)
        percentages = {k: round((v / total) * 100, 2) for k, v in counts.items()}

        return jsonify({
            "is_available": True,
            "filename": DEFAULT_DATASET_PATH.name,
            "total_samples": total,
            "counts": counts,
            "percentages": percentages,
            "classes": BLOOM_CLASSES,
            "schema": ["question", "bloom_level"]
        })
    except Exception as e:
        return jsonify({"error": f"Failed to load dataset info: {str(e)}"}), 500

@api_bp.route("/download-results", methods=["POST"])
def download_results():
    """Generates a downloadable CSV containing question classification results."""
    data = request.get_json(silent=True) or {}
    questions = data.get("questions", [])

    if not questions:
        return jsonify({"error": "No question results provided."}), 400

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Question_ID",
        "Question_Text",
        "Predicted_Bloom_Level",
        "Cognitive_Level_Number",
        "Confidence_Score",
        "Confidence_Percentage",
        "Low_Confidence_Flag",
        "P_Remember",
        "P_Understand",
        "P_Apply",
        "P_Analyze",
        "P_Evaluate",
        "P_Create"
    ])

    for q in questions:
        probs = q.get("class_probabilities", {})
        writer.writerow([
            q.get("id", ""),
            q.get("question", ""),
            q.get("predicted_level", ""),
            q.get("cognitive_order", ""),
            q.get("confidence", ""),
            q.get("confidence_percentage", ""),
            "YES" if q.get("is_low_confidence") else "NO",
            probs.get("Remember", ""),
            probs.get("Understand", ""),
            probs.get("Apply", ""),
            probs.get("Analyze", ""),
            probs.get("Evaluate", ""),
            probs.get("Create", "")
        ])

    csv_data = output.getvalue()
    response = make_response(csv_data)
    response.headers["Content-Disposition"] = "attachment; filename=deeplearnq_analysis_report.csv"
    response.headers["Content-Type"] = "text/csv; charset=utf-8"
    return response
