import io
import os
import sys
from pathlib import Path
import pytest
import numpy as np
import fitz  # PyMuPDF

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.app import app
from backend.config import BLOOM_CLASSES, LABEL2ID, ID2LABEL
from backend.training.dataset import normalize_label, load_and_validate_dataset
from backend.training.evaluate import compute_metrics
from backend.services.pdf_extractor import (
    extract_text_from_pdf,
    segment_questions_from_text,
    parse_manual_questions
)

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_health_endpoint(client):
    """Test /api/health returns 200 and valid structure."""
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "healthy"
    assert "model_status" in data
    assert "device" in data
    assert "pytorch_version" in data

def test_model_info_endpoint(client):
    """Test /api/model-info returns 6 Bloom classes and metadata."""
    res = client.get("/api/model-info")
    assert res.status_code == 200
    data = res.get_json()
    assert data["num_classes"] == 6
    assert data["classes"] == BLOOM_CLASSES
    assert "metadata" in data
    for cls in BLOOM_CLASSES:
        assert cls in data["metadata"]

def test_dataset_info_endpoint(client):
    """Test /api/dataset-info returns valid sample counts."""
    res = client.get("/api/dataset-info")
    assert res.status_code == 200
    data = res.get_json()
    assert data["is_available"] is True
    assert data["total_samples"] > 0
    assert "counts" in data

def test_empty_analyze_request(client):
    """Test /api/analyze with empty question list returns 400 error."""
    res = client.post("/api/analyze", json={"questions": []})
    assert res.status_code == 400
    data = res.get_json()
    assert "error" in data

def test_invalid_blank_questions(client):
    """Test /api/analyze with whitespace-only questions returns 400."""
    res = client.post("/api/analyze", json={"questions": ["   ", "\n\t"]})
    assert res.status_code == 400

def test_invalid_file_upload_type(client):
    """Test /api/upload with non-PDF file rejects with 400."""
    data = {
        "file": (io.BytesIO(b"Not a pdf"), "sample.txt")
    }
    res = client.post("/api/upload", data=data, content_type="multipart/form-data")
    assert res.status_code == 400
    assert "Invalid file format" in res.get_json()["error"]

def test_pdf_extraction_and_segmentation():
    """Test PyMuPDF extraction and question segmentation on a programmatic PDF."""
    doc = fitz.open()
    page = doc.new_page()
    text = (
        "Mid-Term Examination 2026\n"
        "Department of Computer Science\n"
        "1. Define supervised learning and give two examples.\n"
        "2. Explain the difference between bagging and boosting.\n"
        "3. Design an architecture for real-time question classification."
    )
    page.insert_text((50, 72), text)
    pdf_bytes = doc.write()
    doc.close()

    full_text, pages, is_scanned = extract_text_from_pdf(pdf_bytes)
    assert not is_scanned
    assert len(pages) == 1

    questions = segment_questions_from_text(full_text)
    assert len(questions) == 3
    assert "supervised learning" in questions[0]["text"].lower()
    assert "difference between" in questions[1]["text"].lower()

def test_scanned_pdf_detection():
    """Test that a blank/empty PDF flags scanned warning."""
    doc = fitz.open()
    doc.new_page()  # Blank page with zero text
    pdf_bytes = doc.write()
    doc.close()

    full_text, pages, is_scanned = extract_text_from_pdf(pdf_bytes)
    assert is_scanned is True

def test_manual_text_parser():
    """Test manual question parsing from multiline string."""
    raw = (
        "1. What is gradient descent?\n"
        "2. Calculate the determinant of the matrix.\n"
        "3. Compare CNNs and Transformers."
    )
    res = parse_manual_questions(raw)
    assert len(res) == 3
    assert res[0]["text"] == "What is gradient descent?"

def test_normalize_labels():
    """Test label normalization helper."""
    assert normalize_label("remember") == "Remember"
    assert normalize_label("bt1") == "Remember"
    assert normalize_label("comprehension") == "Understand"
    assert normalize_label("create") == "Create"
    assert normalize_label("invalid_label") is None

def test_compute_metrics():
    """Test multi-class metric calculation."""
    y_true = [0, 1, 2, 3, 4, 5]
    y_pred = [0, 1, 2, 3, 4, 5]
    metrics = compute_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert len(metrics["confusion_matrix"]) == 6

def test_real_inference_prediction(client):
    """Test actual inference pipeline on sample questions."""
    payload = {
        "questions": [
            "Define the second law of thermodynamics.",
            "Explain the architecture of a transformer network.",
            "Design a microservices pipeline for question paper analysis."
        ],
        "threshold": 0.55
    }
    res = client.post("/api/analyze", json=payload)
    assert res.status_code == 200
    data = res.get_json()
    assert "questions" in data
    assert len(data["questions"]) == 3
    assert "summary" in data

    q1 = data["questions"][0]
    assert "predicted_level" in q1
    assert q1["predicted_level"] in BLOOM_CLASSES
    assert "confidence" in q1
    assert 0.0 <= q1["confidence"] <= 1.0
    assert "class_probabilities" in q1
    assert len(q1["class_probabilities"]) == 6

    # Verify summary structure
    s = data["summary"]
    assert s["total_questions"] == 3
    assert "dominant_level" in s
    assert "chart_data" in s

def test_evaluation_endpoint(client):
    """Test /api/evaluation returns saved test metrics."""
    res = client.get("/api/evaluation")
    assert res.status_code == 200
    data = res.get_json()
    assert data["is_available"] is True
    assert "metrics" in data
    m = data["metrics"]
    assert "accuracy" in m
    assert "macro_f1" in m
    assert "confusion_matrix" in m

def test_download_results_csv(client):
    """Test /api/download-results produces valid CSV headers and data."""
    sample_preds = [
        {
            "id": 1,
            "question": "What is machine learning?",
            "predicted_level": "Remember",
            "cognitive_order": 1,
            "confidence": 0.88,
            "confidence_percentage": 88.0,
            "is_low_confidence": False,
            "class_probabilities": {"Remember": 0.88, "Understand": 0.05, "Apply": 0.02, "Analyze": 0.02, "Evaluate": 0.01, "Create": 0.02}
        }
    ]
    res = client.post("/api/download-results", json={"questions": sample_preds})
    assert res.status_code == 200
    assert res.headers["Content-Type"].startswith("text/csv")
    csv_text = res.get_data(as_text=True)
    assert "Question_ID,Question_Text,Predicted_Bloom_Level" in csv_text
    assert "What is machine learning?" in csv_text
