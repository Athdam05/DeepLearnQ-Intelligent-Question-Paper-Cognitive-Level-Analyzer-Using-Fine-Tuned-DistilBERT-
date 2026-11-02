import os
import sys
from pathlib import Path
from flask import Flask, jsonify
from flask_cors import CORS

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.routes.api import api_bp

def create_app():
    app = Flask(__name__)
    
    # Configure CORS for local frontend dev server (Vite default is 5173/3000)
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # Max upload limit (20MB)
    app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

    # Register API blueprint
    app.register_blueprint(api_bp, url_prefix="/api")

    @app.route("/")
    def index():
        return jsonify({
            "project": "DeepLearnQ — Intelligent Question Paper Analyzer",
            "version": "1.0.0",
            "description": "Deep learning system for classifying examination questions into Bloom's Taxonomy cognitive levels.",
            "endpoints": [
                "/api/health",
                "/api/model-info",
                "/api/upload",
                "/api/analyze",
                "/api/evaluation",
                "/api/dataset-info",
                "/api/download-results"
            ]
        })

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "Resource not found"}), 404

    @app.errorhandler(500)
    def internal_error(e):
        return jsonify({"error": "Internal server error occurred"}), 500

    @app.errorhandler(413)
    def file_too_large(e):
        return jsonify({"error": "File size exceeds the 20MB maximum allowed limit"}), 413

    return app

app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5080))
    print(f"Starting DeepLearnQ Backend Server on http://localhost:{port}...")
    app.run(host="127.0.0.1", port=port, debug=False)
