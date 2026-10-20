# DeepLearnQ Artifacts Directory

This directory stores trained model checkpoints, evaluation metrics, and training logs.

## Subdirectories:
- `model/`: Stores the fine-tuned DistilBERT model weights (`model.safetensors`), configuration (`config.json`), and tokenizer files (`tokenizer_config.json`, `vocab.txt`, etc.).
- `metrics/`: Stores machine-readable JSON evaluation results (`eval_results.json`) and training loss curves (`loss_history.json`).

These artifacts are loaded by the Flask API server during startup to serve inference requests and populate the Model Evaluation dashboard.
