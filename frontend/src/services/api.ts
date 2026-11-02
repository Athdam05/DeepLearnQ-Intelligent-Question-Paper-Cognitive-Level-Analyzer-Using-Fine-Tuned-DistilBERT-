import type { 
  HealthInfo, 
  ModelInfo, 
  UploadResponse, 
  AnalysisResponse, 
  EvaluationResponse, 
  DatasetInfo, 
  QuestionPrediction 
} from '../types';

const API_BASE = 'http://127.0.0.1:5080/api';

export const apiService = {
  async getHealth(): Promise<HealthInfo> {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`);
    return res.json();
  },

  async getModelInfo(): Promise<ModelInfo> {
    const res = await fetch(`${API_BASE}/model-info`);
    if (!res.ok) throw new Error(`Failed to load model info: ${res.statusText}`);
    return res.json();
  },

  async uploadPdf(file: File): Promise<UploadResponse> {
    const formData = new FormData();
    formData.append('file', file);

    const res = await fetch(`${API_BASE}/upload`, {
      method: 'POST',
      body: formData,
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Failed to upload PDF');
    return data;
  },

  async parseText(text: string): Promise<{ extracted_count: number; questions: Array<{ id: number; number: number; text: string }> }> {
    const res = await fetch(`${API_BASE}/parse-text`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Failed to parse text');
    return data;
  },

  async analyzeQuestions(questions: Array<string | { text: string }>, threshold: number = 0.55): Promise<AnalysisResponse> {
    const res = await fetch(`${API_BASE}/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ questions, threshold }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Analysis failed');
    return data;
  },

  async getEvaluation(): Promise<EvaluationResponse> {
    const res = await fetch(`${API_BASE}/evaluation`);
    if (!res.ok) throw new Error(`Failed to load evaluation: ${res.statusText}`);
    return res.json();
  },

  async getDatasetInfo(): Promise<DatasetInfo> {
    const res = await fetch(`${API_BASE}/dataset-info`);
    if (!res.ok) throw new Error(`Failed to load dataset info: ${res.statusText}`);
    return res.json();
  },

  async downloadResultsCsv(questions: QuestionPrediction[]): Promise<void> {
    const res = await fetch(`${API_BASE}/download-results`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ questions }),
    });

    if (!res.ok) throw new Error('Failed to generate CSV download');

    const blob = await res.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `DeepLearnQ_Analysis_Report_${new Date().toISOString().slice(0, 10)}.csv`;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
  }
};
