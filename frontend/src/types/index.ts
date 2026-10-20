export type BloomLevel = 
  | 'Remember' 
  | 'Understand' 
  | 'Apply' 
  | 'Analyze' 
  | 'Evaluate' 
  | 'Create';

export interface ClassProbabilityMap {
  Remember: number;
  Understand: number;
  Apply: number;
  Analyze: number;
  Evaluate: number;
  Create: number;
  [key: string]: number;
}

export interface QuestionPrediction {
  id: number;
  question: string;
  predicted_level: BloomLevel;
  cognitive_order: number;
  confidence: number;
  confidence_percentage: number;
  is_low_confidence: boolean;
  class_probabilities: ClassProbabilityMap;
  description: string;
  color: string;
}

export interface ChartDataPoint {
  category: BloomLevel;
  count: number;
  percentage: number;
  color: string;
  level: number;
}

export interface AnalysisSummary {
  total_questions: number;
  counts: Record<BloomLevel, number>;
  percentages: Record<BloomLevel, number>;
  dominant_level: BloomLevel | string;
  dominant_percentage: number;
  average_confidence: number;
  low_confidence_count: number;
  confidence_threshold: number;
  chart_data: ChartDataPoint[];
  higher_order_percentage: number;
  lower_order_percentage: number;
}

export interface AnalysisResponse {
  questions: QuestionPrediction[];
  summary: AnalysisSummary;
  model_status: 'TRAINED' | 'UNTRAINED_DEMO' | 'ERROR';
  is_trained: boolean;
  status_message: string;
}

export interface ExtractedQuestion {
  id: number;
  number: number;
  text: string;
}

export interface UploadResponse {
  filename: string;
  total_pages: number;
  is_scanned: boolean;
  warning?: string;
  extracted_count: number;
  questions: ExtractedQuestion[];
  sample_extracted_text?: string;
  error?: string;
}

export interface ModelInfo {
  model_name: string;
  architecture: string;
  num_classes: number;
  classes: BloomLevel[];
  metadata: Record<string, {
    level: number;
    description: string;
    keywords: string[];
    color: string;
  }>;
  status: string;
  is_trained: boolean;
  status_message: string;
  default_threshold: number;
}

export interface HealthInfo {
  status: string;
  model_status: string;
  is_trained: boolean;
  status_message: string;
  device: string;
  pytorch_version: string;
  cuda_available: boolean;
}

export interface PerClassMetric {
  precision: number;
  recall: number;
  f1_score: number;
  support: number;
}

export interface EvaluationMetrics {
  accuracy: number;
  macro_f1: number;
  macro_precision: number;
  macro_recall: number;
  weighted_f1: number;
  weighted_precision: number;
  weighted_recall: number;
  test_loss: number;
  test_samples: number;
  training_time_seconds?: number;
  best_epoch?: number;
  total_epochs?: number;
  dataset_size?: number;
  architecture?: string;
  per_class: Record<string, PerClassMetric>;
  confusion_matrix: number[][];
  confusion_matrix_percentage: number[][];
  classes: string[];
}

export interface LossHistory {
  epochs: number[];
  train_loss: number[];
  val_loss: number[];
  val_accuracy: number[];
  val_macro_f1: number[];
}

export interface EvaluationResponse {
  is_available: boolean;
  message: string;
  metrics?: EvaluationMetrics;
  loss_history?: LossHistory;
}

export interface DatasetInfo {
  is_available: boolean;
  filename: string;
  total_samples: number;
  counts: Record<string, number>;
  percentages: Record<string, number>;
  classes: string[];
  schema: string[];
}
