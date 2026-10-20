import React, { useState } from 'react';
import { 
  Upload, 
  FileText, 
  FileEdit, 
  Trash2, 
  Plus, 
  Play, 
  AlertTriangle, 
  CheckCircle, 
  Loader2, 
  SlidersHorizontal,
  Info
} from 'lucide-react';
import { apiService } from '../services/api';
import { ExtractedQuestion, AnalysisResponse } from '../types';

interface AnalyzePageProps {
  initialMode?: 'upload' | 'manual';
  onAnalysisComplete: (result: AnalysisResponse) => void;
}

export const AnalyzePage: React.FC<AnalyzePageProps> = ({ 
  initialMode = 'upload', 
  onAnalysisComplete 
}) => {
  const [mode, setMode] = useState<'upload' | 'manual'>(initialMode);
  
  // Upload state
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [uploadWarning, setUploadWarning] = useState<string | null>(null);

  // Manual input state
  const [manualText, setManualText] = useState<string>('');
  const [isParsing, setIsParsing] = useState<boolean>(false);

  // Questions workspace
  const [questions, setQuestions] = useState<ExtractedQuestion[]>([]);
  const [threshold, setThreshold] = useState<number>(0.55);
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Sample questions to test with 1-click
  const sampleQuestions = [
    "Define the first law of thermodynamics and state its mathematical expression.",
    "Explain the architectural differences between recurrent neural networks and transformers.",
    "Calculate the resultant electric field at the midpoint between two point charges of 5 microcoulombs separated by 10 cm.",
    "Compare and contrast supervised learning with reinforcement learning in terms of reward signal and data efficiency.",
    "Critique the trade-offs of using NoSQL databases instead of traditional ACID-compliant relational databases.",
    "Design an end-to-end question paper cognitive analyzer using deep learning and a microservices backend."
  ];

  const handleFileDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.name.toLowerCase().endsWith('.pdf')) {
        setSelectedFile(file);
        setUploadWarning(null);
        setErrorMessage(null);
      } else {
        setErrorMessage('Only PDF files are supported.');
      }
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      if (file.name.toLowerCase().endsWith('.pdf')) {
        setSelectedFile(file);
        setUploadWarning(null);
        setErrorMessage(null);
      } else {
        setErrorMessage('Only PDF files are supported.');
      }
    }
  };

  const handleUploadPdf = async () => {
    if (!selectedFile) return;
    setIsUploading(true);
    setErrorMessage(null);
    setUploadWarning(null);

    try {
      const res = await apiService.uploadPdf(selectedFile);
      if (res.is_scanned && res.warning) {
        setUploadWarning(res.warning);
      }
      if (res.questions && res.questions.length > 0) {
        setQuestions(res.questions);
      } else if (!res.is_scanned) {
        setErrorMessage('No clearly numbered questions were detected in this PDF. You can paste them manually.');
      }
    } catch (err: any) {
      setErrorMessage(err.message || 'Failed to extract text from PDF.');
    } finally {
      setIsUploading(false);
    }
  };

  const handleParseManual = async () => {
    if (!manualText.trim()) {
      setErrorMessage('Please enter question text before parsing.');
      return;
    }
    setIsParsing(true);
    setErrorMessage(null);

    try {
      const res = await apiService.parseText(manualText);
      if (res.questions && res.questions.length > 0) {
        setQuestions(res.questions);
      } else {
        setErrorMessage('Could not extract any questions from the entered text.');
      }
    } catch (err: any) {
      setErrorMessage(err.message || 'Failed to parse text.');
    } finally {
      setIsParsing(false);
    }
  };

  const handleLoadSamples = () => {
    const sampleItems = sampleQuestions.map((q, idx) => ({
      id: idx + 1,
      number: idx + 1,
      text: q
    }));
    setQuestions(sampleItems);
    setErrorMessage(null);
    setUploadWarning(null);
  };

  const handleQuestionTextChange = (id: number, newText: string) => {
    setQuestions(prev => prev.map(q => q.id === id ? { ...q, text: newText } : q));
  };

  const handleDeleteQuestion = (id: number) => {
    setQuestions(prev => prev.filter(q => q.id !== id));
  };

  const handleAddQuestion = () => {
    const nextId = questions.length > 0 ? Math.max(...questions.map(q => q.id)) + 1 : 1;
    setQuestions(prev => [
      ...prev,
      { id: nextId, number: prev.length + 1, text: '' }
    ]);
  };

  const handleRunAnalysis = async () => {
    const validQuestions = questions.filter(q => q.text.trim().length >= 4);
    if (validQuestions.length === 0) {
      setErrorMessage('Please provide at least one valid question with at least 4 characters.');
      return;
    }

    setIsAnalyzing(true);
    setErrorMessage(null);

    try {
      const result = await apiService.analyzeQuestions(validQuestions, threshold);
      onAnalysisComplete(result);
    } catch (err: any) {
      setErrorMessage(err.message || 'Error occurred during classification inference.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="container" style={{ padding: '2.5rem 1.5rem', display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      
      {/* Header */}
      <div>
        <h1 style={{ fontSize: '1.75rem', fontWeight: 800, color: 'var(--text-main)', letterSpacing: '-0.02em' }}>
          Analyze Question Paper
        </h1>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', marginTop: '0.35rem' }}>
          Upload a syllabus examination paper PDF or paste questions to run DistilBERT cognitive taxonomy inference.
        </p>
      </div>

      {/* Error message */}
      {errorMessage && (
        <div style={{
          padding: '1rem',
          borderRadius: 'var(--radius-md)',
          background: '#fef2f2',
          border: '1px solid #fecaca',
          color: '#b91c1c',
          display: 'flex',
          alignItems: 'center',
          gap: '0.75rem',
          fontSize: '0.875rem'
        }}>
          <AlertTriangle size={18} />
          <span>{errorMessage}</span>
        </div>
      )}

      {/* Mode selection tabs */}
      <div style={{
        display: 'flex',
        borderBottom: '1px solid var(--border-light)',
        gap: '1rem',
        paddingBottom: '0.5rem'
      }}>
        <button
          onClick={() => { setMode('upload'); setErrorMessage(null); }}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            padding: '0.5rem 1rem',
            fontWeight: 600,
            fontSize: '0.925rem',
            color: mode === 'upload' ? 'var(--primary)' : 'var(--text-muted)',
            borderBottom: mode === 'upload' ? '2px solid var(--primary)' : 'none',
            marginBottom: '-0.5rem'
          }}
        >
          <Upload size={18} />
          PDF Document Upload
        </button>

        <button
          onClick={() => { setMode('manual'); setErrorMessage(null); }}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            padding: '0.5rem 1rem',
            fontWeight: 600,
            fontSize: '0.925rem',
            color: mode === 'manual' ? 'var(--primary)' : 'var(--text-muted)',
            borderBottom: mode === 'manual' ? '2px solid var(--primary)' : 'none',
            marginBottom: '-0.5rem'
          }}
        >
          <FileEdit size={18} />
          Manual Question Entry
        </button>
      </div>

      {/* Upload or Manual input card */}
      {mode === 'upload' ? (
        <div className="card">
          <div
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleFileDrop}
            style={{
              border: '2px dashed var(--border-medium)',
              borderRadius: 'var(--radius-lg)',
              padding: '3rem 2rem',
              textAlign: 'center',
              backgroundColor: '#fafafa',
              transition: 'border-color 0.2s',
              cursor: 'pointer'
            }}
            onClick={() => document.getElementById('pdf-upload-input')?.click()}
          >
            <input 
              id="pdf-upload-input" 
              type="file" 
              accept=".pdf" 
              style={{ display: 'none' }} 
              onChange={handleFileChange}
            />
            <div style={{
              width: '3.5rem',
              height: '3.5rem',
              borderRadius: '50%',
              backgroundColor: '#e0e7ff',
              color: '#4f46e5',
              display: 'inline-flex',
              alignItems: 'center',
              justifyContent: 'center',
              marginBottom: '1rem'
            }}>
              <Upload size={28} />
            </div>
            <h3 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '0.35rem' }}>
              {selectedFile ? selectedFile.name : 'Choose a PDF examination paper or drag and drop'}
            </h3>
            <p style={{ fontSize: '0.825rem', color: 'var(--text-muted)' }}>
              Supports university exam PDFs up to 15MB with PyMuPDF text stream extraction
            </p>

            {selectedFile && (
              <div style={{ marginTop: '1rem', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem', fontSize: '0.85rem', color: '#047857' }}>
                <CheckCircle size={16} />
                <span>Selected: {selectedFile.name} ({(selectedFile.size / 1024).toFixed(1)} KB)</span>
              </div>
            )}
          </div>

          {uploadWarning && (
            <div style={{
              marginTop: '1.25rem',
              padding: '1rem',
              borderRadius: 'var(--radius-md)',
              background: '#fffbeb',
              border: '1px solid #fde68a',
              color: '#92400e',
              display: 'flex',
              alignItems: 'flex-start',
              gap: '0.75rem',
              fontSize: '0.85rem',
              lineHeight: 1.5
            }}>
              <Info size={18} style={{ flexShrink: 0, marginTop: '2px' }} />
              <div>
                <strong>Scanned PDF Notice:</strong> {uploadWarning}
              </div>
            </div>
          )}

          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '1.5rem' }}>
            <button
              onClick={handleLoadSamples}
              className="btn btn-secondary"
              style={{ fontSize: '0.825rem' }}
            >
              Load 6 Illustrative College Questions
            </button>

            <button
              onClick={handleUploadPdf}
              disabled={!selectedFile || isUploading}
              className="btn btn-primary"
              style={{ opacity: !selectedFile || isUploading ? 0.6 : 1 }}
            >
              {isUploading ? (
                <>
                  <Loader2 size={16} className="animate-spin" />
                  Extracting Text with PyMuPDF...
                </>
              ) : (
                <>
                  <FileText size={16} />
                  Extract Questions from PDF
                </>
              )}
            </button>
          </div>
        </div>
      ) : (
        <div className="card">
          <label style={{ display: 'block', fontWeight: 600, fontSize: '0.9rem', marginBottom: '0.5rem' }}>
            Paste Examination Questions:
          </label>
          <textarea
            rows={7}
            value={manualText}
            onChange={(e) => setManualText(e.target.value)}
            placeholder={`Paste your exam questions here. Example:&#10;1. Define the first law of thermodynamics.&#10;2. Explain the mechanism of photosynthesis.&#10;3. Calculate the electric flux through a closed Gaussian surface.`}
            style={{
              width: '100%',
              padding: '0.85rem',
              borderRadius: 'var(--radius-md)',
              border: '1px solid var(--border-medium)',
              lineHeight: 1.5,
              resize: 'vertical'
            }}
          />

          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '1rem' }}>
            <button
              onClick={handleLoadSamples}
              className="btn btn-secondary"
              style={{ fontSize: '0.825rem' }}
            >
              Load 6 Illustrative College Questions
            </button>

            <button
              onClick={handleParseManual}
              disabled={isParsing || !manualText.trim()}
              className="btn btn-primary"
              style={{ opacity: isParsing || !manualText.trim() ? 0.6 : 1 }}
            >
              {isParsing ? (
                <>
                  <Loader2 size={16} className="animate-spin" />
                  Parsing Questions...
                </>
              ) : (
                <>
                  <FileEdit size={16} />
                  Parse Questions into Review Table
                </>
              )}
            </button>
          </div>
        </div>
      )}

      {/* Review & Edit Workspace */}
      {questions.length > 0 && (
        <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          
          <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'space-between', alignItems: 'center', gap: '1rem' }}>
            <div>
              <h2 style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--text-main)' }}>
                Review & Edit Extracted Questions ({questions.length})
              </h2>
              <p style={{ fontSize: '0.825rem', color: 'var(--text-muted)' }}>
                Inspect, modify question text, delete unwanted items, or insert custom questions before classification.
              </p>
            </div>

            {/* Threshold slider */}
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.75rem',
              background: 'var(--bg-subtle)',
              padding: '0.45rem 0.85rem',
              borderRadius: 'var(--radius-md)',
              border: '1px solid var(--border-light)'
            }}>
              <SlidersHorizontal size={16} color="var(--text-muted)" />
              <span style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-muted)' }}>
                Review Threshold:
              </span>
              <input 
                type="range" 
                min={0.30} 
                max={0.90} 
                step={0.05} 
                value={threshold} 
                onChange={(e) => setThreshold(parseFloat(e.target.value))}
                style={{ width: '90px' }}
              />
              <span style={{ fontSize: '0.8rem', fontWeight: 700, minWidth: '35px' }}>
                {(threshold * 100).toFixed(0)}%
              </span>
            </div>
          </div>

          {/* Questions list */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', maxHeight: '480px', overflowY: 'auto', paddingRight: '0.35rem' }}>
            {questions.map((q, idx) => (
              <div 
                key={q.id}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.75rem',
                  padding: '0.65rem 0.85rem',
                  border: '1px solid var(--border-light)',
                  borderRadius: 'var(--radius-md)',
                  background: '#ffffff'
                }}
              >
                <div style={{
                  width: '28px',
                  height: '28px',
                  borderRadius: '50%',
                  background: '#f1f5f9',
                  color: '#475569',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: '0.8rem',
                  fontWeight: 700,
                  flexShrink: 0
                }}>
                  {idx + 1}
                </div>

                <input 
                  type="text"
                  value={q.text}
                  onChange={(e) => handleQuestionTextChange(q.id, e.target.value)}
                  placeholder="Enter question text..."
                  style={{
                    flex: 1,
                    padding: '0.45rem 0.65rem',
                    borderRadius: 'var(--radius-sm)',
                    border: '1px solid var(--border-light)',
                    outline: 'none'
                  }}
                />

                <button
                  onClick={() => handleDeleteQuestion(q.id)}
                  className="btn btn-danger"
                  style={{ padding: '0.45rem', borderRadius: 'var(--radius-sm)' }}
                  title="Remove question"
                >
                  <Trash2 size={16} />
                </button>
              </div>
            ))}
          </div>

          {/* Action buttons */}
          <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'space-between', alignItems: 'center', paddingTop: '1rem', borderTop: '1px solid var(--border-light)', gap: '1rem' }}>
            <button
              onClick={handleAddQuestion}
              className="btn btn-secondary"
              style={{ fontSize: '0.85rem' }}
            >
              <Plus size={16} />
              Add Another Question
            </button>

            <button
              onClick={handleRunAnalysis}
              disabled={isAnalyzing}
              className="btn btn-primary"
              style={{ padding: '0.75rem 1.75rem', fontSize: '0.95rem', fontWeight: 600 }}
            >
              {isAnalyzing ? (
                <>
                  <Loader2 size={18} className="animate-spin" />
                  Running DistilBERT Multi-Class Inference...
                </>
              ) : (
                <>
                  <Play size={18} fill="currentColor" />
                  Classify Question Paper
                </>
              )}
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
