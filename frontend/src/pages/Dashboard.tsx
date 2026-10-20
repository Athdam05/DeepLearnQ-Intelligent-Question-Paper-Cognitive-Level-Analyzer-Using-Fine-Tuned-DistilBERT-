import React from 'react';
import { 
  Upload, 
  FileEdit, 
  Cpu, 
  Database, 
  CheckCircle, 
  ArrowRight, 
  BrainCircuit,
  BarChart2
} from 'lucide-react';
import { HealthInfo, ModelInfo, DatasetInfo, AnalysisResponse } from '../types';

interface DashboardProps {
  health: HealthInfo | null;
  modelInfo: ModelInfo | null;
  datasetInfo: DatasetInfo | null;
  lastAnalysis: AnalysisResponse | null;
  onNavigate: (tab: string, mode?: 'upload' | 'manual') => void;
}

export const Dashboard: React.FC<DashboardProps> = ({
  health,
  modelInfo,
  datasetInfo,
  lastAnalysis,
  onNavigate
}) => {
  const bloomLevels = [
    { level: 1, name: 'Remember', verb: 'Define, List, Recall, State, Identify', color: '#2563eb', desc: 'Retrieving relevant knowledge from long-term memory.' },
    { level: 2, name: 'Understand', verb: 'Explain, Describe, Summarize, Interpret', color: '#0891b2', desc: 'Constructing meaning from instructional messages.' },
    { level: 3, name: 'Apply', verb: 'Calculate, Execute, Solve, Implement', color: '#059669', desc: 'Carrying out or using a procedure in a given situation.' },
    { level: 4, name: 'Analyze', verb: 'Differentiate, Organize, Compare, Distinguish', color: '#d97706', desc: 'Breaking material into constituent parts and detecting relations.' },
    { level: 5, name: 'Evaluate', verb: 'Appraise, Critique, Judge, Justify, Assess', color: '#ea580c', desc: 'Making judgments based on criteria and standards.' },
    { level: 6, name: 'Create', verb: 'Design, Formulate, Construct, Plan, Compose', color: '#7c3aed', desc: 'Putting elements together to form a novel, coherent whole.' },
  ];

  return (
    <div className="container" style={{ padding: '2.5rem 1.5rem', display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      
      {/* Hero Header */}
      <div style={{
        background: 'linear-gradient(135deg, #1e1b4b 0%, #312e81 60%, #1e293b 100%)',
        borderRadius: 'var(--radius-xl)',
        padding: '2.5rem',
        color: '#ffffff',
        display: 'flex',
        flexWrap: 'wrap',
        justifyContent: 'space-between',
        alignItems: 'center',
        gap: '2rem',
        boxShadow: 'var(--shadow-md)'
      }}>
        <div style={{ maxWidth: '640px' }}>
          <div style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '0.5rem',
            padding: '0.3rem 0.75rem',
            borderRadius: '9999px',
            background: 'rgba(255, 255, 255, 0.1)',
            backdropFilter: 'blur(8px)',
            fontSize: '0.75rem',
            fontWeight: 600,
            marginBottom: '1rem',
            border: '1px solid rgba(255, 255, 255, 0.15)'
          }}>
            <BrainCircuit size={14} />
            Deep Learning Academic Project
          </div>
          <h1 style={{ fontSize: '2.25rem', fontWeight: 800, letterSpacing: '-0.025em', lineHeight: 1.2, marginBottom: '0.75rem' }}>
            DeepLearnQ — Question Paper Cognitive Analyzer
          </h1>
          <p style={{ color: '#cbd5e1', fontSize: '0.98rem', lineHeight: 1.6, marginBottom: '1.75rem' }}>
            Automated multi-class classification of examination questions across the 6 cognitive domains 
            of Bloom's Revised Taxonomy using a fine-tuned DistilBERT transformer network.
          </p>

          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.75rem' }}>
            <button 
              onClick={() => onNavigate('analyze', 'upload')}
              className="btn btn-primary"
              style={{ padding: '0.65rem 1.35rem', background: '#4f46e5', fontWeight: 600 }}
            >
              <Upload size={18} />
              Upload Question Paper (PDF)
            </button>
            <button 
              onClick={() => onNavigate('analyze', 'manual')}
              className="btn"
              style={{
                background: 'rgba(255, 255, 255, 0.12)',
                color: '#ffffff',
                border: '1px solid rgba(255, 255, 255, 0.2)',
                fontWeight: 600,
                padding: '0.65rem 1.2rem'
              }}
            >
              <FileEdit size={18} />
              Manual Text Entry
            </button>
          </div>
        </div>

        {/* Hero status box */}
        <div style={{
          background: 'rgba(255, 255, 255, 0.06)',
          backdropFilter: 'blur(12px)',
          border: '1px solid rgba(255, 255, 255, 0.12)',
          borderRadius: 'var(--radius-lg)',
          padding: '1.5rem',
          minWidth: '260px'
        }}>
          <h3 style={{ fontSize: '0.85rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: '#94a3b8', marginBottom: '1rem' }}>
            Inference Engine Status
          </h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
            <div>
              <span style={{ fontSize: '0.75rem', color: '#cbd5e1' }}>Model Checkpoint</span>
              <p style={{ fontWeight: 600, fontSize: '0.95rem' }}>
                {health?.is_trained ? 'Fine-tuned DistilBERT' : 'Base DistilBERT (Demo)'}
              </p>
            </div>
            <div>
              <span style={{ fontSize: '0.75rem', color: '#cbd5e1' }}>Hardware Accelerator</span>
              <p style={{ fontWeight: 600, fontSize: '0.95rem' }}>
                {health?.device?.toUpperCase() || 'CPU (PyTorch)'}
              </p>
            </div>
            <div>
              <span style={{ fontSize: '0.75rem', color: '#cbd5e1' }}>Cognitive Categories</span>
              <p style={{ fontWeight: 600, fontSize: '0.95rem' }}>
                6 Levels (Remember → Create)
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Stats and Info Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '1.5rem' }}>
        
        {/* Model Card */}
        <div className="card">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1rem' }}>
            <div style={{ padding: '0.5rem', borderRadius: 'var(--radius-md)', background: '#e0e7ff', color: '#4338ca' }}>
              <Cpu size={22} />
            </div>
            <div>
              <h2 style={{ fontSize: '1.05rem', fontWeight: 600 }}>Deep Learning Classifier</h2>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>DistilBERT Sequence Classification</p>
            </div>
          </div>
          <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)', marginBottom: '1rem', lineHeight: 1.5 }}>
            {modelInfo?.status_message || 'DistilBERT encoder with pre-classification dense layer and 6-class softmax output.'}
          </p>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.825rem', paddingTop: '0.75rem', borderTop: '1px solid var(--border-light)' }}>
            <span style={{ color: 'var(--text-muted)' }}>Status:</span>
            <span style={{ fontWeight: 600, color: health?.is_trained ? '#059669' : '#d97706' }}>
              {health?.is_trained ? 'Active & Fine-Tuned' : 'Base Demo Mode'}
            </span>
          </div>
        </div>

        {/* Dataset Card */}
        <div className="card">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1rem' }}>
            <div style={{ padding: '0.5rem', borderRadius: 'var(--radius-md)', background: '#ecfdf5', color: '#047857' }}>
              <Database size={22} />
            </div>
            <div>
              <h2 style={{ fontSize: '1.05rem', fontWeight: 600 }}>Bloom's Dataset</h2>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Academic Question Corpus</p>
            </div>
          </div>
          <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)', marginBottom: '1rem', lineHeight: 1.5 }}>
            Curated examination questions mapped across all 6 cognitive orders with zero data leakage.
          </p>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.825rem', paddingTop: '0.75rem', borderTop: '1px solid var(--border-light)' }}>
            <span style={{ color: 'var(--text-muted)' }}>Total Cleaned Records:</span>
            <span style={{ fontWeight: 600 }}>{datasetInfo?.total_samples ? `${datasetInfo.total_samples.toLocaleString()} questions` : '8,758 questions'}</span>
          </div>
        </div>

        {/* Last Analysis Card */}
        <div className="card">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1rem' }}>
            <div style={{ padding: '0.5rem', borderRadius: 'var(--radius-md)', background: '#fef3c7', color: '#b45309' }}>
              <BarChart2 size={22} />
            </div>
            <div>
              <h2 style={{ fontSize: '1.05rem', fontWeight: 600 }}>Session Activity</h2>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Latest Paper Analysis</p>
            </div>
          </div>
          {lastAnalysis ? (
            <div>
              <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>
                Classified <strong>{lastAnalysis.summary.total_questions}</strong> questions. Dominant cognitive level: 
                <span className={`badge badge-${lastAnalysis.summary.dominant_level.toLowerCase()}`} style={{ marginLeft: '0.5rem' }}>
                  {lastAnalysis.summary.dominant_level}
                </span>
              </p>
              <button 
                onClick={() => onNavigate('results')}
                className="btn btn-secondary" 
                style={{ width: '100%', marginTop: '0.75rem', fontSize: '0.8rem' }}
              >
                View Full Results <ArrowRight size={14} />
              </button>
            </div>
          ) : (
            <div>
              <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)', marginBottom: '1rem' }}>
                No question papers analyzed in this session yet. Upload a PDF or paste questions to generate cognitive analytics.
              </p>
              <button 
                onClick={() => onNavigate('analyze')}
                className="btn btn-secondary" 
                style={{ width: '100%', fontSize: '0.8rem' }}
              >
                Start First Analysis <ArrowRight size={14} />
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Bloom's Taxonomy Guide Section */}
      <div className="card">
        <div style={{ marginBottom: '1.5rem' }}>
          <h2 style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--text-main)' }}>
            Bloom's Revised Taxonomy Cognitive Framework
          </h2>
          <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
            The 6 hierarchical cognitive categories evaluated by DeepLearnQ, ranked from foundational recall to creative synthesis.
          </p>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '1rem' }}>
          {bloomLevels.map((b) => (
            <div 
              key={b.name} 
              style={{
                border: '1px solid var(--border-light)',
                borderRadius: 'var(--radius-md)',
                padding: '1rem',
                borderLeft: `4px solid ${b.color}`,
                background: '#ffffff'
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.4rem' }}>
                <span style={{ fontWeight: 700, fontSize: '0.95rem', color: b.color }}>
                  Level {b.level}: {b.name}
                </span>
                <span style={{ fontSize: '0.7rem', color: 'var(--text-subtle)', fontWeight: 600 }}>
                  Order {b.level}/6
                </span>
              </div>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '0.5rem', lineHeight: 1.4 }}>
                {b.desc}
              </p>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-main)', background: 'var(--bg-subtle)', padding: '0.35rem 0.5rem', borderRadius: '4px' }}>
                <strong style={{ color: 'var(--text-muted)' }}>Action Verbs:</strong> {b.verb}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
