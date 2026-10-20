import React from 'react';
import { 
  Cpu, 
  Award,
  Layers,
  Sparkles
} from 'lucide-react';

export const AboutPage: React.FC = () => {
  return (
    <div className="container" style={{ padding: '2.5rem 1.5rem', display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      
      {/* Title */}
      <div>
        <h1 style={{ fontSize: '1.85rem', fontWeight: 800, color: 'var(--text-main)', letterSpacing: '-0.02em' }}>
          About DeepLearnQ
        </h1>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', marginTop: '0.25rem' }}>
          Academic documentation, system objectives, and architectural design of the question paper cognitive analyzer.
        </p>
      </div>

      {/* Problem Statement Card */}
      <div className="card">
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1rem' }}>
          <div style={{ padding: '0.5rem', borderRadius: 'var(--radius-md)', background: '#e0e7ff', color: '#4338ca' }}>
            <Award size={22} />
          </div>
          <div>
            <h2 style={{ fontSize: '1.15rem', fontWeight: 700 }}>Problem Statement & Motivation</h2>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Outcome-Based Education (OBE) & Cognitive Balance</p>
          </div>
        </div>

        <p style={{ fontSize: '0.9rem', color: 'var(--text-muted)', lineHeight: 1.6, marginBottom: '1rem' }}>
          In modern higher education engineering curricula (under NBA, ABET, and NAAC frameworks), course assessments 
          must test a balanced range of cognitive skills rather than merely rote fact memorization. Manually tagging 
          every question in every semester examination is time-consuming, subjective, and prone to significant 
          inter-evaluator inconsistency.
        </p>
        <p style={{ fontSize: '0.9rem', color: 'var(--text-muted)', lineHeight: 1.6 }}>
          <strong>DeepLearnQ</strong> automates this workflow: it ingests examination papers in PDF format, 
          segments individual questions using heuristic boundary detection, and feeds each question through 
          a fine-tuned DistilBERT transformer network to produce an objective cognitive distribution report 
          with confidence estimation.
        </p>
      </div>

      {/* Architecture Deep Dive */}
      <div className="card">
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1rem' }}>
          <div style={{ padding: '0.5rem', borderRadius: 'var(--radius-md)', background: '#ecfdf5', color: '#047857' }}>
            <Cpu size={22} />
          </div>
          <div>
            <h2 style={{ fontSize: '1.15rem', fontWeight: 700 }}>Deep Learning Architecture Pipeline</h2>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>End-to-end processing pipeline</p>
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1rem', marginTop: '1rem' }}>
          <div style={{ border: '1px solid var(--border-light)', borderRadius: 'var(--radius-md)', padding: '1.25rem', background: '#fafafa' }}>
            <span style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--primary)', textTransform: 'uppercase' }}>Step 1: Ingestion</span>
            <h3 style={{ fontSize: '0.95rem', fontWeight: 600, margin: '0.35rem 0' }}>PDF Extraction & Segmentation</h3>
            <p style={{ fontSize: '0.825rem', color: 'var(--text-muted)', lineHeight: 1.5 }}>
              PyMuPDF streams document text vectors. Regular expression boundary detectors identify question numbers (e.g. Q1., 1., marks patterns) and segment questions into discrete strings while flagging scanned image PDFs.
            </p>
          </div>

          <div style={{ border: '1px solid var(--border-light)', borderRadius: 'var(--radius-md)', padding: '1.25rem', background: '#fafafa' }}>
            <span style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--primary)', textTransform: 'uppercase' }}>Step 2: Representation</span>
            <h3 style={{ fontSize: '0.95rem', fontWeight: 600, margin: '0.35rem 0' }}>WordPiece Tokenization</h3>
            <p style={{ fontSize: '0.825rem', color: 'var(--text-muted)', lineHeight: 1.5 }}>
              DistilBERT Tokenizer maps question tokens to 30,522 vocabulary IDs with [CLS] and [SEP] special tokens. Padding and truncation are applied up to a maximum sequence length of 128 tokens.
            </p>
          </div>

          <div style={{ border: '1px solid var(--border-light)', borderRadius: 'var(--radius-md)', padding: '1.25rem', background: '#fafafa' }}>
            <span style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--primary)', textTransform: 'uppercase' }}>Step 3: Transformer</span>
            <h3 style={{ fontSize: '0.95rem', fontWeight: 600, margin: '0.35rem 0' }}>DistilBERT Encoder</h3>
            <p style={{ fontSize: '0.825rem', color: 'var(--text-muted)', lineHeight: 1.5 }}>
              6 transformer layers with 12 self-attention heads and 768 hidden dimension. Produces a contextual 768-dimensional pooler embedding for the [CLS] sequence representation.
            </p>
          </div>

          <div style={{ border: '1px solid var(--border-light)', borderRadius: 'var(--radius-md)', padding: '1.25rem', background: '#fafafa' }}>
            <span style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--primary)', textTransform: 'uppercase' }}>Step 4: Classification</span>
            <h3 style={{ fontSize: '0.95rem', fontWeight: 600, margin: '0.35rem 0' }}>Softmax Head & Confidence</h3>
            <p style={{ fontSize: '0.825rem', color: 'var(--text-muted)', lineHeight: 1.5 }}>
              Pre-classifier dense layer (ReLU) followed by dropout (0.2) and a 6-node linear projection layer. Softmax calculates probability distribution across Bloom levels, flagging ambiguous predictions below the review threshold.
            </p>
          </div>
        </div>
      </div>

      {/* Bloom's Revised Taxonomy Categories */}
      <div className="card">
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1rem' }}>
          <div style={{ padding: '0.5rem', borderRadius: 'var(--radius-md)', background: '#fef3c7', color: '#b45309' }}>
            <Layers size={22} />
          </div>
          <div>
            <h2 style={{ fontSize: '1.15rem', fontWeight: 700 }}>Bloom's Revised Taxonomy (Anderson & Krathwohl, 2001)</h2>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>The 6 cognitive levels classified by DeepLearnQ</p>
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1rem', marginTop: '1rem' }}>
          <div style={{ borderLeft: '4px solid #2563eb', padding: '0.75rem 1rem', background: '#f8fafc', borderRadius: '0 var(--radius-md) var(--radius-md) 0' }}>
            <span style={{ fontWeight: 700, color: '#2563eb', fontSize: '0.9rem' }}>Level 1: Remember</span>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>Recalling relevant knowledge, definitions, facts, and terminology from memory.</p>
          </div>
          <div style={{ borderLeft: '4px solid #0891b2', padding: '0.75rem 1rem', background: '#f8fafc', borderRadius: '0 var(--radius-md) var(--radius-md) 0' }}>
            <span style={{ fontWeight: 700, color: '#0891b2', fontSize: '0.9rem' }}>Level 2: Understand</span>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>Constructing meaning from instructional messages; explaining, interpreting, summarizing.</p>
          </div>
          <div style={{ borderLeft: '4px solid #059669', padding: '0.75rem 1rem', background: '#f8fafc', borderRadius: '0 var(--radius-md) var(--radius-md) 0' }}>
            <span style={{ fontWeight: 700, color: '#059669', fontSize: '0.9rem' }}>Level 3: Apply</span>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>Executing or implementing procedures and formulas in concrete problem-solving situations.</p>
          </div>
          <div style={{ borderLeft: '4px solid #d97706', padding: '0.75rem 1rem', background: '#f8fafc', borderRadius: '0 var(--radius-md) var(--radius-md) 0' }}>
            <span style={{ fontWeight: 700, color: '#d97706', fontSize: '0.9rem' }}>Level 4: Analyze</span>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>Breaking material into constituent parts and determining how parts relate to one another.</p>
          </div>
          <div style={{ borderLeft: '4px solid #ea580c', padding: '0.75rem 1rem', background: '#f8fafc', borderRadius: '0 var(--radius-md) var(--radius-md) 0' }}>
            <span style={{ fontWeight: 700, color: '#ea580c', fontSize: '0.9rem' }}>Level 5: Evaluate</span>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>Making judgments, appraisals, and critiques based on established criteria and standards.</p>
          </div>
          <div style={{ borderLeft: '4px solid #7c3aed', padding: '0.75rem 1rem', background: '#f8fafc', borderRadius: '0 var(--radius-md) var(--radius-md) 0' }}>
            <span style={{ fontWeight: 700, color: '#7c3aed', fontSize: '0.9rem' }}>Level 6: Create</span>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>Putting elements together to form a novel, coherent whole or designing original systems.</p>
          </div>
        </div>
      </div>
    </div>
  );
};
