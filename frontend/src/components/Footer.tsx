import React from 'react';

export const Footer: React.FC = () => {
  return (
    <footer style={{
      marginTop: 'auto',
      borderTop: '1px solid var(--border-light)',
      background: '#ffffff',
      padding: '2rem 0',
      color: 'var(--text-muted)',
      fontSize: '0.85rem'
    }}>
      <div className="container" style={{
        display: 'flex',
        flexWrap: 'wrap',
        justifyContent: 'space-between',
        alignItems: 'center',
        gap: '1rem'
      }}>
        <div>
          <p style={{ fontWeight: 600, color: 'var(--text-main)' }}>
            DeepLearnQ — Question Paper Cognitive Analyzer
          </p>
          <p style={{ fontSize: '0.78rem', color: 'var(--text-subtle)', marginTop: '0.2rem' }}>
            College Deep Learning Mini-Project • Fine-Tuned DistilBERT Transformer Classifier
          </p>
        </div>

        <div style={{ display: 'flex', gap: '1.5rem', fontSize: '0.8rem' }}>
          <span>PyTorch 2.11</span>
          <span>•</span>
          <span>Hugging Face Transformers</span>
          <span>•</span>
          <span>PyMuPDF</span>
          <span>•</span>
          <span>Flask REST API</span>
          <span>•</span>
          <span>React + Vite</span>
        </div>
      </div>
    </footer>
  );
};
