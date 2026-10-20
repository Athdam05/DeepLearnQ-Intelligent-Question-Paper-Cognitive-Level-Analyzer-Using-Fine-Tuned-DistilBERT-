import React from 'react';
import { 
  BookOpen, 
  BarChart3, 
  FileText, 
  CheckCircle2, 
  HelpCircle,
  LayoutDashboard
} from 'lucide-react';
import type { HealthInfo } from '../types';

interface NavbarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  health: HealthInfo | null;
  hasResults: boolean;
}

export const Navbar: React.FC<NavbarProps> = ({ 
  activeTab, 
  setActiveTab, 
  hasResults 
}) => {
  return (
    <header style={{
      background: '#ffffff',
      borderBottom: '1px solid var(--border-light)',
      position: 'sticky',
      top: 0,
      zIndex: 50,
      boxShadow: 'var(--shadow-sm)'
    }}>
      <div className="container" style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        height: '4.25rem',
      }}>
        {/* Logo and Brand */}
        <div 
          onClick={() => setActiveTab('dashboard')} 
          style={{ display: 'flex', alignItems: 'center', gap: '0.85rem', cursor: 'pointer' }}
        >
          <div style={{
            width: '2.5rem',
            height: '2.5rem',
            borderRadius: 'var(--radius-md)',
            background: 'linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: '#ffffff',
            boxShadow: '0 2px 4px rgba(79, 70, 229, 0.25)'
          }}>
            <BookOpen size={20} strokeWidth={2.2} />
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <span style={{ fontSize: '1.25rem', fontWeight: 700, letterSpacing: '-0.02em', color: '#0f172a' }}>
                DeepLearn<span style={{ color: 'var(--primary)' }}>Q</span>
              </span>
              <span style={{
                fontSize: '0.65rem',
                padding: '0.15rem 0.45rem',
                borderRadius: '4px',
                background: 'var(--bg-subtle)',
                color: 'var(--text-muted)',
                fontWeight: 600,
                border: '1px solid var(--border-light)'
              }}>
                DistilBERT
              </span>
            </div>
            <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', margin: 0 }}>
              Intelligent Question Paper Cognitive Analyzer
            </p>
          </div>
        </div>

        {/* Navigation Links */}
        <nav style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
          <button 
            onClick={() => setActiveTab('dashboard')}
            className="btn"
            style={{
              background: activeTab === 'dashboard' ? 'var(--primary-light)' : 'transparent',
              color: activeTab === 'dashboard' ? 'var(--primary)' : 'var(--text-muted)',
              fontWeight: activeTab === 'dashboard' ? 600 : 500,
            }}
          >
            <LayoutDashboard size={17} />
            Dashboard
          </button>

          <button 
            onClick={() => setActiveTab('analyze')}
            className="btn"
            style={{
              background: activeTab === 'analyze' ? 'var(--primary-light)' : 'transparent',
              color: activeTab === 'analyze' ? 'var(--primary)' : 'var(--text-muted)',
              fontWeight: activeTab === 'analyze' ? 600 : 500,
            }}
          >
            <FileText size={17} />
            Analyze Paper
          </button>

          <button 
            onClick={() => setActiveTab('results')}
            className="btn"
            disabled={!hasResults}
            style={{
              background: activeTab === 'results' ? 'var(--primary-light)' : 'transparent',
              color: activeTab === 'results' ? 'var(--primary)' : (!hasResults ? 'var(--text-subtle)' : 'var(--text-muted)'),
              fontWeight: activeTab === 'results' ? 600 : 500,
              opacity: !hasResults ? 0.6 : 1,
              cursor: !hasResults ? 'not-allowed' : 'pointer'
            }}
          >
            <BarChart3 size={17} />
            Results
          </button>

          <button 
            onClick={() => setActiveTab('evaluation')}
            className="btn"
            style={{
              background: activeTab === 'evaluation' ? 'var(--primary-light)' : 'transparent',
              color: activeTab === 'evaluation' ? 'var(--primary)' : 'var(--text-muted)',
              fontWeight: activeTab === 'evaluation' ? 600 : 500,
            }}
          >
            <CheckCircle2 size={17} />
            Model Evaluation
          </button>

          <button 
            onClick={() => setActiveTab('about')}
            className="btn"
            style={{
              background: activeTab === 'about' ? 'var(--primary-light)' : 'transparent',
              color: activeTab === 'about' ? 'var(--primary)' : 'var(--text-muted)',
              fontWeight: activeTab === 'about' ? 600 : 500,
            }}
          >
            <HelpCircle size={17} />
            About
          </button>
        </nav>
      </div>
    </header>
  );
};
