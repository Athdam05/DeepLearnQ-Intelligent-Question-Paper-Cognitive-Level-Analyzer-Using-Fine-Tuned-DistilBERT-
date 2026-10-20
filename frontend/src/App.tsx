import React, { useState, useEffect } from 'react';
import { Navbar } from './components/Navbar';
import { Footer } from './components/Footer';
import { Dashboard } from './pages/Dashboard';
import { AnalyzePage } from './pages/AnalyzePage';
import { ResultsPage } from './pages/ResultsPage';
import { EvaluationPage } from './pages/EvaluationPage';
import { AboutPage } from './pages/AboutPage';
import { apiService } from './services/api';
import { HealthInfo, ModelInfo, DatasetInfo, AnalysisResponse } from './types';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<string>('dashboard');
  const [analyzeMode, setAnalyzeMode] = useState<'upload' | 'manual'>('upload');
  const [health, setHealth] = useState<HealthInfo | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [datasetInfo, setDatasetInfo] = useState<DatasetInfo | null>(null);
  const [lastAnalysis, setLastAnalysis] = useState<AnalysisResponse | null>(null);

  useEffect(() => {
    const checkSystem = async () => {
      try {
        const [h, m, d] = await Promise.all([
          apiService.getHealth().catch(() => null),
          apiService.getModelInfo().catch(() => null),
          apiService.getDatasetInfo().catch(() => null),
        ]);
        if (h) setHealth(h);
        if (m) setModelInfo(m);
        if (d) setDatasetInfo(d);
      } catch (e) {
        console.warn('System initialization notice:', e);
      }
    };
    checkSystem();
    // Refresh health every 25 seconds
    const interval = setInterval(checkSystem, 25000);
    return () => clearInterval(interval);
  }, []);

  const handleNavigate = (tab: string, mode?: 'upload' | 'manual') => {
    if (mode) setAnalyzeMode(mode);
    setActiveTab(tab);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const handleAnalysisComplete = (result: AnalysisResponse) => {
    setLastAnalysis(result);
    setActiveTab('results');
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <Navbar 
        activeTab={activeTab} 
        setActiveTab={setActiveTab} 
        health={health} 
        hasResults={lastAnalysis !== null} 
      />

      <main style={{ flex: 1 }}>
        {activeTab === 'dashboard' && (
          <Dashboard 
            health={health}
            modelInfo={modelInfo}
            datasetInfo={datasetInfo}
            lastAnalysis={lastAnalysis}
            onNavigate={handleNavigate}
          />
        )}

        {activeTab === 'analyze' && (
          <AnalyzePage 
            initialMode={analyzeMode}
            onAnalysisComplete={handleAnalysisComplete}
          />
        )}

        {activeTab === 'results' && lastAnalysis && (
          <ResultsPage 
            analysis={lastAnalysis}
            onAnalyzeAnother={() => handleNavigate('analyze')}
          />
        )}

        {activeTab === 'evaluation' && (
          <EvaluationPage />
        )}

        {activeTab === 'about' && (
          <AboutPage />
        )}
      </main>

      <Footer />
    </div>
  );
};

export default App;
