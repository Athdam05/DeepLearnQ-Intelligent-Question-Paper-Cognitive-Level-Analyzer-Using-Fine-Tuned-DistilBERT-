import React, { useState } from 'react';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer, 
  Cell,
  PieChart, 
  Pie, 
  Legend
} from 'recharts';
import { 
  Download, 
  Search, 
  Filter, 
  AlertCircle, 
  CheckCircle, 
  ChevronDown, 
  ChevronUp, 
  Layers, 
  Sparkles,
  Eye,
  X
} from 'lucide-react';
import { AnalysisResponse, QuestionPrediction, BloomLevel } from '../types';
import { apiService } from '../services/api';

interface ResultsPageProps {
  analysis: AnalysisResponse;
  onAnalyzeAnother: () => void;
}

export const ResultsPage: React.FC<ResultsPageProps> = ({ analysis, onAnalyzeAnother }) => {
  const { summary, questions } = analysis;

  const [searchQuery, setSearchQuery] = useState<string>('');
  const [selectedCategory, setSelectedCategory] = useState<string>('ALL');
  const [showLowConfidenceOnly, setShowLowConfidenceOnly] = useState<boolean>(false);
  const [sortBy, setSortBy] = useState<'id' | 'confidence-desc' | 'confidence-asc'>('id');
  const [inspectedQuestion, setInspectedQuestion] = useState<QuestionPrediction | null>(null);
  const [isDownloading, setIsDownloading] = useState<boolean>(false);

  // Filter questions
  const filteredQuestions = questions.filter(q => {
    const matchesSearch = q.question.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesCategory = selectedCategory === 'ALL' || q.predicted_level === selectedCategory;
    const matchesLowConf = !showLowConfidenceOnly || q.is_low_confidence;
    return matchesSearch && matchesCategory && matchesLowConf;
  }).sort((a, b) => {
    if (sortBy === 'id') return a.id - b.id;
    if (sortBy === 'confidence-desc') return b.confidence - a.confidence;
    if (sortBy === 'confidence-asc') return a.confidence - b.confidence;
    return 0;
  });

  const handleDownloadCsv = async () => {
    setIsDownloading(true);
    try {
      await apiService.downloadResultsCsv(questions);
    } catch (err: any) {
      alert('Failed to download CSV: ' + err.message);
    } finally {
      setIsDownloading(false);
    }
  };

  const pieData = summary.chart_data.filter(d => d.count > 0).map(d => ({
    name: d.category,
    value: d.count,
    color: d.color
  }));

  return (
    <div className="container" style={{ padding: '2.5rem 1.5rem', display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      
      {/* Page Title & Top Actions */}
      <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'space-between', alignItems: 'center', gap: '1rem' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem' }}>
            <h1 style={{ fontSize: '1.85rem', fontWeight: 800, color: 'var(--text-main)', letterSpacing: '-0.02em' }}>
              Cognitive Level Analysis Results
            </h1>
            <span className="badge badge-remember" style={{ fontSize: '0.72rem' }}>
              DistilBERT Inference
            </span>
          </div>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', marginTop: '0.25rem' }}>
            Cognitive taxonomy distribution and confidence calibration for <strong>{summary.total_questions}</strong> examination questions.
          </p>
        </div>

        <div style={{ display: 'flex', gap: '0.75rem' }}>
          <button onClick={onAnalyzeAnother} className="btn btn-secondary">
            Analyze Another Paper
          </button>
          <button 
            onClick={handleDownloadCsv} 
            disabled={isDownloading} 
            className="btn btn-primary"
            style={{ fontWeight: 600 }}
          >
            <Download size={16} />
            {isDownloading ? 'Exporting CSV...' : 'Download CSV Report'}
          </button>
        </div>
      </div>

      {/* KPI Summary Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1.25rem' }}>
        <div className="card">
          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: 500 }}>Total Questions</span>
          <p style={{ fontSize: '2rem', fontWeight: 800, color: 'var(--text-main)', marginTop: '0.25rem' }}>
            {summary.total_questions}
          </p>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-subtle)' }}>Extracted & Classified</span>
        </div>

        <div className="card">
          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: 500 }}>Dominant Cognitive Level</span>
          <p style={{ fontSize: '1.6rem', fontWeight: 800, color: 'var(--primary)', marginTop: '0.25rem' }}>
            {summary.dominant_level}
          </p>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            {summary.dominant_percentage}% of total paper weight
          </span>
        </div>

        <div className="card">
          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: 500 }}>Average Model Confidence</span>
          <p style={{ fontSize: '2rem', fontWeight: 800, color: '#059669', marginTop: '0.25rem' }}>
            {(summary.average_confidence * 100).toFixed(1)}%
          </p>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-subtle)' }}>Softmax confidence score</span>
        </div>

        <div className="card">
          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: 500 }}>Higher-Order Thinking (HOT)</span>
          <p style={{ fontSize: '2rem', fontWeight: 800, color: '#7c3aed', marginTop: '0.25rem' }}>
            {summary.higher_order_percentage}%
          </p>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-subtle)' }}>Analyze, Evaluate & Create</span>
        </div>

        <div className="card">
          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: 500 }}>Low-Confidence Flagged</span>
          <p style={{ fontSize: '2rem', fontWeight: 800, color: summary.low_confidence_count > 0 ? '#ea580c' : '#059669', marginTop: '0.25rem' }}>
            {summary.low_confidence_count}
          </p>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-subtle)' }}>
            Score &lt; {(summary.confidence_threshold * 100).toFixed(0)}% (flagged for review)
          </span>
        </div>
      </div>

      {/* Visual Analytics Section */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(450px, 1fr))', gap: '1.5rem' }}>
        
        {/* Bar Chart */}
        <div className="card">
          <div style={{ marginBottom: '1.25rem' }}>
            <h2 style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--text-main)' }}>
              Cognitive Level Distribution (Bloom's Taxonomy)
            </h2>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              Question frequency across cognitive hierarchy levels
            </p>
          </div>

          <div style={{ width: '100%', height: 280 }}>
            <ResponsiveContainer>
              <BarChart data={summary.chart_data} margin={{ top: 10, right: 10, left: -15, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis dataKey="category" tick={{ fontSize: 12 }} interval={0} />
                <YAxis allowDecimals={false} tick={{ fontSize: 12 }} />
                <Tooltip 
                  formatter={(val: any, name: any, item: any) => [
                    `${val} questions (${item.payload.percentage}%)`, 
                    'Count'
                  ]}
                  contentStyle={{ backgroundColor: '#ffffff', borderRadius: '8px', border: '1px solid #e2e8f0', fontSize: '0.85rem' }}
                />
                <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                  {summary.chart_data.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Pie / Donut Chart */}
        <div className="card">
          <div style={{ marginBottom: '1.25rem' }}>
            <h2 style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--text-main)' }}>
              Cognitive Domain Proportion
            </h2>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              Percentage share of question paper cognitive depth
            </p>
          </div>

          <div style={{ width: '100%', height: 280, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            {pieData.length > 0 ? (
              <ResponsiveContainer>
                <PieChart>
                  <Pie
                    data={pieData}
                    cx="50%"
                    cy="50%"
                    innerRadius={65}
                    outerRadius={105}
                    paddingAngle={3}
                    dataKey="value"
                    label={({ name, percent }: any) => `${name} (${(percent * 100).toFixed(0)}%)`}
                    labelLine={false}
                  >
                    {pieData.map((entry, index) => (
                      <Cell key={`pie-cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip 
                    formatter={(val: any) => [`${val} questions`, 'Weight']}
                    contentStyle={{ backgroundColor: '#ffffff', borderRadius: '8px', border: '1px solid #e2e8f0', fontSize: '0.85rem' }}
                  />
                </PieChart>
              </ResponsiveContainer>
            ) : (
              <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>No data to display</p>
            )}
          </div>
        </div>
      </div>

      {/* Category Breakdown Table */}
      <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
        <div style={{ padding: '1.25rem 1.5rem', borderBottom: '1px solid var(--border-light)' }}>
          <h2 style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--text-main)' }}>
            Category-Wise Cognitive Breakdown
          </h2>
          <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
            Aggregated metric summary of each Bloom taxonomy tier
          </p>
        </div>

        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.875rem' }}>
            <thead style={{ background: '#f8fafc', borderBottom: '1px solid var(--border-light)', color: 'var(--text-muted)', fontSize: '0.78rem', textTransform: 'uppercase' }}>
              <tr>
                <th style={{ padding: '0.75rem 1.5rem' }}>Order</th>
                <th style={{ padding: '0.75rem 1rem' }}>Category</th>
                <th style={{ padding: '0.75rem 1rem' }}>Question Count</th>
                <th style={{ padding: '0.75rem 1rem' }}>Percentage</th>
                <th style={{ padding: '0.75rem 1.5rem' }}>Cognitive Skill Focus</th>
              </tr>
            </thead>
            <tbody>
              {summary.chart_data.map((c) => (
                <tr key={c.category} style={{ borderBottom: '1px solid var(--border-light)' }}>
                  <td style={{ padding: '0.75rem 1.5rem', fontWeight: 600, color: 'var(--text-muted)' }}>
                    Level {c.level}
                  </td>
                  <td style={{ padding: '0.75rem 1rem' }}>
                    <span className={`badge badge-${c.category.toLowerCase()}`}>
                      {c.category}
                    </span>
                  </td>
                  <td style={{ padding: '0.75rem 1rem', fontWeight: 700 }}>
                    {c.count}
                  </td>
                  <td style={{ padding: '0.75rem 1rem' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <div style={{ width: '80px', height: '6px', background: '#e2e8f0', borderRadius: '4px', overflow: 'hidden' }}>
                        <div style={{ width: `${c.percentage}%`, height: '100%', background: c.color }} />
                      </div>
                      <span style={{ fontSize: '0.8rem', fontWeight: 600 }}>{c.percentage}%</span>
                    </div>
                  </td>
                  <td style={{ padding: '0.75rem 1.5rem', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                    {c.category === 'Remember' && 'Recalling knowledge, definitions, and facts'}
                    {c.category === 'Understand' && 'Explaining ideas, interpreting principles'}
                    {c.category === 'Apply' && 'Executing formulas, problem-solving in practice'}
                    {c.category === 'Analyze' && 'Dissecting concepts, identifying relationships'}
                    {c.category === 'Evaluate' && 'Critiquing, justifying decisions, appraising'}
                    {c.category === 'Create' && 'Synthesizing novel designs, formulating structures'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Question-wise Classification Table */}
      <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
        
        <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'space-between', alignItems: 'center', gap: '1rem' }}>
          <div>
            <h2 style={{ fontSize: '1.2rem', fontWeight: 700, color: 'var(--text-main)' }}>
              Question-Wise Classification Table ({filteredQuestions.length} of {questions.length})
            </h2>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              Individual predictions generated by fine-tuned DistilBERT with model confidence ratings
            </p>
          </div>

          {/* Filters & Search */}
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.75rem', alignItems: 'center' }}>
            {/* Search */}
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              padding: '0.4rem 0.75rem',
              border: '1px solid var(--border-medium)',
              borderRadius: 'var(--radius-md)',
              background: '#ffffff'
            }}>
              <Search size={16} color="var(--text-subtle)" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search question text..."
                style={{ border: 'none', outline: 'none', width: '160px' }}
              />
            </div>

            {/* Category Filter */}
            <select
              value={selectedCategory}
              onChange={(e) => setSelectedCategory(e.target.value)}
              style={{
                padding: '0.4rem 0.75rem',
                border: '1px solid var(--border-medium)',
                borderRadius: 'var(--radius-md)',
                background: '#ffffff',
                color: 'var(--text-main)',
                outline: 'none'
              }}
            >
              <option value="ALL">All Categories</option>
              <option value="Remember">Remember</option>
              <option value="Understand">Understand</option>
              <option value="Apply">Apply</option>
              <option value="Analyze">Analyze</option>
              <option value="Evaluate">Evaluate</option>
              <option value="Create">Create</option>
            </select>

            {/* Sort */}
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as any)}
              style={{
                padding: '0.4rem 0.75rem',
                border: '1px solid var(--border-medium)',
                borderRadius: 'var(--radius-md)',
                background: '#ffffff',
                color: 'var(--text-main)',
                outline: 'none'
              }}
            >
              <option value="id">Sort by Question #</option>
              <option value="confidence-desc">Confidence: High to Low</option>
              <option value="confidence-asc">Confidence: Low to High</option>
            </select>

            {/* Low Confidence Filter Checkbox */}
            <label style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.45rem',
              fontSize: '0.825rem',
              cursor: 'pointer',
              color: 'var(--text-muted)'
            }}>
              <input
                type="checkbox"
                checked={showLowConfidenceOnly}
                onChange={(e) => setShowLowConfidenceOnly(e.target.checked)}
              />
              <span>Review flagged only</span>
            </label>
          </div>
        </div>

        {/* Table Content */}
        <div style={{ overflowX: 'auto', border: '1px solid var(--border-light)', borderRadius: 'var(--radius-md)' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.875rem' }}>
            <thead style={{ background: '#f8fafc', borderBottom: '1px solid var(--border-light)', color: 'var(--text-muted)', fontSize: '0.78rem', textTransform: 'uppercase' }}>
              <tr>
                <th style={{ padding: '0.75rem 1rem', width: '50px' }}>#</th>
                <th style={{ padding: '0.75rem 1rem' }}>Question Text</th>
                <th style={{ padding: '0.75rem 1rem', width: '140px' }}>Predicted Category</th>
                <th style={{ padding: '0.75rem 1rem', width: '160px' }}>Model Confidence</th>
                <th style={{ padding: '0.75rem 1rem', width: '120px' }}>Review Flag</th>
                <th style={{ padding: '0.75rem 1rem', width: '110px' }}>Probabilities</th>
              </tr>
            </thead>
            <tbody>
              {filteredQuestions.length > 0 ? (
                filteredQuestions.map((q) => (
                  <tr key={q.id} style={{ borderBottom: '1px solid var(--border-light)' }}>
                    <td style={{ padding: '0.85rem 1rem', fontWeight: 600, color: 'var(--text-muted)' }}>
                      Q{q.id}
                    </td>
                    <td style={{ padding: '0.85rem 1rem', lineHeight: 1.45, color: '#1e293b' }}>
                      {q.question}
                    </td>
                    <td style={{ padding: '0.85rem 1rem' }}>
                      <span className={`badge badge-${q.predicted_level.toLowerCase()}`}>
                        {q.predicted_level}
                      </span>
                    </td>
                    <td style={{ padding: '0.85rem 1rem' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <div style={{ width: '70px', height: '6px', background: '#e2e8f0', borderRadius: '4px', overflow: 'hidden' }}>
                          <div 
                            style={{ 
                              width: `${q.confidence_percentage}%`, 
                              height: '100%', 
                              background: q.is_low_confidence ? '#ea580c' : '#059669' 
                            }} 
                          />
                        </div>
                        <span style={{ fontSize: '0.8rem', fontWeight: 600, color: q.is_low_confidence ? '#ea580c' : '#059669' }}>
                          {q.confidence_percentage}%
                        </span>
                      </div>
                    </td>
                    <td style={{ padding: '0.85rem 1rem' }}>
                      {q.is_low_confidence ? (
                        <span style={{
                          display: 'inline-flex',
                          alignItems: 'center',
                          gap: '0.25rem',
                          padding: '0.2rem 0.5rem',
                          borderRadius: '4px',
                          background: '#fff7ed',
                          color: '#c2410c',
                          fontSize: '0.725rem',
                          fontWeight: 600,
                          border: '1px solid #ffedd5'
                        }}>
                          <AlertCircle size={12} />
                          Review
                        </span>
                      ) : (
                        <span style={{
                          display: 'inline-flex',
                          alignItems: 'center',
                          gap: '0.25rem',
                          padding: '0.2rem 0.5rem',
                          borderRadius: '4px',
                          background: '#f0fdf4',
                          color: '#15803d',
                          fontSize: '0.725rem',
                          fontWeight: 600
                        }}>
                          <CheckCircle size={12} />
                          High Conf
                        </span>
                      )}
                    </td>
                    <td style={{ padding: '0.85rem 1rem' }}>
                      <button
                        onClick={() => setInspectedQuestion(q)}
                        className="btn btn-secondary"
                        style={{ padding: '0.3rem 0.6rem', fontSize: '0.75rem' }}
                      >
                        <Eye size={13} />
                        View
                      </button>
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={6} style={{ padding: '2.5rem', textAlign: 'center', color: 'var(--text-muted)' }}>
                    No questions match the selected filters or search query.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Probabilities Inspection Modal */}
      {inspectedQuestion && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'rgba(15, 23, 42, 0.6)',
          backdropFilter: 'blur(4px)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 100,
          padding: '1rem'
        }}>
          <div style={{
            background: '#ffffff',
            borderRadius: 'var(--radius-lg)',
            maxWidth: '560px',
            width: '100%',
            padding: '1.75rem',
            boxShadow: 'var(--shadow-lg)',
            display: 'flex',
            flexDirection: 'column',
            gap: '1.25rem'
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', fontWeight: 600 }}>QUESTION #{inspectedQuestion.id} PROBABILITY DISTRIBUTION</span>
                <h3 style={{ fontSize: '1.15rem', fontWeight: 700, marginTop: '0.15rem' }}>
                  Softmax Output Breakdown
                </h3>
              </div>
              <button onClick={() => setInspectedQuestion(null)} style={{ color: 'var(--text-muted)' }}>
                <X size={20} />
              </button>
            </div>

            <div style={{ background: 'var(--bg-subtle)', padding: '0.85rem', borderRadius: 'var(--radius-md)', fontSize: '0.875rem', color: '#1e293b', fontStyle: 'italic', border: '1px solid var(--border-light)' }}>
              "{inspectedQuestion.question}"
            </div>

            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem', fontSize: '0.825rem' }}>
                <span style={{ fontWeight: 600 }}>Predicted Bloom Order:</span>
                <span className={`badge badge-${inspectedQuestion.predicted_level.toLowerCase()}`}>
                  {inspectedQuestion.predicted_level} ({(inspectedQuestion.confidence * 100).toFixed(1)}%)
                </span>
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.65rem', marginTop: '1rem' }}>
                {Object.entries(inspectedQuestion.class_probabilities).map(([category, prob]) => {
                  const percent = (prob * 100).toFixed(1);
                  const isTop = category === inspectedQuestion.predicted_level;
                  return (
                    <div key={category}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', marginBottom: '0.2rem' }}>
                        <span style={{ fontWeight: isTop ? 700 : 500, color: isTop ? 'var(--text-main)' : 'var(--text-muted)' }}>
                          {category}
                        </span>
                        <span style={{ fontWeight: isTop ? 700 : 500, color: isTop ? 'var(--primary)' : 'var(--text-muted)' }}>
                          {percent}%
                        </span>
                      </div>
                      <div style={{ width: '100%', height: '8px', background: '#f1f5f9', borderRadius: '4px', overflow: 'hidden' }}>
                        <div 
                          style={{ 
                            width: `${percent}%`, 
                            height: '100%', 
                            background: isTop ? 'var(--primary)' : '#94a3b8' 
                          }} 
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            <button 
              onClick={() => setInspectedQuestion(null)}
              className="btn btn-secondary" 
              style={{ width: '100%', marginTop: '0.5rem' }}
            >
              Close
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
