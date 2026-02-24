import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './Dashboard.css';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const Dashboard = ({ onLogout, onNavigate }) => {
  const [profile, setProfile] = useState(null);
  const [diagnosisHistory, setDiagnosisHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [selectedDiagnosis, setSelectedDiagnosis] = useState(null);
  const [statsLoading, setStatsLoading] = useState(true);

  useEffect(() => {
    fetchProfileAndHistory();
  }, []);

  const getAuthHeaders = () => {
    const token = localStorage.getItem('token');
    return {
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json'
      },
      timeout: 10000  // 10 second timeout
    };
  };

  const fetchProfileAndHistory = async () => {
    try {
      setLoading(true);
      const token = localStorage.getItem('token');

      if (!token) {
        if (onNavigate) {
          onNavigate('login');
        }
        return;
      }

      const headers = getAuthHeaders();

      // Fetch profile first (should be fast)
      const profileResponse = await axios.get(`${API_BASE_URL}/api/patient/profile`, headers);
      setProfile(profileResponse.data);
      setLoading(false);  // Show dashboard with profile loaded

      // Fetch diagnosis history in background (can be slow)
      try {
        const historyResponse = await axios.get(`${API_BASE_URL}/api/patient/diagnosis-history`, headers);
        setDiagnosisHistory(historyResponse.data.recent_diagnoses || []);
      } catch (historyErr) {
        console.error('Failed to load diagnosis history:', historyErr);
        setDiagnosisHistory([]);
      }
      setStatsLoading(false);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to load data');
      setLoading(false);
      if (err.response?.status === 401) {
        localStorage.clear();
        if (onNavigate) {
          onNavigate('login');
        }
      }
    }
  };

  const handleLogout = async () => {
    const token = localStorage.getItem('token');
    if (token) {
      try {
        await axios.post(`${API_BASE_URL}/api/auth/logout`, {}, getAuthHeaders());
      } catch (err) {
        // Logout request failed, but proceed with client-side logout anyway
        console.error('Logout request failed:', err);
      }
    }
    localStorage.clear();
    if (onLogout) {
      onLogout();
    }
  };

  const getClassificationBadge = (classification) => {
    if (classification === 'ADHD') {
      return <span className="badge badge-adhd">⚠️ ADHD Detected</span>;
    } else if (classification === 'Control') {
      return <span className="badge badge-control">✓ No ADHD</span>;
    }
    return <span className="badge badge-pending">⏳ Pending</span>;
  };

  const getRiskLevelColor = (riskLevel) => {
    switch (riskLevel) {
      case 'High':
        return 'risk-high';
      case 'Moderate':
        return 'risk-moderate';
      case 'Low':
        return 'risk-low';
      default:
        return 'risk-unknown';
    }
  };

  const getConfidencePercentage = (confidence) => {
    if (!confidence) return 0;
    return Math.round(confidence * 100);
  };

  const handleUploadClick = () => {
    if (onNavigate) {
      onNavigate('upload');
    }
  };

  if (loading) {
    return (
      <div className="dashboard-container">
        <div className="loading-spinner">
          <div className="spinner"></div>
          <p>Loading your profile...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="dashboard-container">
      {/* Header */}
      <header className="dashboard-header">
        <div className="header-left">
          <h1>EEG ADHD Detection System</h1>
          <p className="tagline">Advanced AI-Powered Diagnosis</p>
        </div>
        <div className="header-right">
          {profile && (
            <div className="user-info">
              <div className="user-avatar">
                {profile.first_name.charAt(0)}{profile.last_name.charAt(0)}
              </div>
              <div className="user-details">
                <p className="user-name">{profile.first_name} {profile.last_name}</p>
                <p className="user-email">{profile.email}</p>
              </div>
              <button onClick={handleLogout} className="logout-btn">Logout</button>
            </div>
          )}
        </div>
      </header>

      {error && (
        <div className="alert alert-error">
          {error}
        </div>
      )}

      {/* Main Content */}
      <div className="dashboard-content">
        {/* Quick Stats */}
        <section className="stats-section">
          <h2>Quick Statistics</h2>
          <div className="stats-grid">
            <div className="stat-card">
              <div className="stat-icon">📊</div>
              <div className="stat-info">
                <p className="stat-label">Total Diagnoses</p>
                <p className="stat-value">{diagnosisHistory.length}</p>
              </div>
            </div>

            {diagnosisHistory.length > 0 && (
              <>
                <div className="stat-card">
                  <div className="stat-icon">⚠️</div>
                  <div className="stat-info">
                    <p className="stat-label">ADHD Cases</p>
                    <p className="stat-value">
                      {diagnosisHistory.filter(d => d.classification === 'ADHD').length}
                    </p>
                  </div>
                </div>

                <div className="stat-card">
                  <div className="stat-icon">✓</div>
                  <div className="stat-info">
                    <p className="stat-label">Control Cases</p>
                    <p className="stat-value">
                      {diagnosisHistory.filter(d => d.classification === 'Control').length}
                    </p>
                  </div>
                </div>

                <div className="stat-card">
                  <div className="stat-icon">📅</div>
                  <div className="stat-info">
                    <p className="stat-label">Latest Diagnosis</p>
                    <p className="stat-value">
                      {new Date(diagnosisHistory[0].upload_date).toLocaleDateString()}
                    </p>
                  </div>
                </div>
              </>
            )}
          </div>
        </section>

        {/* Action Buttons */}
        <section className="actions-section">
          <button onClick={handleUploadClick} className="primary-btn">
            + New EEG Analysis
          </button>
        </section>

        {/* Diagnosis History */}
        <section className="history-section">
          <div className="section-header">
            <h2>Diagnosis History</h2>
            <p className="section-subtitle">Your previous EEG analyses</p>
          </div>

          {diagnosisHistory.length === 0 ? (
            <div className="empty-state">
              <div className="empty-icon">🔬</div>
              <h3>No Diagnoses Yet</h3>
              <p>Start by uploading an EEG file to get your first diagnosis</p>
              <button onClick={handleUploadClick} className="primary-btn">
                Upload EEG File
              </button>
            </div>
          ) : (
            <div className="history-table-container">
              <table className="history-table">
                <thead>
                  <tr>
                    <th>Date</th>
                    <th>File Name</th>
                    <th>Status</th>
                    <th>Classification</th>
                    <th>Probability</th>
                    <th>Confidence</th>
                    <th>Risk Level</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {diagnosisHistory.map((diagnosis) => (
                    <tr key={diagnosis.upload_id} className="history-row">
                      <td className="date-cell">
                        {new Date(diagnosis.upload_date).toLocaleDateString()}
                        <span className="time">
                          {new Date(diagnosis.upload_date).toLocaleTimeString([], {
                            hour: '2-digit',
                            minute: '2-digit'
                          })}
                        </span>
                      </td>
                      <td className="filename-cell">
                        <span title={diagnosis.filename}>{diagnosis.filename}</span>
                      </td>
                      <td>
                        <span className={`status-badge status-${diagnosis.status}`}>
                          {diagnosis.status.charAt(0).toUpperCase() + diagnosis.status.slice(1)}
                        </span>
                      </td>
                      <td>
                        {diagnosis.classification ? getClassificationBadge(diagnosis.classification) : '-'}
                      </td>
                      <td className="probability-cell">
                        {diagnosis.adhd_probability ? `${(diagnosis.adhd_probability * 100).toFixed(1)}%` : '-'}
                      </td>
                      <td className="confidence-cell">
                        {diagnosis.confidence ? `${getConfidencePercentage(diagnosis.confidence)}%` : '-'}
                      </td>
                      <td>
                        {diagnosis.risk_level && (
                          <span className={`risk-badge ${getRiskLevelColor(diagnosis.risk_level)}`}>
                            {diagnosis.risk_level}
                          </span>
                        )}
                      </td>
                      <td className="actions-cell">
                        {diagnosis.pdf_path && (
                          <a href={`${API_BASE_URL}/download/${diagnosis.pdf_path}`} className="action-link" download>
                            📥 Report
                          </a>
                        )}
                        {diagnosis.classification && (
                          <button
                            onClick={() => setSelectedDiagnosis(diagnosis)}
                            className="action-link view-btn"
                          >
                            👁️ View
                          </button>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </div>

      {/* Detail Modal */}
      {selectedDiagnosis && (
        <div className="modal-overlay" onClick={() => setSelectedDiagnosis(null)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>Diagnosis Details</h3>
              <button className="close-btn" onClick={() => setSelectedDiagnosis(null)}>×</button>
            </div>
            <div className="modal-body">
              <div className="detail-row">
                <span className="detail-label">File Name:</span>
                <span className="detail-value">{selectedDiagnosis.filename}</span>
              </div>
              <div className="detail-row">
                <span className="detail-label">Date:</span>
                <span className="detail-value">
                  {new Date(selectedDiagnosis.upload_date).toLocaleString()}
                </span>
              </div>
              <div className="detail-row">
                <span className="detail-label">Status:</span>
                <span className={`status-badge status-${selectedDiagnosis.status}`}>
                  {selectedDiagnosis.status}
                </span>
              </div>
              <div className="detail-row">
                <span className="detail-label">Classification:</span>
                <span className="detail-value">
                  {getClassificationBadge(selectedDiagnosis.classification)}
                </span>
              </div>
              <div className="detail-row">
                <span className="detail-label">ADHD Probability:</span>
                <span className="detail-value">
                  {selectedDiagnosis.adhd_probability ? `${(selectedDiagnosis.adhd_probability * 100).toFixed(2)}%` : '-'}
                </span>
              </div>
              <div className="detail-row">
                <span className="detail-label">Confidence Score:</span>
                <span className="detail-value">
                  {selectedDiagnosis.confidence ? `${getConfidencePercentage(selectedDiagnosis.confidence)}%` : '-'}
                </span>
              </div>
              <div className="detail-row">
                <span className="detail-label">Risk Level:</span>
                <span className={`risk-badge ${getRiskLevelColor(selectedDiagnosis.risk_level)}`}>
                  {selectedDiagnosis.risk_level}
                </span>
              </div>
            </div>
            <div className="modal-footer">
              {selectedDiagnosis.pdf_path && (
                <a href={`${API_BASE_URL}/download/${selectedDiagnosis.pdf_path}`} className="primary-btn" download>
                  Download Report
                </a>
              )}
              <button onClick={() => setSelectedDiagnosis(null)} className="secondary-btn">
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Dashboard;
