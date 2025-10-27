import React from 'react';
import { Link } from "react-router-dom";


const Results = ({ prediction, loading }) => {
  if (!prediction || !prediction.success) {
    return null;
  }

  const { severity, risk_level, risk_color, confidence } = prediction.prediction;
  const { probabilities, input_summary } = prediction;

  const getSeverityIcon = (severity) => {
    switch (severity) {
      case 'Fatal injury': return '☠️';
      case 'Serious Injury': return '🚑';
      case 'Slight Injury': return '⚕️';
      default: return '⚠️';
    }
  };

  return (
     <div className="results-container">
      {/* Main Prediction Card */}
      <div className="card prediction-card" style={{ borderColor: risk_color }}>
        <div className="card__body">
          <div className="prediction-header">
            <span className="severity-icon">{getSeverityIcon(severity)}</span>
            <div>
              <h2>Predicted Severity</h2>
              <p className="text-secondary">Based on provided accident details</p>
            </div>
          </div>

          <div className="prediction-result" style={{ borderLeftColor: risk_color }}>
            <div className="severity-label">
              <h1 style={{ color: risk_color }}>{severity}</h1>
              <span className="status" style={{ 
                backgroundColor: `${risk_color}26`,
                color: risk_color,
                border: `1px solid ${risk_color}40`
              }}>
                {risk_level} Risk
              </span>
            </div>
            <div className="confidence-score">
              <span className="confidence-label">Confidence</span>
              <span className="confidence-value" style={{ color: risk_color }}>
                {confidence.toFixed(1)}%
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Probabilities Card */}
      <div className="card">
        <div className="card__body">
          <h3>Severity Probabilities</h3>
          <div className="probabilities-list">
            {Object.entries(probabilities)
              .sort(([, a], [, b]) => b - a)
              .map(([severityType, probability]) => (
                <div key={severityType} className="probability-item">
                  <div className="probability-header">
                    <span>{getSeverityIcon(severityType)} {severityType}</span>
                    <span className="probability-value">{probability.toFixed(1)}%</span>
                  </div>
                  <div className="probability-bar-container">
                    <div 
                      className="probability-bar"
                      style={{ 
                        width: `${probability}%`,
                        backgroundColor: severityType === severity ? risk_color : '#ddd'
                      }}
                    />
                  </div>
                </div>
              ))}
          </div>
        </div>
      </div>

      {/* Input Summary Card */}
      <div className="card">
        <div className="card__body">
          <h3>Scenario Summary</h3>
          <div className="summary-grid">
            <div className="summary-item">
              <span className="summary-label">Driver Age: </span>
              <span className="summary-value">{input_summary.driver_age}</span>
            </div>
            <div className="summary-item">
              <span className="summary-label">Experience: </span>
              <span className="summary-value">{input_summary.experience}</span>
            </div>
            <div className="summary-item">
              <span className="summary-label">Weather: </span>
              <span className="summary-value">{input_summary.conditions.weather}</span>
            </div>
            <div className="summary-item">
              <span className="summary-label">Light: </span>
              <span className="summary-value">{input_summary.conditions.light}</span>
            </div>
            <div className="summary-item">
              <span className="summary-label">Road Condition: </span>
              <span className="summary-value">{input_summary.conditions.road}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Safety Recommendations */}
      <div className="card recommendations-card">
        <div className="card__body">
          <h3>🛡️ Safety Recommendations</h3>
          <ul className="recommendations-list">
            {severity === 'Fatal injury' && (
              <>
                <li>⛔ <strong>CRITICAL:</strong> This scenario has high fatal risk. Avoid if possible.</li>
                <li>🚨 Emergency services should be on standby</li>
                <li>📍 Consider alternative routes or timing</li>
              </>
            )}
            {severity === 'Serious Injury' && (
              <>
                <li>⚠️ <strong>HIGH RISK:</strong> Exercise extreme caution</li>
                <li>🦺 Ensure all safety equipment is functional</li>
                <li>🚗 Reduce speed and increase following distance</li>
              </>
            )}
            {severity === 'Slight Injury' && (
              <>
                <li>✅ Moderate risk - maintain standard safety practices</li>
                <li>👀 Stay alert and avoid distractions</li>
                <li>🛣️ Follow traffic rules and speed limits</li>
              </>
            )}
          </ul>
        </div>
      </div>
  
      {/* ✅ View Detailed Analysis Button */}
      {prediction && !loading && (
        <div style={{ textAlign: "center", marginTop: "20px" }}>
          <Link to="/analysis" state={{ result: prediction }}>
            <button  className="btn btn--primary btn--lg">
              🔍 View Detailed Analysis
            </button>
          </Link>
        </div>
      )}

    </div>
  );
};

export default Results;
