import React from 'react';

const ModelInfo = ({ modelInfo, onClose }) => {
  return (
    <div className="card model-info-card">
      <div className="card__body">
        <div className="flex justify-between items-center mb-16">
          <h2>🤖 Model Information</h2>
          <button  className="btn btn--secondary btn--sm close" onClick={onClose}>
            ✕ Close
          </button>
        </div>

        <div className="info-grid">
          <div className="info-item">
            <span className="info-label">Model Type</span>
            <span className="info-value">{modelInfo.model_type}</span>
          </div>
          <div className="info-item">
            <span className="info-label">Algorithm</span>
            <span className="info-value">{modelInfo.algorithm}</span>
          </div>
          <div className="info-item">
            <span className="info-label">Balancing Technique</span>
            <span className="info-value">{modelInfo.balancing_technique}</span>
          </div>
          <div className="info-item">
            <span className="info-label">Total Features</span>
            <span className="info-value">{modelInfo.total_features}</span>
          </div>
        </div>

        <h3 className="mt-24 mb-12">📊 Performance Metrics</h3>
        <div className="metrics-grid">
          {Object.entries(modelInfo.metrics).map(([metric, value]) => (
            <div key={metric} className="metric-card">
              <span className="metric-name">
                {metric.replace('_', ' ').toUpperCase()}
              </span>
              <span className="metric-value">{value}</span>
            </div>
          ))}
        </div>

        <h3 className="mt-24 mb-12">🎯 Predicted Classes</h3>
        <div className="classes-list">
          {modelInfo.classes.map(className => (
            <span key={className} className="status status--info">
              {className}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
};

export default ModelInfo;
