import React, { useState, useEffect } from 'react';
import axios from 'axios';
import PredictionForm from './components/PredictionForm';
import Results from './components/Results';
import ModelInfo from './components/ModelInfo';
import './App.css';
import Analysis from "./Analysis";
import { Routes, Route } from "react-router-dom";   // ✅ Only import Routes/Route

const API_URL = 'http://localhost:5000';

function App() {
  const [features, setFeatures] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showModelInfo, setShowModelInfo] = useState(false);

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    try {
      const [featuresRes, modelRes] = await Promise.all([
        axios.get(`${API_URL}/features`),
        axios.get(`${API_URL}/model-info`)
      ]);
      setFeatures(featuresRes.data);
      setModelInfo(modelRes.data);
    } catch (err) {
      console.error('Error:', err);
      setError('Failed to connect to backend. Make sure Flask server is running on port 5000.');
    }
  };

  const handlePredict = async (formData) => {
    setLoading(true);
    setError(null);
    setPrediction(null);

    try {
      const response = await axios.post(`${API_URL}/predict`, formData);
      setPrediction(response.data);
    } catch (err) {
      setError(err.response?.data?.error || 'Prediction failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setPrediction(null);
    setError(null);
  };

  return (
    <Routes>

      {/* ✅ MAIN HOME PAGE */}
      <Route path="/" element={
        <div className="app">
          <header className="app-header">
            <div className="container">
              <h1>🚗 Accident Severity Predictor</h1>
              <p>AI-Powered Risk Assessment System</p>
              <button 
                className="info-btn"
                onClick={() => setShowModelInfo(!showModelInfo)}
              >
                ℹ️ Model Info
              </button>
            </div>
          </header>

          <main className="main-content">
            <div className="container">
              {showModelInfo && modelInfo && (
                <ModelInfo modelInfo={modelInfo} onClose={() => setShowModelInfo(false)} />
              )}

              {error && (
                <div className="error-message">
                  <span>⚠️</span>
                  <p>{error}</p>
                </div>
              )}

              <div className="content-grid">
                <div className="form-section">
                  <PredictionForm 
                    features={features}
                    onSubmit={handlePredict}
                    onReset={handleReset}
                    loading={loading}
                  />
                </div>

                <div className="results-section">
                  <Results 
                    prediction={prediction}
                    loading={loading}
                  />
                </div>
              </div>
            </div>
          </main>

          <footer className="app-footer">
            <p>© 2025 Road Accident Severity Prediction System</p>
            <p>Powered by Random Forest ML | Accuracy: {modelInfo?.metrics?.accuracy || 'N/A'}</p>
          </footer>
        </div>
      } />

      {/* ✅ ANALYSIS PAGE */}
      <Route path="/analysis" element={<Analysis />} />

    </Routes>
  );
}

export default App;
