import React, { useState } from 'react';

const PredictionForm = ({ features, onSubmit, onReset, loading }) => {
  const [formData, setFormData] = useState({});

  if (!features) {
    return <div className="card"><p>Loading form...</p></div>;
  }

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit(formData);
  };

  const handleResetForm = () => {
    setFormData({});
    onReset();
  };

  const renderInput = (featureName) => {
    const options = features.categorical_options[featureName];
    
    // If it's a categorical feature with options
    if (options) {
      return (
        <select
          name={featureName}
          value={formData[featureName] || ''}
          onChange={handleChange}
          required
          className="form-control"
        >
          <option value="">-- Select --</option>
          {options.map(option => (
            <option key={option} value={option}>
              {option}
            </option>
          ))}
        </select>
      );
    }
    
    // Numeric input
    return (
      <input
        type="number"
        name={featureName}
        value={formData[featureName] || ''}
        onChange={handleChange}
        required
        min="0"
        className="form-control"
        placeholder="Enter number"
      />
    );
  };

  // Group features by category
  const featureGroups = {
    'Driver Information': [
      'Age_band_of_driver',
      'Sex_of_driver',
      'Educational_level',
      'Driving_experience',
      'Vehicle_driver_relation'
    ],
    'Vehicle Details': [
      'Type_of_vehicle',
      'Owner_of_vehicle',
      'Number_of_vehicles_involved'
    ],
    'Time & Date': [
      'Day_of_week',
      'Time_Period'
    ],
    'Road Conditions': [
      'Area_accident_occured',
      'Lanes_or_Medians',
      'Road_allignment',
      'Types_of_Junction',
      'Road_surface_type',
      'Road_surface_conditions'
    ],
    'Environmental Conditions': [
      'Light_conditions',
      'Weather_conditions'
    ],
    'Accident Details': [
      'Type_of_collision',
      'Vehicle_movement',
      'Cause_of_accident',
      'Number_of_casualties',
      'Casualty_severity'
    ]
  };

  const formatLabel = (name) => {
    return name.replace(/_/g, ' ')
      .split(' ')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  return (
    <div className="card">
      <div className="card__body">
        <h2>Enter Accident Details</h2>
        <p className="text-secondary">Fill in all fields to predict severity</p>

        <form onSubmit={handleSubmit}>
          {Object.entries(featureGroups).map(([groupName, groupFeatures]) => (
            <div key={groupName} className="form-group-section">
              <h3 className="group-title">{groupName}</h3>
              <div className="form-grid">
                {groupFeatures.map(feature => (
                  features.features.includes(feature) && (
                    <div key={feature} className="form-group">
                      <label className="form-label">
                        {formatLabel(feature)}
                      </label>
                      {renderInput(feature)}
                    </div>
                  )
                ))}
              </div>
            </div>
          ))}

          <div className="form-actions">
            <button 
              type="submit" 
              className="btn btn--primary btn--lg"
              disabled={loading}
            >
              {loading ? 'Predicting...' : '🔍 Predict Severity'}
            </button>
            <button 
              type="button" 
              className="btn btn--secondary btn--lg"
              onClick={handleResetForm}
            >
              🔄 Reset Form
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default PredictionForm;
