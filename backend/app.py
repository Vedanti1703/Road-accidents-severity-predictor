
from flask import Flask, send_from_directory
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__, static_folder="static_charts")
CORS(app)  # Enable CORS for React frontend

# Configuration
MODEL_PATH = 'random_forest_model.pkl'
ENCODERS_PATH = 'label_encoders.pkl'
FEATURES_PATH = 'feature_info.pkl'

# Global variables
model = None
label_encoders = None
feature_info = None

def load_model_files():
    """Load ML model and preprocessing artifacts"""
    global model, label_encoders, feature_info
    
    try:
        print("Loading model files...")
        
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        print("✓ Model loaded")
        
        with open(ENCODERS_PATH, 'rb') as f:
            label_encoders = pickle.load(f)
        print("✓ Encoders loaded")
        
        with open(FEATURES_PATH, 'rb') as f:
            feature_info = pickle.load(f)
        print("✓ Features loaded")
        
        return True
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("⚠️  Make sure you have run train_model_random_forest.py first!")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
# Pre-load your dataset at the top of app.py
df = pd.read_csv('cleaned_data.csv')

@app.route('/static_charts/<path:filename>')
def static_charts(filename):
    return send_from_directory(app.static_folder, filename)

@app.route('/')
def home():
    """Health check endpoint"""
    return jsonify({
        'status': 'online',
        'message': 'Road Accident Severity Prediction API',
        'model': 'Random Forest Classifier',
        'endpoints': {
            '/': 'Health check',
            '/features': 'GET - Get available features',
            '/model-info': 'GET - Get model information',
            '/predict': 'POST - Make prediction'
        }
    })

@app.route('/features', methods=['GET'])
def get_features():
    """Return all features and their possible values"""
    if feature_info is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'features': feature_info['features'],
        'categorical_options': feature_info['categorical_features']
    })

@app.route('/model-info', methods=['GET'])
def get_model_info():
    """Return model performance metrics"""
    if feature_info is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    metrics = feature_info.get('model_metrics', {})
    
    return jsonify({
        'model_type': ': Random Forest Classifier',
        'algorithm': ': Ensemble Learning - Bagging with Decision Trees',
        'balancing_technique': ': SMOTE (Synthetic Minority Over-sampling)',
        'n_estimators: ': 150,
        'max_depth: ': 12,
        'metrics': {
            'accuracy - ': f"{metrics.get('accuracy', 0)*100:.2f}%",
            'f1_score - ': f"{metrics.get('f1_score', 0)*100:.2f}%",
            'precision - ': f"{metrics.get('precision', 0)*100:.2f}%",
            'recall - ': f"{metrics.get('recall', 0)*100:.2f}%"
        },
        'classes': list(label_encoders['target'].classes_),
        'total_features: ': len(feature_info['features']),
        'training_samples': 7873
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Make accident severity prediction
    
    Expected JSON format:
    {
        "Day_of_week": "Monday",
        "Age_band_of_driver": "18-30",
        "Sex_of_driver": "Male",
        ... (all 23 features)
    }
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate all required features are present
        required_features = feature_info['features']
        missing_features = [f for f in required_features if f not in data]
        
        if missing_features:
            return jsonify({
                'error': 'Missing required features',
                'missing': missing_features
            }), 400
        
        # Create DataFrame with correct column order
        input_df = pd.DataFrame([data], columns=required_features)
        
        # Encode categorical variables
        encoded_data = input_df.copy()
        for column in required_features:
            if column in label_encoders:
                try:
                    encoded_data[column] = label_encoders[column].transform(
                        encoded_data[column].astype(str)
                    )
                except ValueError as e:
                    return jsonify({
                        'error': f'Invalid value for {column}',
                        'detail': str(e),
                        'valid_options': list(label_encoders[column].classes_)
                    }), 400
        
        # Make prediction
        prediction_encoded = model.predict(encoded_data)[0]
        prediction_proba = model.predict_proba(encoded_data)[0]
        
        # Decode prediction
        prediction = label_encoders['target'].inverse_transform([prediction_encoded])[0]
        
        # Get class probabilities
        classes = label_encoders['target'].classes_
        probabilities = {
            classes[i]: float(prediction_proba[i]) * 100 
            for i in range(len(classes))
        }
        
        # Determine risk level and styling
        if prediction == 'Fatal injury':
            risk_level = 'Critical'
            risk_color = '#ff4444'
            risk_icon = '☠️'
        elif prediction == 'Serious Injury':
            risk_level = 'High'
            risk_color = '#ff9800'
            risk_icon = '🚑'
        else:
            risk_level = 'Moderate'
            risk_color = '#4caf50'
            risk_icon = '⚕️'
        
        # Prepare response
        response = {
            'success': True,
            'prediction': {
                'severity': prediction,
                'risk_level': risk_level,
                'risk_color': risk_color,
                'risk_icon': risk_icon,
                'confidence': float(max(prediction_proba)) * 100
            },
            'probabilities': probabilities,
            'input_summary': {
                'driver_age': data.get('Age_band_of_driver'),
                'experience': data.get('Driving_experience'),
                'vehicle_type': data.get('Type_of_vehicle'),
                'num_vehicles': data.get('Number_of_vehicles_involved'),
                'casualties': data.get('Number_of_casualties'),
                'cause': data.get('Cause_of_accident'),
                'conditions': {
                   
                    'light': data.get('Light_conditions'),
                }
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({
            'error': 'Prediction failed',
            'detail': str(e)
        }), 500

if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("  ROAD ACCIDENT SEVERITY PREDICTION API")
    print("  Random Forest Classifier")
    print("=" * 80 + "\n")
    
    # Load model artifacts
    if load_model_files():
        print("\n✓ All files loaded successfully!")
        print(f"✓ Model ready with {len(feature_info['features'])} features")
        print(f"✓ Target classes: {list(label_encoders['target'].classes_)}")
        print("\n" + "=" * 80)
        print("  Starting Flask server on http://localhost:5000")
        print("=" * 80 + "\n")
        
        # Run Flask app
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\n❌ Failed to load model files!")
        print("⚠️  Please run train_model_random_forest.py first to generate:")
        print("   - random_forest_model.pkl")
        print("   - label_encoders.pkl")
        print("   - feature_info.pkl")