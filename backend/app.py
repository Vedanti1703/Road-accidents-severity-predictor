"""
Road Accident Severity Prediction - Flask API
Provides REST endpoints for ML predictions + What-If Simulation + Top-10 Feature Interaction Heatmap
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
from treeinterpreter import treeinterpreter as ti
import pickle
import pandas as pd
import traceback
import seaborn as sns
import matplotlib.pyplot as plt
import base64
from io import BytesIO


app = Flask(__name__)
CORS(app)

MODEL_PATH = 'random_forest_model.pkl'
ENCODERS_PATH = 'label_encoders.pkl'
FEATURES_PATH = 'feature_info.pkl'

model = None
label_encoders = None
feature_info = None

# Load dataset for heatmap
df = pd.read_csv("./cleaned_data.csv")

def get_accidents_by_time_period(df):
    """
    Count accidents by Time_Period (Morning/Afternoon/Evening/Night).
    """
    if "Time_Period" not in df.columns:
        return None

    counts = df["Time_Period"].value_counts().reindex(["Morning","Afternoon","Evening","Night"]).fillna(0)

    return {
        "labels": counts.index.tolist(),
        "values": counts.values.tolist()
    }



# ---------------- HEATMAP (Top 10 Feature Correlation) ----------------
from difflib import get_close_matches

def generate_feature_correlation_heatmap(df, top_features, limit=10):
    """
    Generates correlation heatmap for top features, correcting name mismatches automatically.
    """
    df_cols = df.columns.tolist()
    selected = []

    # Fuzzy match each top feature to closest column in dataset
    for f in top_features:
        match = get_close_matches(f, df_cols, n=1, cutoff=0.6)
        if match:
            selected.append(match[0])

    selected = selected[:limit]

    if len(selected) < 2:
        return None

    temp = df[selected].copy()

    # Convert categoricals to numeric
    for col in temp.columns:
        if temp[col].dtype == "object":
            temp[col] = temp[col].astype("category").cat.codes

    corr = temp.corr()

    return {
        "x_labels": corr.columns.tolist(),
        "y_labels": corr.index.tolist(),
        "values": corr.values.round(2).tolist()
    }

# ---------------- SAFETY RECOMMENDATION ENGINE ----------------
def generate_recommendations(input_data):
    recommendations = []

    age = input_data.get("Age_band_of_driver", "")
    exp = input_data.get("Driving_experience", "")
    vehicle = input_data.get("Type_of_vehicle", "")

    # AGE
    if age in ["Under 18", "18-30"]:
        recommendations.append({
            "factor": "Driver Age",
            "value": age,
            "risk": "High Risk",
            "recommendation": "Younger drivers are statistically more prone to speeding and risk-taking behavior."
        })
    else:
        recommendations.append({
            "factor": "Driver Age",
            "value": age,
            "risk": "Low Risk",
            "recommendation": "This age range typically demonstrates more stable driving patterns."
        })

    # EXPERIENCE
    if exp in ["Below 1yr", "1-2yr", "2-5yr"]:
        recommendations.append({
            "factor": "Driving Experience",
            "value": exp,
            "risk": "High Risk",
            "recommendation": "Limited driving experience increases the chance of misjudging road situations. Extra caution advised."
        })
    else:
        recommendations.append({
            "factor": "Driving Experience",
            "value": exp,
            "risk": "Low Risk",
            "recommendation": "Experience contributes to safer decision-making during driving."
        })

    # VEHICLE TYPE
    if vehicle in ["Motorcycle", "Bicycle", "Bajaj"]:
        recommendations.append({
            "factor": "Vehicle Type",
            "value": vehicle,
            "risk": "High Risk",
            "recommendation": "Two/three-wheelers offer less physical protection. Wear protective gear and avoid speeding."
        })
    else:
        recommendations.append({
            "factor": "Vehicle Type",
            "value": vehicle,
            "risk": "Moderate/Low Risk",
            "recommendation": "This vehicle type provides better stability and protection compared to two-wheelers."
        })

    return recommendations



# ---------------- MODEL LOADING ----------------
def load_model_files():
    global model, label_encoders, feature_info
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(ENCODERS_PATH, 'rb') as f:
            label_encoders = pickle.load(f)
        with open(FEATURES_PATH, 'rb') as f:
            feature_info = pickle.load(f)
        return True
    except Exception as e:
        print("❌ Error loading model files:", e)
        return False


@app.route('/')
def home():
    return jsonify({'status': 'online', 'message': 'Severity Prediction API Active'})

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
        'model_type': 'Random Forest Classifier',
        'algorithm': 'Ensemble Learning - Bagging with Decision Trees',
        'balancing_technique': 'SMOTE (Synthetic Minority Over-sampling)',
        'n_estimators': 150,
        'max_depth': 12,
        'metrics': {
            'accuracy': f"{metrics.get('accuracy', 0)*100:.2f}%",
            'f1_score': f"{metrics.get('f1_score', 0)*100:.2f}%",
            'precision': f"{metrics.get('precision', 0)*100:.2f}%",
            'recall': f"{metrics.get('recall', 0)*100:.2f}%"
        },
        'classes': list(label_encoders['target'].classes_),
        'total_features': len(feature_info['features']),
        'training_samples': 7873
    })

# ---------------- PREDICT ----------------
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
          # ----- Feature Contribution (TreeInterpreter) -----
        pred_val, bias, contributions = ti.predict(model, encoded_data.values)
        contrib = contributions[0][prediction_encoded]

        feature_contribs = sorted(
            list(zip(required_features, contrib)),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        top_features = [
            {"feature": f, "impact": float(round(v, 4))}
            for f, v in feature_contribs[:10]
        ]

        # ----- Explanation Sentence -----
        reason = "Severity mainly influenced by: " + ", ".join([t["feature"] for t in top_features[:5]]) + "."

        # ----- Heatmap -----
        heatmap = generate_feature_correlation_heatmap(df, [t["feature"] for t in top_features], limit=10)

        # ----- Time Period Histogram -----
        time_distribution = get_accidents_by_time_period(df)

        # ----- Recommendations -----
        recommendations = generate_recommendations(data)
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
                'conditions': {
                    'weather': data.get('Weather_conditions'),
                    'light': data.get('Light_conditions'),
                    'road': data.get('Road_surface_conditions')
                }
            },
            'explanation': {
                'top_features': top_features,
                'reason': reason
            },
            'heatmap': heatmap,
            'time_distribution': time_distribution,
            'recommendations': recommendations
        }
        response['recommendations'] = generate_recommendations(data)

        return jsonify(response)
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({
            'error': 'Prediction failed',
            'detail': str(e)
        }), 500




# ---------------- SIMULATE (WHAT-IF) ----------------
@app.route('/simulate', methods=['POST'])
def simulate():
    try:
        partial = request.get_json()
        required = feature_info['features']

        full = {}
        for f in required:
            if f in partial:
                full[f] = partial[f]
            else:
                full[f] = feature_info['categorical_features'][f][0] if f in feature_info['categorical_features'] else 0

        df_input = pd.DataFrame([full], columns=required)
        encoded = df_input.copy()
        for col in required:
            if col in label_encoders:
                encoded[col] = label_encoders[col].transform(encoded[col].astype(str))

        pred = model.predict(encoded)[0]
        proba = model.predict_proba(encoded)[0]
        pred_label = label_encoders['target'].inverse_transform([pred])[0]

        pred_val, bias, contributions = ti.predict(model, encoded.values)
        contrib = contributions[0][pred]

        feature_contribs = sorted(list(zip(required, contrib)),
                                  key=lambda x: abs(x[1]),
                                  reverse=True)

        top = [{"feature": f, "impact": float(round(v, 4))} for f, v in feature_contribs[:10]]

    
        recommendations = generate_recommendations(full)

        reason = f"Predicted severity: {pred_label}. Influenced mainly by: " + ", ".join([t["feature"] for t in top[:5]]) + "."
        # Generate heatmap for the top selected features
        heatmap = generate_feature_correlation_heatmap(df, [t["feature"] for t in top], limit=10)
              
        return jsonify({
            'success': True,
            'prediction': pred_label,
            'probabilities': list(map(float, proba)),
            'heatmap': heatmap,
            'explanation': {
                'top_features': top,
                'reason': reason
            },
            'recommendations': recommendations,
            'time_distribution': get_accidents_by_time_period(df),
            'heatmap': generate_feature_correlation_heatmap(df, [t["feature"] for t in top]),

        })

    except Exception:
        traceback.print_exc()
        return jsonify({'error': 'Simulation failed'}), 500

@app.route('/time-distribution', methods=['GET'])
def time_distribution():
    hist = get_accidents_by_time_period(df)
    if hist is None:
        return jsonify({'error': 'Time_Period column not found'}), 400
    return jsonify(hist)

# ---------------- RUN SERVER ----------------
if __name__ == '__main__':
    print("\nLoading model...")
    if load_model_files():
        print("✅ Model loaded successfully\n")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to load model.\n")
