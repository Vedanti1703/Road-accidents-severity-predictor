"""
Road Accident Severity Prediction - Random Forest Model Training Script
Author: Your Name
Date: October 2025

This script trains a Random Forest model with SMOTE balancing for 
multi-class accident severity prediction.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, 
                            accuracy_score, f1_score, recall_score, precision_score)
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(filepath='cleaned_data.csv'):
    """Load and preprocess the dataset"""
    print("=" * 80)
    print("STEP 1: LOADING AND PREPROCESSING DATA")
    print("=" * 80)
    
    df = pd.read_csv(filepath)
    print(f"✓ Dataset loaded: {df.shape}")
    print(f"✓ Total records: {len(df)}")
    # Select only the most important features
    keep_features = [
    "Number_of_casualties", "Cause_of_accident", "Day_of_week", "Type_of_vehicle",
    "Area_accident_occured", "Age_band_of_driver", "Driving_experience",
    "Number_of_vehicles_involved", "Time_Period", "Types_of_Junction",
    "Lanes_or_Medians", "Vehicle_movement","Light_conditions"
]
    keep_features_target = keep_features + ["Accident_severity"]
    df = df[keep_features_target]

    # Separate features and target
    X = df.drop(['Accident_severity'], axis=1)
    y = df['Accident_severity']
    
    print(f"\n✓ Features: {X.shape[1]} columns")
    print(f"✓ Target distribution:")
    print(y.value_counts())
    print(f"\n✓ Class percentages:")
    print((y.value_counts(normalize=True) * 100).round(2))
    
    return X, y

def encode_features(X, y):
    """Encode all categorical variables"""
    print("\n" + "=" * 80)
    print("STEP 2: ENCODING CATEGORICAL VARIABLES")
    print("=" * 80)
    
    X_encoded = X.copy()
    label_encoders = {}
    
    # Encode each categorical column
    for column in X_encoded.columns:
        if X_encoded[column].dtype == 'object':
            le = LabelEncoder()
            X_encoded[column] = le.fit_transform(X_encoded[column].astype(str))
            label_encoders[column] = le
            print(f"✓ Encoded: {column:.<40} ({len(le.classes_)} classes)")
    
    # Encode target variable
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)
    label_encoders['target'] = target_encoder
    
    print(f"\n✓ Total encoders created: {len(label_encoders)}")
    print(f"✓ Target classes mapped to: {dict(enumerate(target_encoder.classes_))}")
    
    return X_encoded, y_encoded, label_encoders

def split_and_balance_data(X, y):
    """Split data and apply SMOTE for balancing"""
    print("\n" + "=" * 80)
    print("STEP 3: SPLITTING DATA (70% Train, 30% Test)")
    print("=" * 80)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"✓ Training set: {X_train.shape[0]} samples")
    print(f"✓ Test set: {X_test.shape[0]} samples")
    
    # Apply SMOTE
    print("\n" + "=" * 80)
    print("STEP 4: APPLYING SMOTE (Balancing Classes)")
    print("=" * 80)
    
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    print(f"\n📊 Before SMOTE:")
    unique, counts = np.unique(y_train, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"   Class {u}: {c:>5} samples")
    
    print(f"\n📊 After SMOTE:")
    unique, counts = np.unique(y_train_balanced, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"   Class {u}: {c:>5} samples")
    
    return X_train_balanced, X_test, y_train_balanced, y_test

def train_random_forest_model(X_train, y_train):
    """Train Random Forest classifier"""
    print("\n" + "=" * 80)
    print("STEP 5: TRAINING RANDOM FOREST MODEL")
    print("=" * 80)
    
    model = RandomForestClassifier(
        n_estimators=150,        # Number of trees in the forest
        max_depth=12,            # Maximum depth of each tree
        min_samples_split=5,     # Minimum samples to split a node
        min_samples_leaf=2,      # Minimum samples in leaf node
        max_features='sqrt',     # Number of features for best split
        random_state=42,
        class_weight='balanced', # Handle any remaining class imbalance
        n_jobs=-1,               # Use all CPU cores
        verbose=1                # Show progress
    )
    
    print("🔄 Training in progress...")
    print(f"   - Trees: 150")
    print(f"   - Max Depth: 12")
    print(f"   - Class Weight: Balanced")
    print(f"   - Using all CPU cores")
    
    model.fit(X_train, y_train)
    print("\n✓ Model training completed successfully!")
    
    return model

def evaluate_model(model, X_test, y_test, target_encoder):
    """Comprehensive model evaluation"""
    print("\n" + "=" * 80)
    print("STEP 6: MODEL EVALUATION")
    print("=" * 80)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    
    print(f"\n🎯 OVERALL PERFORMANCE:")
    print(f"   Accuracy:  {accuracy*100:.2f}%")
    print(f"   Precision: {precision*100:.2f}%")
    print(f"   Recall:    {recall*100:.2f}%")
    print(f"   F1-Score:  {f1*100:.2f}%")
    
    print(f"\n📊 DETAILED CLASSIFICATION REPORT:")
    print("-" * 80)
    print(classification_report(y_test, y_pred, 
                              target_names=target_encoder.classes_,
                              digits=4))
    
    print(f"\n🔢 CONFUSION MATRIX:")
    print("-" * 80)
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, 
                         index=[f"True {c}" for c in target_encoder.classes_], 
                         columns=[f"Pred {c}" for c in target_encoder.classes_])
    print(cm_df)
    
    # Per-class accuracy
    print(f"\n📈 PER-CLASS ACCURACY:")
    print("-" * 80)
    for i, class_name in enumerate(target_encoder.classes_):
        class_accuracy = cm[i, i] / cm[i].sum() * 100
        print(f"   {class_name:.<30} {class_accuracy:>6.2f}%")
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall
    }

def get_feature_importance(model, feature_names):
    """Display feature importance"""
    print("\n" + "=" * 80)
    print("STEP 7: FEATURE IMPORTANCE ANALYSIS")
    print("=" * 80)
    
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\n🔝 TOP 15 MOST IMPORTANT FEATURES:")
    print("-" * 80)
    for idx, row in feature_importance.head(15).iterrows():
        bar_length = int(row['Importance'] * 50)
        bar = '█' * bar_length
        print(f"{row['Feature']:.<35} {row['Importance']:.4f} {bar}")
    
    return feature_importance

def save_artifacts(model, label_encoders, feature_names, metrics):
    """Save model and all necessary artifacts"""
    print("\n" + "=" * 80)
    print("STEP 8: SAVING MODEL AND ARTIFACTS")
    print("=" * 80)
    
    # Save Random Forest model
    with open('random_forest_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("✓ Saved: random_forest_model.pkl")
    
    # Save label encoders
    with open('label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)
    print("✓ Saved: label_encoders.pkl")
    
    # Save feature information for frontend
    feature_info = {
        'features': feature_names,
        'categorical_features': {
            col: list(label_encoders[col].classes_) 
            for col in feature_names if col in label_encoders
        },
        'model_metrics': metrics
    }
    
    with open('feature_info.pkl', 'wb') as f:
        pickle.dump(feature_info, f)
    print("✓ Saved: feature_info.pkl")
    
    # Save as JSON for easy reading
    feature_info_json = {
        'features': feature_names,
        'categorical_features': {
            col: list(label_encoders[col].classes_) 
            for col in feature_names if col in label_encoders
        },
        'model_metrics': {k: float(v) for k, v in metrics.items()},
        'target_classes': list(label_encoders['target'].classes_),
        'model_type': 'Random Forest Classifier',
        'n_estimators': 150,
        'max_depth': 12
    }
    
    with open('model_info.json', 'w') as f:
        json.dump(feature_info_json, f, indent=2)
    print("✓ Saved: model_info.json")
    
    print(f"\n✓ All artifacts saved successfully!")

def main():
    """Main training pipeline"""
    print("\n" + "=" * 100)
    print(" " * 30 + "ROAD ACCIDENT SEVERITY PREDICTION")
    print(" " * 30 + "RANDOM FOREST MODEL TRAINING PIPELINE")
    print("=" * 100 + "\n")
    
    # Load data
    X, y = load_and_preprocess_data('cleaned_data.csv')
    
    # Encode variables
    X_encoded, y_encoded, label_encoders = encode_features(X, y)
    
    # Split and balance
    X_train, X_test, y_train, y_test = split_and_balance_data(X_encoded, y_encoded)
    
    # Train model
    model = train_random_forest_model(X_train, y_train)
    
    # Evaluate
    metrics = evaluate_model(model, X_test, y_test, label_encoders['target'])
    
    # Feature importance
    feature_importance = get_feature_importance(model, list(X.columns))
    
    # Save everything
    save_artifacts(model, label_encoders, list(X.columns), metrics)
    
    print("\n" + "=" * 100)
    print(" " * 35 + "✅ TRAINING COMPLETE!")
    print("=" * 100)
    print(f"\n📁 Generated Files:")
    print(f"   1. random_forest_model.pkl   - Trained Random Forest model")
    print(f"   2. label_encoders.pkl        - Feature encoders")
    print(f"   3. feature_info.pkl          - Feature metadata")
    print(f"   4. model_info.json           - Human-readable model info")
    print(f"\n🎯 Model Performance Summary:")
    print(f"   Model Type: Random Forest Classifier")
    print(f"   Trees: 150, Max Depth: 12")
    print(f"   Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"   F1-Score:  {metrics['f1_score']*100:.2f}%")
    print(f"   Precision: {metrics['precision']*100:.2f}%")
    print(f"   Recall:    {metrics['recall']*100:.2f}%")
    print(f"\n🚀 Ready for deployment!")
    print("=" * 100 + "\n")

if __name__ == "__main__":
    main()

