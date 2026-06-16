import os
import pickle
import numpy as np
import pandas as pd

MODEL_PATH = "data/models/disease_risk_model.pkl"

def get_model():
    """Load model if exists, otherwise train it first."""
    if not os.path.exists(MODEL_PATH):
        print("Model file not found. Running training script...")
        from scripts.train_ml import train_model
        train_model()
        
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def predict_disease_risk(age: int, gender: str, systolic_bp: int, diastolic_bp: int,
                         heart_rate: int, temperature: float, fasting_blood_sugar: int,
                         cholesterol: int, hemoglobin: float, has_cough: bool,
                         has_chest_pain: bool, has_dyspnea: bool, has_fatigue: bool) -> dict:
    """
    Predict potential diseases and health risks using the trained RandomForest model.
    Generates a disease risk score and highlights contributing factors.
    """
    # 1. Format inputs
    gender_num = 1 if gender.lower() == "male" else 0
    cough = 1 if has_cough else 0
    chest_pain = 1 if has_chest_pain else 0
    dyspnea = 1 if has_dyspnea else 0
    fatigue = 1 if has_fatigue else 0
    
    # Load model package
    model_pkg = get_model()
    model = model_pkg['model']
    scaler = model_pkg['scaler']
    feature_names = model_pkg['feature_names']
    target_names = model_pkg['target_names']
    
    # 2. Arrange features in correct order
    input_data = pd.DataFrame([{
        'age': age,
        'gender': gender_num,
        'systolic_bp': systolic_bp,
        'diastolic_bp': diastolic_bp,
        'heart_rate': heart_rate,
        'temperature': temperature,
        'fasting_blood_sugar': fasting_blood_sugar,
        'cholesterol': cholesterol,
        'hemoglobin': hemoglobin,
        'has_cough': cough,
        'has_chest_pain': chest_pain,
        'has_dyspnea': dyspnea,
        'has_fatigue': fatigue
    }])
    
    # Scale features
    input_scaled = scaler.transform(input_data)
    
    # 3. Predict class and probabilities
    pred_class_idx = model.predict(input_scaled)[0]
    probs = model.predict_proba(input_scaled)[0]
    
    # Risk score is the probability of the predicted disease class
    risk_score = float(probs[pred_class_idx])
    predicted_disease = target_names[pred_class_idx]
    
    # Get all potential risks (probabilities > 15%) sorted
    all_risks = []
    for idx, prob in enumerate(probs):
        if prob > 0.10 and target_names[idx] != 'Healthy':
            all_risks.append({
                "disease": target_names[idx],
                "probability": float(prob)
            })
    all_risks = sorted(all_risks, key=lambda x: x['probability'], reverse=True)
    
    # 4. Highlight contributing factors using feature values and random forest importances
    # We find features that deviate from normal and weigh them by feature importances
    importances = model.feature_importances_
    
    contributing_factors = []
    
    # Standard normal ranges:
    # bp: < 120 / < 80
    # temp: < 37.2
    # blood sugar: < 100
    # cholesterol: < 200
    # hemoglobin: > 12.0
    
    # Check deviations
    deviations = {}
    if systolic_bp >= 140 or diastolic_bp >= 90:
        deviations['systolic_bp'] = f"Elevated blood pressure ({systolic_bp}/{diastolic_bp} mmHg)"
        deviations['diastolic_bp'] = f"Elevated blood pressure ({systolic_bp}/{diastolic_bp} mmHg)"
    if temperature >= 37.8:
        deviations['temperature'] = f"Fever ({temperature}°C)"
    if fasting_blood_sugar >= 126:
        deviations['fasting_blood_sugar'] = f"High blood sugar ({fasting_blood_sugar} mg/dL)"
    if cholesterol >= 240:
        deviations['cholesterol'] = f"High cholesterol ({cholesterol} mg/dL)"
    if hemoglobin < 11.5:
        deviations['hemoglobin'] = f"Low hemoglobin ({hemoglobin} g/dL)"
    if cough:
        deviations['has_cough'] = "Active cough reported"
    if chest_pain:
        deviations['has_chest_pain'] = "Active chest pain reported"
    if dyspnea:
        deviations['has_dyspnea'] = "Active dyspnea (shortness of breath)"
    if fatigue:
        deviations['has_fatigue'] = "Fatigue reported"

    # Match deviations with their feature importance weights
    for idx, feat in enumerate(feature_names):
        if feat in deviations:
            contributing_factors.append({
                "factor": deviations[feat],
                "importance_weight": float(importances[idx])
            })
            
    # Sort contributing factors by model importance weight
    contributing_factors = sorted(contributing_factors, key=lambda x: x['importance_weight'], reverse=True)
    # Deduplicate factor descriptions (since sys and dia BP point to the same description)
    seen_factors = set()
    dedup_factors = []
    for f in contributing_factors:
        if f['factor'] not in seen_factors:
            seen_factors.add(f['factor'])
            dedup_factors.append(f)

    # If no deviations, but predicted disease isn't healthy, list top feature importances for that class
    if not dedup_factors and predicted_disease != 'Healthy':
        # Default top features
        dedup_factors = [{"factor": f"Age and demographic factors (Age: {age})", "importance_weight": 0.1}]

    return {
        "predicted_disease": predicted_disease,
        "risk_score": risk_score,
        "all_risks": all_risks if all_risks else [{"disease": "Healthy", "probability": 1.0}],
        "contributing_factors": dedup_factors
    }
