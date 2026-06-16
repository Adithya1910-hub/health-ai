import re

def classify_triage(symptoms: str, age: int, systolic_bp: int = None, diastolic_bp: int = None, 
                    heart_rate: int = None, temperature: float = None, lab_results: str = "") -> dict:
    """
    Classify a patient's condition into a priority level (Critical, High, Medium, Low)
    and recommend an appropriate medical specialist or department.
    """
    symptoms_lower = symptoms.lower()
    lab_lower = lab_results.lower()
    
    # 1. Check for CRITICAL criteria
    critical_triggers = [
        "chest pain", "crushing chest", "unconscious", "confusion", "difficulty breathing",
        "severe dyspnea", "shortness of breath", "unable to breathe", "loss of speech", "paralysis"
    ]
    
    is_critical_symptom = any(trigger in symptoms_lower for trigger in critical_triggers)
    
    # Vitals check
    is_critical_vitals = False
    reasons = []
    
    if systolic_bp is not None and systolic_bp >= 180:
        is_critical_vitals = True
        reasons.append(f"Severe systolic BP: {systolic_bp} mmHg")
    if diastolic_bp is not None and diastolic_bp >= 120:
        is_critical_vitals = True
        reasons.append(f"Severe diastolic BP: {diastolic_bp} mmHg")
    if heart_rate is not None and (heart_rate > 130 or heart_rate < 40):
        is_critical_vitals = True
        reasons.append(f"Severe heart rate: {heart_rate} bpm")
    if temperature is not None and temperature >= 40.0:
        is_critical_vitals = True
        reasons.append(f"Extreme body temperature: {temperature}°C")
        
    # Check lab result keywords for critical status (e.g. cardiac markers)
    if "troponin" in lab_lower and "high" in lab_lower or "elevated" in lab_lower:
        is_critical_vitals = True
        reasons.append("Elevated Cardiac Troponin")

    if is_critical_symptom or is_critical_vitals:
        if is_critical_symptom:
            reasons.append("Critical symptom reported (e.g., chest pain/respiratory distress)")
        
        # Decide department
        dept = "Emergency Medicine / Cardiology" if "chest" in symptoms_lower else "Emergency Medicine"
        if "breathing" in symptoms_lower or "dyspnea" in symptoms_lower:
            dept = "Emergency Medicine / Pulmonology"
            
        return {
            "priority_level": "Critical",
            "recommended_department": dept,
            "triage_reason": "; ".join(reasons) if reasons else "Severe acute symptoms requiring immediate attention",
            "time_estimate": "Immediate attention (< 10 mins)"
        }
        
    # 2. Check for HIGH criteria
    high_triggers = [
        "severe headache", "high fever", "coughing blood", "severe abdominal pain", 
        "persistent vomiting", "dizziness", "moderate dyspnea"
    ]
    
    is_high_symptom = any(trigger in symptoms_lower for trigger in high_triggers)
    is_high_vitals = False
    
    if systolic_bp is not None and (160 <= systolic_bp < 180):
        is_high_vitals = True
        reasons.append(f"Elevated Stage 2 systolic BP: {systolic_bp} mmHg")
    if diastolic_bp is not None and (100 <= diastolic_bp < 120):
        is_high_vitals = True
        reasons.append(f"Elevated Stage 2 diastolic BP: {diastolic_bp} mmHg")
    if heart_rate is not None and (110 <= heart_rate <= 130):
        is_high_vitals = True
        reasons.append(f"Tachycardia heart rate: {heart_rate} bpm")
    if temperature is not None and (38.8 <= temperature < 40.0):
        is_high_vitals = True
        reasons.append(f"High fever temperature: {temperature}°C")
        
    if is_high_symptom or is_high_vitals:
        if is_high_symptom:
            reasons.append("High-severity symptom reported (e.g., high fever, dizziness)")
            
        # Department recommendation
        dept = "General Medicine"
        if "fever" in symptoms_lower or "cough" in symptoms_lower:
            dept = "Pulmonology / Infectious Diseases"
        elif "headache" in symptoms_lower or "bp" in reasons[0].lower() if reasons else False:
            dept = "Cardiology"
        elif "abdominal" in symptoms_lower:
            dept = "Gastroenterology"
            
        return {
            "priority_level": "High",
            "recommended_department": dept,
            "triage_reason": "; ".join(reasons) if reasons else "High-risk symptoms requiring evaluation",
            "time_estimate": "See doctor within 1 hour"
        }

    # 3. Check for MEDIUM criteria
    medium_triggers = [
        "cough", "fever", "fatigue", "nausea", "frequent urination", "increased thirst", "joint pain", "pale skin"
    ]
    
    is_medium_symptom = any(trigger in symptoms_lower for trigger in medium_triggers)
    is_medium_vitals = False
    
    if systolic_bp is not None and (140 <= systolic_bp < 160):
        is_medium_vitals = True
        reasons.append(f"Stage 1 systolic BP: {systolic_bp} mmHg")
    if diastolic_bp is not None and (90 <= diastolic_bp < 100):
        is_medium_vitals = True
        reasons.append(f"Stage 1 diastolic BP: {diastolic_bp} mmHg")
    if temperature is not None and (37.5 <= temperature < 38.8):
        is_medium_vitals = True
        reasons.append(f"Low-grade fever temperature: {temperature}°C")
        
    if "blood sugar" in lab_lower or "glucose" in lab_lower:
        # Extract number if possible
        sugar_val = re.findall(r'\d+', lab_lower)
        if sugar_val and int(sugar_val[0]) > 140:
            is_medium_vitals = True
            reasons.append(f"Elevated blood glucose: {sugar_val[0]} mg/dL")
            
    if is_medium_symptom or is_medium_vitals:
        if is_medium_symptom:
            reasons.append("Moderate symptoms (e.g., cough, fatigue)")
            
        dept = "General Medicine"
        if "urination" in symptoms_lower or "thirst" in symptoms_lower or "sugar" in lab_lower:
            dept = "Endocrinology"
        elif "pale" in symptoms_lower or "anemia" in symptoms_lower or "hemoglobin" in lab_lower:
            dept = "Hematology"
        elif "joint" in symptoms_lower:
            dept = "Rheumatology / Orthopedics"
            
        return {
            "priority_level": "Medium",
            "recommended_department": dept,
            "triage_reason": "; ".join(reasons) if reasons else "Moderate clinical signs requiring standard review",
            "time_estimate": "Schedule appointment within 24-48 hours"
        }

    # 4. Fallback to LOW criteria
    dept = "General Medicine"
    if age > 65:
        dept = "Geriatrics"
        
    return {
        "priority_level": "Low",
        "recommended_department": dept,
        "triage_reason": "Routine checkup or minor, stable symptoms",
        "time_estimate": "Routine checkup or next available slot"
    }
