import os
import sys

def generate_pdfs():
    print("Generating clinical guidelines PDFs...")
    try:
        from fpdf import FPDF
    except ImportError:
        print("fpdf2 not installed, installing it temporarily or writing plain text files as fallback.")
        # Fallback to creating text files that can be read by RAG if PDF generator isn't ready
        os.makedirs("data/medical_guidelines", exist_ok=True)
        write_text_fallback()
        return

    os.makedirs("data/medical_guidelines", exist_ok=True)

    # 1. Hypertension Guidelines
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", "B", 16)
    pdf.cell(0, 10, "WHO Clinical Guidelines for the Treatment of Hypertension", ln=True, align="C")
    pdf.ln(5)
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 10, "1. Diagnosis and Classification", ln=True)
    pdf.set_font("helvetica", "", 10)
    pdf.multi_cell(0, 5, (
        "Hypertension is diagnosed when a patient's systolic blood pressure (SBP) is >= 140 mmHg "
        "and/or their diastolic blood pressure (DBP) is >= 90 mmHg on two separate visits.\n"
        "- Normal Blood Pressure: SBP < 120 mmHg and DBP < 80 mmHg.\n"
        "- Prehypertension: SBP 120-139 mmHg or DBP 80-89 mmHg.\n"
        "- Stage 1 Hypertension: SBP 140-159 mmHg or DBP 90-99 mmHg.\n"
        "- Stage 2 Hypertension: SBP >= 160 mmHg or DBP >= 100 mmHg.\n"
        "- Hypertensive Crisis: SBP > 180 mmHg and/or DBP > 120 mmHg (requires immediate critical triage)."
    ))
    pdf.ln(5)
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 10, "2. Pharmacological Treatment Protocols", ln=True)
    pdf.set_font("helvetica", "", 10)
    pdf.multi_cell(0, 5, (
        "First-line pharmacological treatments include:\n"
        "- ACE Inhibitors (e.g., Lisinopril, 10-40 mg daily) - recommended for patients with diabetes or chronic kidney disease.\n"
        "- Angiotensin Receptor Blockers (ARBs, e.g., Losartan, 50-100 mg daily).\n"
        "- Calcium Channel Blockers (CCBs, e.g., Amlodipine, 5-10 mg daily) - highly effective in elderly patients.\n"
        "- Thiazide Diuretics (e.g., Hydrochlorothiazide, 12.5-25 mg daily).\n"
        "Beta-blockers (e.g., Metoprolol) are generally reserved for patients with comorbid coronary artery disease or heart failure."
    ))
    pdf.ln(5)
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 10, "3. Lifestyle and Follow-up Recommendations", ln=True)
    pdf.set_font("helvetica", "", 10)
    pdf.multi_cell(0, 5, (
        "Lifestyle modifications are mandatory for all hypertension levels:\n"
        "- Sodium restriction to < 2.0 grams per day.\n"
        "- DASH (Dietary Approaches to Stop Hypertension) diet, rich in fruits, vegetables, and low-fat dairy.\n"
        "- Regular aerobic physical activity (at least 30 minutes daily, 5 days a week).\n"
        "- Limitation of alcohol consumption and cessation of smoking.\n"
        "- Follow-up: Re-evaluate Stage 1 hypertension within 3-4 weeks; Stage 2 hypertension within 1-2 weeks."
    ))
    pdf.output("data/medical_guidelines/Hypertension_WHO_Guidelines.pdf")
    print("Generated: Hypertension_WHO_Guidelines.pdf")

    # 2. Diabetes Management Protocol
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", "B", 16)
    pdf.cell(0, 10, "Clinical Protocol for Type 2 Diabetes Mellitus Management", ln=True, align="C")
    pdf.ln(5)
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 10, "1. Diagnostic Thresholds", ln=True)
    pdf.set_font("helvetica", "", 10)
    pdf.multi_cell(0, 5, (
        "Type 2 Diabetes Mellitus (T2DM) is diagnosed based on any of the following criteria:\n"
        "- HbA1c >= 6.5%.\n"
        "- Fasting Plasma Glucose (FPG) >= 126 mg/dL (7.0 mmol/L).\n"
        "- 2-hour oral glucose tolerance test (OGTT) value >= 200 mg/dL.\n"
        "- Random plasma glucose >= 200 mg/dL in a patient with classic symptoms of hyperglycemia (polyuria, polydipsia, unexplained weight loss)."
    ))
    pdf.ln(5)
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 10, "2. Treatment Algorithm", ln=True)
    pdf.set_font("helvetica", "", 10)
    pdf.multi_cell(0, 5, (
        "First-Line Therapy: Metformin (initial dose 500 mg once or twice daily, titrate up to 2000 mg daily as tolerated). "
        "Metformin should be continued unless contraindicated (e.g., eGFR < 30 mL/min/1.73m2).\n"
        "Combination Therapy (if HbA1c target not met after 3 months):\n"
        "- Add SGLT2 Inhibitors (e.g., Empagliflozin, Dapagliflozin) for patients with heart failure or diabetic nephropathy.\n"
        "- Add GLP-1 Receptor Agonists (e.g., Semaglutide, Liraglutide) for patients with high cardiovascular risk.\n"
        "- Add DPP-4 Inhibitors (e.g., Sitagliptin) or Sulfonylureas (e.g., Glimepiride) as alternative cost-effective options.\n"
        "- Insulin therapy should be initiated if HbA1c is > 9.0% with severe symptoms."
    ))
    pdf.ln(5)
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 10, "3. Lifestyle and Monitoring", ln=True)
    pdf.set_font("helvetica", "", 10)
    pdf.multi_cell(0, 5, (
        "Lifestyle Recommendations:\n"
        "- Carbohydrate-controlled diet focusing on complex carbohydrates and high-fiber foods.\n"
        "- Moderate-intensity exercise (at least 150 minutes per week, no more than 2 consecutive days without exercise).\n"
        "- Weight management target: 5-7% weight reduction for patients who are overweight or obese.\n"
        "Monitoring Guidelines:\n"
        "- Measure HbA1c every 3 months for patients not meeting targets; twice yearly for stable patients meeting targets.\n"
        "- Annual screening for diabetic nephropathy (urine albumin-to-creatinine ratio), retinopathy (dilated eye exam), and diabetic neuropathy (foot sensory examination)."
    ))
    pdf.output("data/medical_guidelines/Diabetes_Management_Protocol.pdf")
    print("Generated: Diabetes_Management_Protocol.pdf")

    # 3. Pneumonia Clinical Protocol
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", "B", 16)
    pdf.cell(0, 10, "Clinical Protocol for Community-Acquired Pneumonia (CAP)", ln=True, align="C")
    pdf.ln(5)
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 10, "1. Diagnosis and Severity Assessment (CURB-65)", ln=True)
    pdf.set_font("helvetica", "", 10)
    pdf.multi_cell(0, 5, (
        "Community-Acquired Pneumonia is characterized by symptoms such as cough, fever, dyspnea, and pleuritic chest pain, "
        "confirmed by a new infiltrate on a chest radiograph.\n"
        "Triage patients using the CURB-65 Scoring System (1 point for each):\n"
        "- C: Confusion (new disorientation in person, place, or time).\n"
        "- U: Urea > 7 mmol/L (Blood Urea Nitrogen > 19 mg/dL).\n"
        "- R: Respiratory Rate >= 30 breaths per minute.\n"
        "- B: Blood Pressure (Systolic < 90 mmHg or Diastolic <= 60 mmHg).\n"
        "- 65: Age >= 65 years.\n"
        "Triage Guidelines:\n"
        "- Score 0-1: Low risk, treat as outpatient.\n"
        "- Score 2: Moderate risk, consider short-stay inpatient admission or close monitoring.\n"
        "- Score 3-5: High risk, urgent inpatient hospital admission; if score is 4-5, evaluate for Intensive Care Unit (ICU) admission."
    ))
    pdf.ln(5)
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 10, "2. Empirical Antibiotic Regimens", ln=True)
    pdf.set_font("helvetica", "", 10)
    pdf.multi_cell(0, 5, (
        "Outpatient Treatment (No comorbidities):\n"
        "- Amoxicillin 1g three times daily OR Doxycycline 100mg twice daily OR Macrolide (Azithromycin 500mg day 1, then 250mg daily) in areas with macrolide resistance < 25%.\n"
        "Outpatient Treatment (With comorbidities like chronic heart/lung/liver/renal disease, diabetes, alcoholism):\n"
        "- Combination therapy: Amoxicillin/Clavulanate (875/125mg twice daily) PLUS a Macrolide (Azithromycin 500mg daily) OR Doxycycline 100mg twice daily.\n"
        "- Alternative: Respiratory Fluoroquinolone (Levofloxacin 750mg daily or Moxifloxacin 400mg daily).\n"
        "Inpatient Treatment (Non-Severe):\n"
        "- Beta-lactam (Ceftriaxone 1-2g daily or Ampicillin-Sulbactam 1.5-3g every 6 hours) PLUS a Macrolide (Azithromycin 500mg daily) OR Respiratory Fluoroquinolone monotherapy."
    ))
    pdf.ln(5)
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 10, "3. Supportive Care and Follow-up", ln=True)
    pdf.set_font("helvetica", "", 10)
    pdf.multi_cell(0, 5, (
        "Supportive Care:\n"
        "- Supplemental oxygen to maintain oxygen saturation >= 92% (or 88-92% in patients with COPD).\n"
        "- Adequate hydration (intravenous fluids if patient is unable to tolerate oral intake).\n"
        "- Antipyretics (Acetaminophen or Ibuprofen) for fever control.\n"
        "Follow-up:\n"
        "- Clinical assessment within 48-72 hours of starting antibiotics to monitor treatment response.\n"
        "- Repeat chest radiograph in 4-6 weeks for patients over 50 years or smokers to confirm resolution and rule out underlying malignancy."
    ))
    pdf.output("data/medical_guidelines/Pneumonia_Clinical_Protocol.pdf")
    print("Generated: Pneumonia_Clinical_Protocol.pdf")

def write_text_fallback():
    # Helper to write txt files in case fpdf is missing
    print("Writing plain text fallbacks for clinical guidelines...")
    os.makedirs("data/medical_guidelines", exist_ok=True)
    
    with open("data/medical_guidelines/Hypertension_WHO_Guidelines.txt", "w", encoding="utf-8") as f:
        f.write(
            "WHO Clinical Guidelines for the Treatment of Hypertension\n\n"
            "1. Diagnosis and Classification\n"
            "Hypertension is diagnosed when a patient's systolic blood pressure (SBP) is >= 140 mmHg "
            "and/or their diastolic blood pressure (DBP) is >= 90 mmHg on two separate visits.\n"
            "- Normal Blood Pressure: SBP < 120 mmHg and DBP < 80 mmHg.\n"
            "- Prehypertension: SBP 120-139 mmHg or DBP 80-89 mmHg.\n"
            "- Stage 1 Hypertension: SBP 140-159 mmHg or DBP 90-99 mmHg.\n"
            "- Stage 2 Hypertension: SBP >= 160 mmHg or DBP >= 100 mmHg.\n"
            "- Hypertensive Crisis: SBP > 180 mmHg and/or DBP > 120 mmHg (requires immediate critical triage).\n\n"
            "2. Pharmacological Treatment Protocols\n"
            "First-line pharmacological treatments include:\n"
            "- ACE Inhibitors (e.g., Lisinopril, 10-40 mg daily) - recommended for patients with diabetes or chronic kidney disease.\n"
            "- Angiotensin Receptor Blockers (ARBs, e.g., Losartan, 50-100 mg daily).\n"
            "- Calcium Channel Blockers (CCBs, e.g., Amlodipine, 5-10 mg daily) - highly effective in elderly patients.\n"
            "- Thiazide Diuretics (e.g., Hydrochlorothiazide, 12.5-25 mg daily).\n"
            "Beta-blockers (e.g., Metoprolol) are generally reserved for patients with comorbid coronary artery disease or heart failure.\n\n"
            "3. Lifestyle and Follow-up Recommendations\n"
            "Lifestyle modifications are mandatory for all hypertension levels:\n"
            "- Sodium restriction to < 2.0 grams per day.\n"
            "- DASH (Dietary Approaches to Stop Hypertension) diet, rich in fruits, vegetables, and low-fat dairy.\n"
            "- Regular aerobic physical activity (at least 30 minutes daily, 5 days a week).\n"
            "- Limitation of alcohol consumption and cessation of smoking.\n"
            "- Follow-up: Re-evaluate Stage 1 hypertension within 3-4 weeks; Stage 2 hypertension within 1-2 weeks."
        )
    
    with open("data/medical_guidelines/Diabetes_Management_Protocol.txt", "w", encoding="utf-8") as f:
        f.write(
            "Clinical Protocol for Type 2 Diabetes Mellitus Management\n\n"
            "1. Diagnostic Thresholds\n"
            "Type 2 Diabetes Mellitus (T2DM) is diagnosed based on any of the following criteria:\n"
            "- HbA1c >= 6.5%.\n"
            "- Fasting Plasma Glucose (FPG) >= 126 mg/dL (7.0 mmol/L).\n"
            "- 2-hour oral glucose tolerance test (OGTT) value >= 200 mg/dL.\n"
            "- Random plasma glucose >= 200 mg/dL in a patient with classic symptoms of hyperglycemia (polyuria, polydipsia, unexplained weight loss).\n\n"
            "2. Treatment Algorithm\n"
            "First-Line Therapy: Metformin (initial dose 500 mg once or twice daily, titrate up to 2000 mg daily as tolerated). "
            "Metformin should be continued unless contraindicated (e.g., eGFR < 30 mL/min/1.73m2).\n"
            "Combination Therapy (if HbA1c target not met after 3 months):\n"
            "- Add SGLT2 Inhibitors (e.g., Empagliflozin, Dapagliflozin) for patients with heart failure or diabetic nephropathy.\n"
            "- Add GLP-1 Receptor Agonists (e.g., Semaglutide, Liraglutide) for patients with high cardiovascular risk.\n"
            "- Add DPP-4 Inhibitors (e.g., Sitagliptin) or Sulfonylureas (e.g., Glimepiride) as alternative cost-effective options.\n"
            "- Insulin therapy should be initiated if HbA1c is > 9.0% with severe symptoms.\n\n"
            "3. Lifestyle and Monitoring\n"
            "Lifestyle Recommendations:\n"
            "- Carbohydrate-controlled diet focusing on complex carbohydrates and high-fiber foods.\n"
            "- Moderate-intensity exercise (at least 150 minutes per week, no more than 2 consecutive days without exercise).\n"
            "- Weight management target: 5-7% weight reduction for patients who are overweight or obese.\n"
            "Monitoring Guidelines:\n"
            "- Measure HbA1c every 3 months for patients not meeting targets; twice yearly for stable patients meeting targets.\n"
            "- Annual screening for diabetic nephropathy (urine albumin-to-creatinine ratio), retinopathy (dilated eye exam), and diabetic neuropathy (foot sensory examination)."
        )

    with open("data/medical_guidelines/Pneumonia_Clinical_Protocol.txt", "w", encoding="utf-8") as f:
        f.write(
            "Clinical Protocol for Community-Acquired Pneumonia (CAP)\n\n"
            "1. Diagnosis and Severity Assessment (CURB-65)\n"
            "Community-Acquired Pneumonia is characterized by symptoms such as cough, fever, dyspnea, and pleuritic chest pain, "
            "confirmed by a new infiltrate on a chest radiograph.\n"
            "Triage patients using the CURB-65 Scoring System (1 point for each):\n"
            "- C: Confusion (new disorientation in person, place, or time).\n"
            "- U: Urea > 7 mmol/L (Blood Urea Nitrogen > 19 mg/dL).\n"
            "- R: Respiratory Rate >= 30 breaths per minute.\n"
            "- B: Blood Pressure (Systolic < 90 mmHg or Diastolic <= 60 mmHg).\n"
            "- 65: Age >= 65 years.\n"
            "Triage Guidelines:\n"
            "- Score 0-1: Low risk, treat as outpatient.\n"
            "- Score 2: Moderate risk, consider short-stay inpatient admission or close monitoring.\n"
            "- Score 3-5: High risk, urgent inpatient hospital admission; if score is 4-5, evaluate for Intensive Care Unit (ICU) admission.\n\n"
            "2. Empirical Antibiotic Regimens\n"
            "Outpatient Treatment (No comorbidities):\n"
            "- Amoxicillin 1g three times daily OR Doxycycline 100mg twice daily OR Macrolide (Azithromycin 500mg day 1, then 250mg daily) in areas with macrolide resistance < 25%.\n"
            "Outpatient Treatment (With comorbidities like chronic heart/lung/liver/renal disease, diabetes, alcoholism):\n"
            "- Combination therapy: Amoxicillin/Clavulanate (875/125mg twice daily) PLUS a Macrolide (Azithromycin 500mg daily) OR Doxycycline 100mg twice daily.\n"
            "- Alternative: Respiratory Fluoroquinolone (Levofloxacin 750mg daily or Moxifloxacin 400mg daily).\n"
            "Inpatient Treatment (Non-Severe):\n"
            "- Beta-lactam (Ceftriaxone 1-2g daily or Ampicillin-Sulbactam 1.5-3g every 6 hours) PLUS a Macrolide (Azithromycin 500mg daily) OR Respiratory Fluoroquinolone monotherapy.\n\n"
            "3. Supportive Care and Follow-up\n"
            "Supportive Care:\n"
            "- Supplemental oxygen to maintain oxygen saturation >= 92% (or 88-92% in patients with COPD).\n"
            "- Adequate hydration (intravenous fluids if patient is unable to tolerate oral intake).\n"
            "- Antipyretics (Acetaminophen or Ibuprofen) for fever control.\n"
            "Follow-up:\n"
            "- Clinical assessment within 48-72 hours of starting antibiotics to monitor treatment response.\n"
            "- Repeat chest radiograph in 4-6 weeks for patients over 50 years or smokers to confirm resolution and rule out underlying malignancy."
        )

if __name__ == "__main__":
    generate_pdfs()
