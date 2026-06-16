import io
import json
import os
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DRUG_DB_PATH = os.path.join(PROJECT_ROOT, "data", "drug_reference.json")

_drug_db = None
_ocr_engine = None


def _load_drug_db():
    global _drug_db
    if _drug_db is None:
        with open(DRUG_DB_PATH, "r", encoding="utf-8") as f:
            _drug_db = json.load(f)
    return _drug_db


def _get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
            _ocr_engine = RapidOCR()
        except Exception:
            _ocr_engine = False
    return _ocr_engine if _ocr_engine is not False else None


def _clean_ocr_text(text):
    """Fix common OCR formatting issues in prescription text."""
    cleaned = text
    replacements = [
        (r"cap\.\s*", " "),
        (r"tab\.\s*", " "),
        (r"vitamin\s*d\s*3\s*60", "vitamin d3 60"),
        (r"vitamin\s*d\s*360", "vitamin d3 60"),
        (r"vitamind\s*3", "vitamin d3"),
        (r"vitamind3", "vitamin d3"),
        (r"vitamind360", "vitamin d3"),
        (r"aftermeals", "after meals"),
        (r"lowsalt", "low salt"),
        (r"lowsugar", "low sugar"),
        (r"lowfat", "low fat"),
    ]
    for pattern, repl in replacements:
        cleaned = re.sub(pattern, repl, cleaned, flags=re.IGNORECASE)
    return cleaned


def _normalize(text):
    cleaned = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def extract_text_from_prescription_image(image_bytes, filename=""):
    """Extract readable text from a prescription image using OCR."""
    if not image_bytes:
        return ""

    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        try:
            import google.generativeai as genai
            from PIL import Image
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            img = Image.open(io.BytesIO(image_bytes))
            response = model.generate_content([
                "Read this medical prescription. List every medication with its dosage, one per line. "
                "Format: DrugName Dosage (e.g. Amlodipine 5mg). Only list medicines.",
                img,
            ])
            if response and response.text:
                return _clean_ocr_text(response.text.strip())
        except Exception:
            pass

    ocr = _get_ocr_engine()
    if ocr:
        try:
            result, _ = ocr(image_bytes)
            if result:
                lines = [line[1] for line in result if line[1].strip()]
                return _clean_ocr_text("\n".join(lines))
        except Exception:
            pass

    try:
        import pytesseract
        from PIL import Image
        img = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(img)
        if text.strip():
            return _clean_ocr_text(text.strip())
    except Exception:
        pass

    return ""


def _extract_dosage_for_drug(text, drug_name):
    pattern = re.compile(
        rf"(?:tab\.?|cap\.?)?\s*{re.escape(drug_name)}[^.\n]*?(\d+\s*(?:mg|mcg|iu|g|ml)[^.\n]*)",
        re.IGNORECASE,
    )
    match = pattern.search(text)
    if match:
        return match.group(0).strip()
    for line in text.splitlines():
        if drug_name.lower() in line.lower():
            return line.strip()
    return ""


def extract_drugs_from_text(text):
    """Detect known drugs from free-text or OCR prescription input."""
    if not text or not text.strip():
        return []

    db = _load_drug_db()
    cleaned = _clean_ocr_text(text)
    text_norm = _normalize(cleaned)
    found = []
    seen = set()

    sorted_drugs = sorted(db["drugs"].items(), key=lambda x: -max(len(a) for a in x[1]["aliases"]))

    for drug_id, info in sorted_drugs:
        matched = False
        for alias in info["aliases"]:
            alias_norm = _normalize(alias)
            pattern = r"\b" + re.escape(alias_norm) + r"\b"
            if re.search(pattern, text_norm):
                matched = True
                break
        if drug_id == "vitamin_d3" and not matched:
            if re.search(r"vitamin\s*d", text_norm) and re.search(r"\bd\s*3\b|\bd3\b|60\s*000\s*iu", text_norm):
                matched = True
        if matched and drug_id not in seen:
            seen.add(drug_id)
            dosage = _extract_dosage_for_drug(cleaned, info["display_name"].split()[0])
            found.append({
                "id": drug_id,
                "name": info["display_name"],
                "category": info["category"],
                "uses": info.get("uses", ""),
                "dosage": dosage,
            })

    return found


def get_drug_details(drug_id):
    db = _load_drug_db()
    info = db["drugs"].get(drug_id)
    if not info:
        return None
    return {
        "id": drug_id,
        "name": info["display_name"],
        "category": info["category"],
        "uses": info.get("uses", ""),
        "side_effects": info["side_effects"],
        "warnings": info["warnings"],
    }


def check_allergy_conflicts(drug_ids, allergies_text):
    if not allergies_text or allergies_text.lower() in ("none", "n/a", ""):
        return []

    allergy_map = {
        "penicillin": ["amoxicillin"],
        "amoxicillin": ["amoxicillin"],
        "sulfa": [],
        "lisinopril": ["lisinopril"],
        "ace inhibitor": ["lisinopril"],
    }

    alerts = []
    allergies_lower = allergies_text.lower()
    for allergy, affected_drugs in allergy_map.items():
        if allergy in allergies_lower:
            for drug_id in drug_ids:
                if drug_id in affected_drugs:
                    info = _load_drug_db()["drugs"][drug_id]
                    alerts.append({
                        "drug": info["display_name"],
                        "allergy": allergy.title(),
                        "severity": "Major",
                        "description": (
                            f"Patient has a documented allergy to {allergy}. "
                            f"{info['display_name']} may cause a severe allergic reaction."
                        ),
                    })
    return alerts


def check_drug_interactions(drug_ids):
    db = _load_drug_db()
    drug_set = set(drug_ids)
    interactions = []

    for entry in db["interactions"]:
        pair = set(entry["drugs"])
        if pair.issubset(drug_set):
            names = [
                db["drugs"][d]["display_name"]
                for d in entry["drugs"]
                if d in db["drugs"]
            ]
            interactions.append({
                "drugs": names if names else entry["drugs"],
                "severity": entry["severity"],
                "description": entry["description"],
            })

    severity_order = {"Major": 0, "Moderate": 1, "Minor": 2}
    interactions.sort(key=lambda x: severity_order.get(x["severity"], 3))
    return interactions


def build_interaction_matrix(drug_ids):
    """Show interaction status for every drug pair in the prescription."""
    db = _load_drug_db()
    known = {}
    for entry in db["interactions"]:
        pair = tuple(sorted(entry["drugs"]))
        known[pair] = entry

    matrix = []
    for i, drug_a in enumerate(drug_ids):
        for drug_b in drug_ids[i + 1:]:
            pair = tuple(sorted([drug_a, drug_b]))
            name_a = db["drugs"][drug_a]["display_name"]
            name_b = db["drugs"][drug_b]["display_name"]
            if pair in known:
                entry = known[pair]
                matrix.append({
                    "drug_a": name_a,
                    "drug_b": name_b,
                    "severity": entry["severity"],
                    "description": entry["description"],
                    "status": entry["severity"],
                })
            else:
                matrix.append({
                    "drug_a": name_a,
                    "drug_b": name_b,
                    "severity": "None",
                    "description": "No known significant interaction between these two medications.",
                    "status": "Safe",
                })
    return matrix


def analyze_prescription(prescription_text, allergies=None, existing_meds=None):
    """Full medication safety analysis: detect drugs, side effects, interactions, allergy alerts."""
    detected = extract_drugs_from_text(prescription_text)

    if existing_meds:
        for med in existing_meds:
            if isinstance(med, dict) and med.get("id"):
                if med["id"] not in {d["id"] for d in detected}:
                    detected.append(med)
            elif isinstance(med, str):
                for d in extract_drugs_from_text(med):
                    if d["id"] not in {x["id"] for x in detected}:
                        detected.append(d)

    drug_ids = [d["id"] for d in detected]
    drug_profiles = []
    for drug_id in drug_ids:
        details = get_drug_details(drug_id)
        if details:
            dosage = next((d.get("dosage", "") for d in detected if d["id"] == drug_id), "")
            details["dosage"] = dosage
            drug_profiles.append(details)

    interactions = check_drug_interactions(drug_ids)
    interaction_matrix = build_interaction_matrix(drug_ids)
    allergy_alerts = check_allergy_conflicts(drug_ids, allergies or "")

    return {
        "detected_drugs": detected,
        "drug_profiles": drug_profiles,
        "side_effects": drug_profiles,
        "interactions": interactions,
        "interaction_matrix": interaction_matrix,
        "allergy_alerts": allergy_alerts,
        "guideline_notes": [],
        "extracted_text": prescription_text,
        "summary": _build_summary(detected, interactions, allergy_alerts),
    }


def _build_summary(detected, interactions, allergy_alerts):
    if not detected:
        return (
            "No recognized medications were found. Upload a clear prescription photo or type drug names "
            "like: Amlodipine 5mg, Metformin 500mg, Atorvastatin 10mg, Pantoprazole 40mg."
        )

    names = ", ".join(d["name"] for d in detected)
    major = sum(1 for i in interactions if i["severity"] == "Major") + len(allergy_alerts)
    moderate = sum(1 for i in interactions if i["severity"] == "Moderate")

    if major > 0:
        return (
            f"Analyzed {len(detected)} medication(s): {names}. "
            f"Found {major} major safety concern(s). Please review with your doctor or pharmacist."
        )
    if moderate > 0:
        return (
            f"Analyzed {len(detected)} medication(s): {names}. "
            f"Found {moderate} moderate interaction(s). Monitor symptoms and discuss with your care team."
        )
    return (
        f"Analyzed {len(detected)} medication(s): {names}. "
        f"No major interactions detected. See detailed side effects and pair-wise analysis below."
    )
