"""
Reset HealthAI database to a clean fresh state.
Keeps ONE main user per role. Removes all test/demo data.
"""
import os
import sys
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.database import (
    SessionLocal, User, Patient, HealthRecord, TriageRecord,
    Appointment, PatientPrescription, hash_password, init_db, DB_PATH,
)

DEFAULT_PASSWORD = "password123"

MAIN_USERS = [
    {"username": "patient1", "role": "Patient", "full_name": "John Doe"},
    {"username": "doctor1", "role": "Doctor", "full_name": "Dr. Sarah Connor"},
    {"username": "nurse1", "role": "Nurse", "full_name": "Nurse Clara Barton"},
    {"username": "admin1", "role": "Administrator", "full_name": "Admin Alex Smith"},
    {"username": "superadmin", "role": "Super Admin", "full_name": "Super Admin Chief"},
]


def clear_uploaded_files():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for folder in ("uploaded_reports", "uploaded_prescriptions"):
        path = os.path.join(project_root, "data", folder)
        if os.path.isdir(path):
            for name in os.listdir(path):
                file_path = os.path.join(path, name)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            print(f"Cleared {path}")


def reset_database():
    print("Resetting HealthAI database...")
    db = SessionLocal()
    try:
        db.query(PatientPrescription).delete()
        db.query(HealthRecord).delete()
        db.query(TriageRecord).delete()
        db.query(Appointment).delete()
        db.query(Patient).delete()
        db.query(User).delete()
        db.commit()
        print("All existing users and records removed.")

        pwd = hash_password(DEFAULT_PASSWORD)
        for u in MAIN_USERS:
            db.add(User(
                username=u["username"],
                password_hash=pwd,
                role=u["role"],
                full_name=u["full_name"],
            ))
        db.commit()

        patient_user = db.query(User).filter(User.username == "patient1").first()
        db.add(Patient(
            user_id=patient_user.id,
            name="John Doe",
            age=45,
            gender="Male",
            contact="555-0100",
            medical_history="None",
            allergies="None",
        ))
        db.commit()
        print("Fresh database seeded with one user per role.")
    finally:
        db.close()

    clear_uploaded_files()
    print("\n=== Fresh Login Credentials (password for all: password123) ===")
    for u in MAIN_USERS:
        print(f"  {u['role']:15} -> username: {u['username']}")
    print(f"\nDatabase: {DB_PATH}")


if __name__ == "__main__":
    reset_database()
