import os
import hashlib
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey, DateTime, Text, event
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DB_PATH = os.path.join(PROJECT_ROOT, "data", "healthcare.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
DATABASE_URL = "sqlite:///" + DB_PATH.replace("\\", "/")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False, "timeout": 30},
)

@event.listens_for(engine, "connect")
def _set_sqlite_pragma(dbapi_conn, _connection_record):
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA busy_timeout=30000")
    cursor.close()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    password_hash = Column(String)
    role = Column(String) # 'Patient', 'Doctor', 'Nurse', 'Administrator', 'Super Admin'
    full_name = Column(String)

class Patient(Base):
    __tablename__ = "patients"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    name = Column(String, index=True)
    age = Column(Integer)
    gender = Column(String)
    contact = Column(String)
    medical_history = Column(Text, default="")
    allergies = Column(Text, default="")
    
    records = relationship("HealthRecord", back_populates="patient", cascade="all, delete-orphan")
    triage = relationship("TriageRecord", back_populates="patient", cascade="all, delete-orphan")
    appointments = relationship("Appointment", back_populates="patient", cascade="all, delete-orphan")
    prescriptions = relationship("PatientPrescription", back_populates="patient", cascade="all, delete-orphan")

class HealthRecord(Base):
    __tablename__ = "health_records"
    
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"))
    visit_date = Column(DateTime, default=datetime.utcnow)
    symptoms = Column(Text, default="")
    systolic_bp = Column(Integer)
    diastolic_bp = Column(Integer)
    heart_rate = Column(Integer)
    temperature = Column(Float)
    lab_results = Column(Text, default="")
    diagnosis = Column(Text, default="")
    prescription = Column(Text, default="")
    notes = Column(Text, default="")
    lab_report_image = Column(String, nullable=True)
    
    patient = relationship("Patient", back_populates="records")

class TriageRecord(Base):
    __tablename__ = "triage_records"
    
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"))
    priority_level = Column(String) # 'Critical', 'High', 'Medium', 'Low'
    symptom_severity = Column(Text)
    recommended_department = Column(String)
    status = Column(String, default="Pending") # 'Pending', 'Checked'
    created_at = Column(DateTime, default=datetime.utcnow)
    
    patient = relationship("Patient", back_populates="triage")

class Appointment(Base):
    __tablename__ = "appointments"
    
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"))
    doctor_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    appointment_date = Column(DateTime)
    reason = Column(Text)
    status = Column(String, default="Scheduled") # 'Scheduled', 'Completed', 'Cancelled'
    
    patient = relationship("Patient", back_populates="appointments")

class PatientPrescription(Base):
    __tablename__ = "patient_prescriptions"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"))
    prescription_text = Column(Text, default="")
    prescription_image = Column(String, nullable=True)
    drugs_detected = Column(Text, default="[]")
    analysis_json = Column(Text, default="{}")
    created_at = Column(DateTime, default=datetime.utcnow)

    patient = relationship("Patient", back_populates="prescriptions")

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def init_db():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    
    try:
        # Check if users already exist
        if db.query(User).count() == 0:
            print("Seeding initial users...")
            # Create a user for each role
            users_to_seed = [
                User(username="patient1", password_hash=hash_password("password123"), role="Patient", full_name="John Doe"),
                User(username="doctor1", password_hash=hash_password("password123"), role="Doctor", full_name="Dr. Sarah Connor"),
                User(username="nurse1", password_hash=hash_password("password123"), role="Nurse", full_name="Nurse Clara Barton"),
                User(username="admin1", password_hash=hash_password("password123"), role="Administrator", full_name="Admin Alex Smith"),
                User(username="superadmin", password_hash=hash_password("password123"), role="Super Admin", full_name="Super Admin Chief")
            ]
            db.add_all(users_to_seed)
            db.commit()
            
        # One default patient profile linked to patient1
        if db.query(Patient).count() == 0:
            print("Seeding default patient profile...")
            patient_user = db.query(User).filter(User.username == "patient1").first()
            db.add(Patient(
                user_id=patient_user.id if patient_user else None,
                name="John Doe",
                age=45,
                gender="Male",
                contact="555-0100",
                medical_history="None",
                allergies="None",
            ))
            db.commit()

    finally:
        db.close()

if __name__ == "__main__":
    init_db()
