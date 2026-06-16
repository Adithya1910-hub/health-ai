import sqlite3
import os

db_path = os.path.join(os.path.dirname(__file__), "..", "data", "healthcare.db")
conn = sqlite3.connect(db_path)
c = conn.cursor()

print("=== PATIENTS ===")
c.execute("SELECT id, user_id, name FROM patients")
for row in c.fetchall():
    print(row)

print("\n=== USERS (Patient role) ===")
c.execute("SELECT id, username, full_name, role FROM users WHERE role='Patient'")
for row in c.fetchall():
    print(row)

print("\n=== HEALTH RECORDS (all) ===")
c.execute("SELECT id, patient_id, diagnosis, lab_report_image FROM health_records ORDER BY id DESC")
for row in c.fetchall():
    print(row)

print("\n=== UPLOADED FILES ===")
upload_dir = os.path.join(os.path.dirname(__file__), "..", "data", "uploaded_reports")
if os.path.exists(upload_dir):
    for f in os.listdir(upload_dir):
        print(f)
else:
    print("(directory does not exist)")

conn.close()
