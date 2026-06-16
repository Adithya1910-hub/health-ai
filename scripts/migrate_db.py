import sqlite3
import os

db_path = "data/healthcare.db"

def migrate():
    print(f"Migrating SQLite database at {db_path}...")
    if not os.path.exists(db_path):
        print("Database does not exist yet. Run the database initializer first.")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("PRAGMA table_info(health_records)")
    columns = [row[1] for row in cursor.fetchall()]

    if "lab_report_image" in columns:
        print("Column 'lab_report_image' already exists in 'health_records'. Skipping.")
    else:
        try:
            cursor.execute("ALTER TABLE health_records ADD COLUMN lab_report_image TEXT;")
            conn.commit()
            print("Migration successful: added 'lab_report_image' column to 'health_records'.")
        except Exception as e:
            print(f"Failed to add lab_report_image column: {e}")

    cursor.execute("PRAGMA table_info(patients)")
    patient_columns = [row[1] for row in cursor.fetchall()]

    if "user_id" in patient_columns:
        print("Column 'user_id' already exists in 'patients'. Skipping.")
    else:
        try:
            cursor.execute("ALTER TABLE patients ADD COLUMN user_id INTEGER REFERENCES users(id);")
            conn.commit()
            print("Migration successful: added 'user_id' column to 'patients'.")
        except Exception as e:
            print(f"Failed to add user_id column: {e}")

    # Link existing patient accounts to their user logins by matching name
    cursor.execute("""
        UPDATE patients
        SET user_id = (
            SELECT u.id FROM users u
            WHERE u.role = 'Patient'
              AND LOWER(u.full_name) = LOWER(patients.name)
            LIMIT 1
        )
        WHERE user_id IS NULL
    """)
    conn.commit()
    linked = cursor.rowcount
    print(f"Linked {linked} patient profile(s) to user accounts by name.")

    # Link demo patient1 account to John Doe if still unlinked
    cursor.execute("""
        UPDATE patients
        SET user_id = (SELECT id FROM users WHERE username = 'patient1' LIMIT 1)
        WHERE user_id IS NULL
          AND LOWER(name) = 'john doe'
    """)
    conn.commit()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='patient_prescriptions'")
    if cursor.fetchone():
        print("Table 'patient_prescriptions' already exists. Skipping.")
    else:
        cursor.execute("""
            CREATE TABLE patient_prescriptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER NOT NULL REFERENCES patients(id),
                prescription_text TEXT DEFAULT '',
                prescription_image TEXT,
                drugs_detected TEXT DEFAULT '[]',
                analysis_json TEXT DEFAULT '{}',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        print("Migration successful: created 'patient_prescriptions' table.")

    conn.close()

if __name__ == "__main__":
    migrate()
