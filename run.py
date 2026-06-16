import os
import sys
import time
import subprocess
import signal

# Ensure we use the virtual environment's Python executable
PYTHON_EXE = os.path.join(".venv", "Scripts", "python.exe")
UVICORN_EXE = os.path.join(".venv", "Scripts", "uvicorn.exe")
STREAMLIT_EXE = os.path.join(".venv", "Scripts", "streamlit.exe")

if not os.path.exists(PYTHON_EXE):
    # Fallback to current system python if venv isn't set up
    PYTHON_EXE = sys.executable
    UVICORN_EXE = "uvicorn"
    STREAMLIT_EXE = "streamlit"

def run_step(description, command_list):
    print(f"\n>>> {description}...")
    try:
        res = subprocess.run(command_list, check=True, capture_output=True, text=True)
        print(res.stdout)
        print(f"[OK] {description} completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"[FAIL] Error during {description.lower()}:")
        print(e.stderr)
        sys.exit(1)

def main():
    print("==================================================================")
    print("                  HEALTHAI SYSTEM STARTUP SCRIPT                 ")
    print("==================================================================")
    
    # 1. Initialize SQLite database & seed
    run_step("Initializing database and seeding mock data", [PYTHON_EXE, "-m", "backend.database"])
    
    # 2. Generate guideline PDFs for RAG
    run_step("Generating clinical guideline reference documents", [PYTHON_EXE, "-m", "scripts.generate_guidelines"])
    
    # 3. Train disease risk prediction model
    run_step("Training Scikit-learn disease risk classifier (>80% target accuracy)", [PYTHON_EXE, "-m", "scripts.train_ml"])
    
    # 4. Start FastAPI backend & Streamlit frontend concurrently
    print("\n>>> Launching HealthAI services...")
    
    processes = []
    
    # Start FastAPI backend
    os.makedirs("data", exist_ok=True)
    backend_log = open("data/backend.log", "w", encoding="utf-8")
    backend_cmd = [UVICORN_EXE, "backend.main:app", "--host", "127.0.0.1", "--port", "8000"]
    print(f"Starting Backend API: {' '.join(backend_cmd)}")
    backend_proc = subprocess.Popen(backend_cmd, stdout=backend_log, stderr=subprocess.STDOUT, text=True)
    processes.append(backend_proc)
    
    # Wait for backend to spin up
    time.sleep(2.0)
    
    # Start Streamlit frontend
    frontend_log = open("data/frontend.log", "w", encoding="utf-8")
    frontend_cmd = [STREAMLIT_EXE, "run", "frontend/app.py", "--server.port", "8501", "--server.headless", "true"]
    print(f"Starting Frontend Dashboard: {' '.join(frontend_cmd)}")
    frontend_proc = subprocess.Popen(frontend_cmd, stdout=frontend_log, stderr=subprocess.STDOUT, text=True)
    processes.append(frontend_proc)
    
    print("\n[INFO] System Running!")
    print("   - Backend API URL: http://127.0.0.1:8000")
    print("   - Frontend Dashboard URL: http://127.0.0.1:8501")
    print("Press Ctrl+C to stop all services.\n")
    
    # Helper function to stream output in the console
    def stream_output(process, prefix):
        pass

    try:
        # Keep monitoring processes
        while True:
            # Check if any process terminated unexpectedly
            for p in processes:
                if p.poll() is not None:
                    print(f"\n[FAIL] One of the services has stopped unexpectedly (exit code {p.returncode}).")
                    raise KeyboardInterrupt
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n>>> Shutting down all HealthAI services...")
        for p in processes:
            try:
                # Send terminate signal
                p.terminate()
                p.wait(timeout=3)
            except Exception:
                p.kill()
        print("[OK] All processes stopped cleanly. Goodbye!")

if __name__ == "__main__":
    main()
