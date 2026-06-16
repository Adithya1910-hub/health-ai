import os
import sys
import time
import subprocess
import shutil

def find_executable(windows_path, unix_path, fallback):
    for candidate in (windows_path, unix_path, shutil.which(fallback), fallback):
        if candidate and (os.path.exists(candidate) or shutil.which(candidate)):
            return candidate
    return fallback


# Prefer the project virtual environment, but work on both Windows and Linux.
PYTHON_EXE = find_executable(
    os.path.join(".venv", "Scripts", "python.exe"),
    os.path.join(".venv", "bin", "python"),
    "python",
)
if not os.path.exists(PYTHON_EXE) and not shutil.which(PYTHON_EXE):
    PYTHON_EXE = sys.executable

UVICORN_EXE = find_executable(
    os.path.join(".venv", "Scripts", "uvicorn.exe"),
    os.path.join(".venv", "bin", "uvicorn"),
    "uvicorn",
)
STREAMLIT_EXE = find_executable(
    os.path.join(".venv", "Scripts", "streamlit.exe"),
    os.path.join(".venv", "bin", "streamlit"),
    "streamlit",
)

BACKEND_HOST = os.getenv("HEALTHAI_BACKEND_HOST", os.getenv("HEALTHAI_HOST", "127.0.0.1"))
FRONTEND_HOST = os.getenv("HEALTHAI_FRONTEND_HOST", os.getenv("HEALTHAI_HOST", "127.0.0.1"))
BACKEND_PORT = os.getenv("HEALTHAI_BACKEND_PORT", "8000")
FRONTEND_PORT = os.getenv("HEALTHAI_FRONTEND_PORT", "8501")

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
    backend_cmd = [UVICORN_EXE, "backend.main:app", "--host", BACKEND_HOST, "--port", BACKEND_PORT]
    print(f"Starting Backend API: {' '.join(backend_cmd)}")
    backend_proc = subprocess.Popen(backend_cmd, stdout=backend_log, stderr=subprocess.STDOUT, text=True)
    processes.append(backend_proc)
    
    # Wait for backend to spin up
    time.sleep(2.0)
    
    # Start Streamlit frontend
    frontend_log = open("data/frontend.log", "w", encoding="utf-8")
    frontend_cmd = [
        STREAMLIT_EXE,
        "run",
        "frontend/app.py",
        "--server.port",
        FRONTEND_PORT,
        "--server.address",
        FRONTEND_HOST,
        "--server.headless",
        "true",
    ]
    print(f"Starting Frontend Dashboard: {' '.join(frontend_cmd)}")
    frontend_proc = subprocess.Popen(frontend_cmd, stdout=frontend_log, stderr=subprocess.STDOUT, text=True)
    processes.append(frontend_proc)
    
    print("\n[INFO] System Running!")
    print(f"   - Backend API URL: http://{BACKEND_HOST}:{BACKEND_PORT}")
    print(f"   - Frontend Dashboard URL: http://{FRONTEND_HOST}:{FRONTEND_PORT}")
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
