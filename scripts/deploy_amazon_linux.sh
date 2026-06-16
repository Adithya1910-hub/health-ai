#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-$HOME/health-ai}"
REPO_URL="${REPO_URL:-https://github.com/Adithya1910-hub/health-ai.git}"

if command -v dnf >/dev/null 2>&1; then
  sudo dnf update -y
  sudo dnf install -y git python3 python3-pip
elif command -v yum >/dev/null 2>&1; then
  sudo yum update -y
  sudo yum install -y git python3 python3-pip
else
  echo "Could not find dnf or yum. This script is intended for Amazon Linux."
  exit 1
fi

if [ ! -d "$APP_DIR/.git" ]; then
  git clone "$REPO_URL" "$APP_DIR"
fi

cd "$APP_DIR"
git pull --ff-only

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-requirements-ec2.txt}"
python -m pip install -r "$REQUIREMENTS_FILE"

cat > .env.ec2 <<'ENV'
HEALTHAI_HOST=0.0.0.0
HEALTHAI_BACKEND_PORT=8000
HEALTHAI_FRONTEND_PORT=8501
HEALTHAI_API_BASE_URL=http://127.0.0.1:8000
ENV

echo "Setup complete."
echo "Start the app with:"
echo "  cd $APP_DIR"
echo "  source .venv/bin/activate"
echo "  set -a; source .env.ec2; set +a; python run.py"
