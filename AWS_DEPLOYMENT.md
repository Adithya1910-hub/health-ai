# Deploy HealthAI on AWS EC2

These steps are for Amazon Linux EC2. Amazon Linux uses `dnf` or `yum`, not `apt`.

## 1. Open EC2 ports

In the EC2 security group, add inbound rules:

```text
TCP 8000  Source: your IP or 0.0.0.0/0
TCP 8501  Source: your IP or 0.0.0.0/0
```

Use `8501` to open the Streamlit dashboard in the browser.

## 2. Clone the repository

If the GitHub repository is private, GitHub requires a personal access token or SSH key. Password authentication does not work.

Token option:

```bash
git clone https://YOUR_GITHUB_TOKEN@github.com/Adithya1910-hub/health-ai.git
cd health-ai
```

Public repository option:

```bash
git clone https://github.com/Adithya1910-hub/health-ai.git
cd health-ai
```

## 3. Install and run

```bash
sudo dnf update -y
sudo dnf install -y git python3 python3-pip

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-ec2.txt

export HEALTHAI_HOST=0.0.0.0
export HEALTHAI_BACKEND_PORT=8000
export HEALTHAI_FRONTEND_PORT=8501
export HEALTHAI_API_BASE_URL=http://127.0.0.1:8000
python run.py
```

If `dnf` is not available, replace the install commands with:

```bash
sudo yum update -y
sudo yum install -y git python3 python3-pip
```

## 4. Open the app

Open:

```text
http://YOUR_EC2_PUBLIC_IP:8501
```

## One-command setup helper

After cloning the repo, you can also run:

```bash
bash scripts/deploy_amazon_linux.sh
```

Then start the app with the command printed by the script.
