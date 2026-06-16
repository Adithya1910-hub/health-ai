import os
import sys
import time
import json

# Ensure root directory is in sys.path so we can import backend modules
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

# Import local backend modules for direct fallback if API server is offline
from backend.database import SessionLocal, User, Patient, HealthRecord, TriageRecord, Appointment, PatientPrescription, hash_password
from backend.triage import classify_triage
from backend.ml_model import predict_disease_risk
from backend.rag_engine import generate_treatment_recommendations
from backend.medication_service import analyze_prescription, extract_text_from_prescription_image

API_BASE_URL = os.getenv("HEALTHAI_API_BASE_URL", "http://127.0.0.1:8000")

def log_debug(msg):
    try:
        os.makedirs("data", exist_ok=True)
        with open("data/debug.log", "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat()} - {msg}\n")
    except Exception:
        pass

log_debug("app.py - starting...")
log_debug("app.py - setting page config...")
st.set_page_config(
    page_title="HealthAI — Clinical Intelligence Platform",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Premium design system ──────────────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Outfit:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
    :root {
        /* Deep Obsidian 2026 SaaS Palette */
        --bg-main: #060913;
        --bg-surface: rgba(13, 20, 38, 0.5);
        --bg-surface-solid: #0d1222;
        --border-color: rgba(255, 255, 255, 0.08);
        --border-color-glow: rgba(6, 182, 212, 0.15);
        
        /* Typography Colors */
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        --text-dark: #0f172a;
        
        /* Accents & States */
        --accent-teal: #0df2c9;
        --accent-cyan: #06b6d4;
        --accent-indigo: #6366f1;
        --accent-violet: #8b5cf6;
        
        /* Alerts */
        --color-critical: #ef4444;
        --color-high: #f97316;
        --color-medium: #fbbf24;
        --color-low: #10b981;
        
        --color-critical-bg: rgba(239, 68, 68, 0.1);
        --color-high-bg: rgba(249, 115, 22, 0.1);
        --color-medium-bg: rgba(251, 191, 36, 0.1);
        --color-low-bg: rgba(16, 185, 129, 0.1);
        
        /* Shadow & Radius */
        --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.4);
        --shadow-md: 0 8px 24px rgba(0, 0, 0, 0.5);
        --shadow-lg: 0 16px 48px rgba(0, 0, 0, 0.6);
        --shadow-glow: 0 0 15px rgba(6, 182, 212, 0.15);
        --radius-lg: 16px;
        --radius-md: 12px;
        --radius-sm: 8px;
    }

    @keyframes ha-float {
        0%, 100% { transform: translate(0, 0) scale(1); }
        50% { transform: translate(15px, -10px) scale(1.03); }
    }
    @keyframes ha-shimmer {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    @keyframes ha-pulse-ring {
        0% { box-shadow: 0 0 0 0 rgba(6, 182, 212, 0.3); }
        70% { box-shadow: 0 0 0 12px rgba(6, 182, 212, 0); }
        100% { box-shadow: 0 0 0 0 rgba(6, 182, 212, 0); }
    }

    #MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }

    .stApp {
        font-family: 'Plus Jakarta Sans', 'Segoe UI', system-ui, sans-serif;
        color: var(--text-primary) !important;
        background: var(--bg-main) !important;
    }
    .stApp::before {
        content: '';
        position: fixed; inset: 0; z-index: 0; pointer-events: none;
        background:
            radial-gradient(ellipse 55% 45% at 8% 12%, rgba(6, 182, 212, 0.1), transparent),
            radial-gradient(ellipse 50% 40% at 92% 88%, rgba(99, 102, 241, 0.1), transparent),
            radial-gradient(ellipse 40% 35% at 55% 45%, rgba(139, 92, 246, 0.05), transparent),
            linear-gradient(165deg, #04060c 0%, #090e1a 50%, #050810 100%);
    }
    [data-testid="stAppViewContainer"], .main { position: relative; z-index: 1; }
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 4rem !important;
        max-width: 1200px !important;
    }

    /* ── Readable Labels & Typography ── */
    [data-testid="stAppViewContainer"] [data-testid="stWidgetLabel"] p,
    [data-testid="stAppViewContainer"] [data-testid="stWidgetLabel"] label,
    [data-testid="stAppViewContainer"] label[data-testid="stWidgetLabel"],
    [data-testid="stAppViewContainer"] .stTextInput label,
    [data-testid="stAppViewContainer"] .stTextArea label,
    [data-testid="stAppViewContainer"] .stNumberInput label,
    [data-testid="stAppViewContainer"] .stSelectbox label,
    [data-testid="stAppViewContainer"] .stDateInput label,
    [data-testid="stAppViewContainer"] .stTimeInput label,
    [data-testid="stAppViewContainer"] .stFileUploader label {
        color: var(--text-secondary) !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.02em !important;
        margin-bottom: 6px !important;
    }
    [data-testid="stAppViewContainer"] .stMarkdown p,
    [data-testid="stAppViewContainer"] .stMarkdown li,
    [data-testid="stAppViewContainer"] .stWrite {
        color: var(--text-secondary) !important;
        font-size: 0.95rem !important;
        line-height: 1.6 !important;
    }
    [data-testid="stAppViewContainer"] h1,
    [data-testid="stAppViewContainer"] h2,
    [data-testid="stAppViewContainer"] h3,
    [data-testid="stAppViewContainer"] h4 {
        color: #ffffff !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }
    [data-testid="stAppViewContainer"] input,
    [data-testid="stAppViewContainer"] textarea {
        color: var(--text-primary) !important;
        -webkit-text-fill-color: var(--text-primary) !important;
    }
    [data-testid="stAppViewContainer"] input::placeholder,
    [data-testid="stAppViewContainer"] textarea::placeholder {
        color: var(--text-muted) !important;
        opacity: 0.8 !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #090d16 0%, #04060b 100%) !important;
        border-right: 1px solid var(--border-color) !important;
        box-shadow: 4px 0 32px rgba(0, 0, 0, 0.6) !important;
    }
    [data-testid="stSidebar"] .stMarkdown p { color: var(--text-secondary) !important; }
    [data-testid="stSidebar"] hr { border-color: var(--border-color) !important; }
    [data-testid="stSidebar"] .stButton > button {
        background: rgba(255, 255, 255, 0.05) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: var(--radius-md) !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 600 !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.2s ease !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(6, 182, 212, 0.15) !important;
        border-color: var(--accent-cyan) !important;
        color: #ffffff !important;
        box-shadow: 0 0 12px rgba(6, 182, 212, 0.2) !important;
    }

    /* ── Main Buttons ── */
    [data-testid="stAppViewContainer"] .stButton > button {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 700 !important;
        border-radius: var(--radius-md) !important;
        padding: 0.6rem 1.4rem !important;
        transition: all 0.22s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    [data-testid="stAppViewContainer"] .stButton > button[kind="primary"],
    [data-testid="stAppViewContainer"] .stButton > button[data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, var(--accent-cyan) 0%, #06b6d4 40%, var(--accent-indigo) 100%) !important;
        color: #ffffff !important;
        border: none !important;
        box-shadow: 0 4px 14px rgba(6, 182, 212, 0.3) !important;
    }
    [data-testid="stAppViewContainer"] .stButton > button[kind="primary"]:hover,
    [data-testid="stAppViewContainer"] .stButton > button[data-testid="baseButton-primary"]:hover {
        transform: translateY(-1.5px);
        box-shadow: 0 6px 20px rgba(6, 182, 212, 0.45) !important;
        opacity: 0.95 !important;
    }
    [data-testid="stAppViewContainer"] .stButton > button[kind="primary"]:active,
    [data-testid="stAppViewContainer"] .stButton > button[data-testid="baseButton-primary"]:active {
        transform: translateY(0.5px);
    }
    
    [data-testid="stAppViewContainer"] .stButton > button:not([kind="primary"]):not([data-testid="baseButton-primary"]) {
        background: rgba(255, 255, 255, 0.04) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
        box-shadow: var(--shadow-sm) !important;
    }
    [data-testid="stAppViewContainer"] .stButton > button:not([kind="primary"]):not([data-testid="baseButton-primary"]):hover {
        background: rgba(255, 255, 255, 0.08) !important;
        border-color: rgba(255, 255, 255, 0.2) !important;
        color: #ffffff !important;
    }

    /* ── Pill Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(13, 20, 38, 0.4);
        backdrop-filter: blur(12px);
        border-radius: var(--radius-lg);
        padding: 6px;
        border: 1px solid var(--border-color);
        box-shadow: var(--shadow-md);
        margin-bottom: 12px;
        overflow-x: auto !important;
        scrollbar-width: thin;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        border-radius: var(--radius-md) !important;
        padding: 8px 16px !important;
        color: var(--text-secondary) !important;
        background: transparent !important;
        border: none !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        flex: 0 0 auto !important;
        white-space: nowrap !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text-primary) !important;
        background: rgba(255, 255, 255, 0.04) !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(6, 182, 212, 0.15) 0%, rgba(99, 102, 241, 0.15) 100%) !important;
        color: var(--accent-teal) !important;
        border: 1px solid rgba(6, 182, 212, 0.3) !important;
        box-shadow: var(--shadow-sm) !important;
    }
    .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }

    /* Frosted glass content cards for tabs */
    [data-testid="stTabContent"] {
        background: var(--bg-surface) !important;
        backdrop-filter: blur(24px) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: var(--radius-lg) !important;
        padding: 2rem !important;
        margin-top: 0.75rem !important;
        box-shadow: var(--shadow-lg) !important;
    }

    /* ── Inputs & Forms ── */
    [data-testid="stAppViewContainer"] .stTextInput input,
    [data-testid="stAppViewContainer"] .stTextArea textarea,
    [data-testid="stAppViewContainer"] .stNumberInput input,
    [data-testid="stAppViewContainer"] .stSelectbox [data-baseweb="select"] > div,
    [data-testid="stAppViewContainer"] .stDateInput input,
    [data-testid="stAppViewContainer"] .stTimeInput input {
        border-radius: var(--radius-md) !important;
        border: 1px solid var(--border-color) !important;
        background: rgba(15, 23, 42, 0.4) !important;
        color: var(--text-primary) !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 0.9rem !important;
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.2) !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    [data-testid="stAppViewContainer"] .stTextInput input:focus,
    [data-testid="stAppViewContainer"] .stTextArea textarea:focus,
    [data-testid="stAppViewContainer"] .stNumberInput input:focus,
    [data-testid="stAppViewContainer"] .stSelectbox [data-baseweb="select"] > div:focus-within {
        border-color: var(--accent-cyan) !important;
        box-shadow: var(--shadow-glow), inset 0 2px 4px rgba(0, 0, 0, 0.2) !important;
        outline: none !important;
    }
    [data-baseweb="popover"], [data-baseweb="menu"], [role="listbox"] {
        background-color: var(--bg-surface-solid) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: var(--radius-md) !important;
        box-shadow: var(--shadow-lg) !important;
    }
    [data-baseweb="option"] {
        color: var(--text-primary) !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 0.9rem !important;
        transition: background-color 0.15s ease !important;
    }
    [data-baseweb="option"]:hover, [data-baseweb="option"][aria-selected="true"] {
        background-color: rgba(255, 255, 255, 0.08) !important;
    }
    [data-testid="stAppViewContainer"] [data-testid="stFileUploaderDropzone"] {
        background: rgba(15, 23, 42, 0.3) !important;
        border-radius: var(--radius-md) !important;
        border: 1.5px dashed var(--border-color) !important;
        padding: 1.5rem !important;
        transition: all 0.2s ease !important;
    }
    [data-testid="stAppViewContainer"] [data-testid="stFileUploaderDropzone"]:hover {
        border-color: var(--accent-cyan) !important;
        background: rgba(6, 182, 212, 0.05) !important;
    }
    [data-testid="stAppViewContainer"] [data-testid="stFileUploaderDropzone"] * {
        color: var(--text-secondary) !important;
    }
    [data-testid="stAppViewContainer"] [data-testid="stNumberInput"] button {
        background: rgba(255, 255, 255, 0.04) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
    }
    [data-testid="stAppViewContainer"] [data-testid="stNumberInput"] button:hover {
        background: rgba(255, 255, 255, 0.08) !important;
        color: #ffffff !important;
    }

    /* ── Details & Expanders ── */
    details[data-testid="stExpander"] {
        border: 1px solid var(--border-color) !important;
        border-radius: var(--radius-md) !important;
        background: rgba(13, 20, 38, 0.2) !important;
        box-shadow: var(--shadow-sm) !important;
        margin-bottom: 0.75rem !important;
        overflow: hidden !important;
    }
    details[data-testid="stExpander"] summary {
        background: rgba(13, 20, 38, 0.4) !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.04) !important;
        min-height: 2.8rem !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.2s ease !important;
    }
    details[data-testid="stExpander"]:not([open]) summary {
        border-bottom: none !important;
        border-radius: var(--radius-md) !important;
    }
    details[data-testid="stExpander"] summary:hover {
        background: rgba(13, 20, 38, 0.6) !important;
        box-shadow: inset 3px 0 0 var(--accent-cyan) !important;
    }
    details[data-testid="stExpander"] summary,
    details[data-testid="stExpander"] summary *,
    details[data-testid="stExpander"] summary p,
    details[data-testid="stExpander"] summary svg {
        color: #ffffff !important;
        fill: #ffffff !important;
        opacity: 1 !important;
        font-weight: 700 !important;
    }
    details[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
        background: transparent !important;
        padding: 1.25rem !important;
        color: var(--text-secondary) !important;
    }

    [data-testid="stDataFrame"] {
        border-radius: var(--radius-md) !important;
        overflow: hidden !important;
        border: 1px solid var(--border-color) !important;
        box-shadow: var(--shadow-sm) !important;
        background: rgba(15, 23, 42, 0.2) !important;
    }
    [data-testid="stAlert"] { border-radius: var(--radius-md) !important; }

    /* ── Login Page Overhaul ── */
    .ha-login-hero {
        text-align: center;
        padding: 3.5rem 1rem 2rem;
    }
    .ha-login-hero .ha-logo-ring {
        width: 80px; height: 80px;
        margin: 0 auto 1.5rem;
        border-radius: 24px;
        background: linear-gradient(135deg, var(--accent-cyan), var(--accent-indigo));
        display: flex; align-items: center; justify-content: center;
        font-size: 2.2rem;
        box-shadow: 0 16px 48px rgba(6, 182, 212, 0.3);
        position: relative;
    }
    .ha-login-hero .ha-logo-ring::after {
        content: '';
        position: absolute; inset: -4px;
        border-radius: 28px;
        border: 2px solid var(--accent-cyan);
        opacity: 0.4;
        animation: pulse 2.5s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.4; }
        50% { transform: scale(1.15); opacity: 0; }
    }
    
    .ha-login-hero h1 {
        font-family: 'Outfit', sans-serif !important;
        font-size: 3.2rem !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #ffffff 30%, var(--text-secondary) 70%, var(--accent-cyan));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0 0 0.6rem !important;
        letter-spacing: -0.03em;
    }
    .ha-login-hero p {
        color: var(--text-secondary);
        font-size: 1.05rem;
        margin: 0 auto;
        max-width: 480px;
        line-height: 1.6;
    }
    .ha-feature-row {
        display: flex; flex-wrap: wrap; gap: 10px;
        justify-content: center; margin: 1.75rem 0 0.5rem;
    }
    .ha-pill {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid var(--border-color);
        border-radius: 99px;
        padding: 6px 14px;
        font-size: 0.75rem;
        font-weight: 600;
        color: var(--text-primary);
        box-shadow: var(--shadow-sm);
        transition: all 0.2s ease;
    }
    .ha-pill:hover {
        background: rgba(255, 255, 255, 0.08);
        border-color: rgba(6, 182, 212, 0.3);
    }
    .ha-form-head { margin-bottom: 1.5rem; padding-bottom: 1rem; border-bottom: 1px solid var(--border-color); }
    .ha-form-head h3 {
        font-family: 'Outfit', sans-serif !important;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        margin: 0 0 0.35rem !important;
    }
    .ha-form-head p { color: var(--text-secondary) !important; font-size: 0.92rem; margin: 0 !important; }

    /* ── Page Hero ── */
    .ha-page-hero {
        background: linear-gradient(135deg, #090d16 0%, #111827 100%);
        border: 1px solid var(--border-color);
        border-radius: var(--radius-lg);
        padding: 2rem;
        margin-bottom: 1.5rem;
        color: var(--text-primary);
        position: relative;
        overflow: hidden;
        box-shadow: var(--shadow-md);
    }
    .ha-page-hero::before {
        content: '';
        position: absolute; top: -50%; right: -10%;
        width: 300px; height: 300px;
        background: radial-gradient(circle, rgba(6, 182, 212, 0.15) 0%, transparent 70%);
        pointer-events: none;
    }
    .ha-page-hero h1 {
        font-family: 'Outfit', sans-serif !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        margin: 0 0 0.5rem !important;
    }
    .ha-page-hero .ha-hero-sub {
        color: var(--text-secondary);
        font-size: 0.95rem;
        line-height: 1.5;
        margin: 0;
    }
    .ha-page-hero .ha-hero-badge {
        display: inline-block;
        margin-top: 0.75rem;
        background: rgba(6, 182, 212, 0.1);
        border: 1px solid rgba(6, 182, 212, 0.25);
        border-radius: 99px;
        padding: 4px 12px;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: var(--accent-teal);
    }

    /* ── Section Headers ── */
    .ha-section-head { margin-bottom: 1.25rem; }
    .ha-section-head h3 {
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        margin: 0 0 0.3rem !important;
    }
    .ha-section-sub { color: var(--text-secondary) !important; font-size: 0.9rem; margin: 0 0 1rem !important; line-height: 1.55; }
    .ha-vitals-label {
        font-size: 0.78rem !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        color: var(--accent-teal) !important;
        margin: 0 0 0.75rem !important;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(13, 242, 201, 0.2);
    }

    /* ── Custom Premium Cards ── */
    .card {
        background: rgba(17, 25, 40, 0.45) !important;
        backdrop-filter: blur(12px);
        border: 1px solid var(--border-color);
        border-radius: var(--radius-lg);
        padding: 1.5rem;
        box-shadow: var(--shadow-md);
        margin-bottom: 1rem;
        color: var(--text-primary);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .card:hover {
        border-color: rgba(255, 255, 255, 0.15);
    }
    .card h3, .card h4 {
        margin-top: 0 !important;
        font-family: 'Outfit', sans-serif !important;
        color: #ffffff !important;
        font-weight: 600;
        margin-bottom: 1rem !important;
    }

    .ha-metric-card {
        background: rgba(17, 25, 40, 0.5);
        backdrop-filter: blur(16px);
        border: 1px solid var(--border-color);
        border-radius: var(--radius-lg);
        padding: 1.5rem;
        text-align: center;
        box-shadow: var(--shadow-md);
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    .ha-metric-card::before {
        content: '';
        position: absolute; inset: 0;
        background: linear-gradient(135deg, rgba(6, 182, 212, 0.05) 0%, transparent 100%);
        opacity: 0;
        transition: opacity 0.25s ease;
    }
    .ha-metric-card:hover {
        transform: translateY(-3px);
        border-color: rgba(6, 182, 212, 0.3);
        box-shadow: var(--shadow-lg), var(--shadow-glow);
    }
    .ha-metric-card:hover::before {
        opacity: 1;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        font-family: 'Outfit', sans-serif;
        background: linear-gradient(135deg, #ffffff, var(--accent-teal));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .metric-label {
        font-size: 0.75rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 700;
        margin-top: 0.5rem;
    }

    /* ── Sidebar Branding & Status ── */
    .ha-sidebar-brand {
        text-align: center;
        padding: 1.5rem 0 1rem;
    }
    .ha-sidebar-brand .ha-sb-icon {
        width: 48px; height: 48px;
        margin: 0 auto 0.75rem;
        border-radius: 14px;
        background: linear-gradient(135deg, var(--accent-cyan), var(--accent-indigo));
        display: flex; align-items: center; justify-content: center;
        font-size: 1.5rem;
        box-shadow: 0 6px 20px rgba(6, 182, 212, 0.3);
    }
    .ha-sidebar-brand h3 {
        font-family: 'Outfit', sans-serif !important;
        font-size: 1.35rem !important;
        color: #ffffff !important;
        margin: 0 !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }
    .ha-sidebar-brand p {
        font-size: 0.72rem;
        color: var(--text-muted) !important;
        margin: 0.15rem 0 0 !important;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .ha-user-chip {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid var(--border-color);
        border-radius: var(--radius-md);
        padding: 12px 14px;
        margin: 0.75rem 0;
        display: flex;
        flex-direction: column;
        gap: 2px;
    }
    .ha-user-chip .ha-name {
        font-weight: 600;
        color: #ffffff !important;
        font-size: 0.9rem;
    }
    .ha-user-chip .ha-role {
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--accent-teal) !important;
        font-weight: 700;
    }
    .ha-status-online {
        background: rgba(16, 185, 129, 0.08) !important;
        border: 1px solid rgba(16, 185, 129, 0.2) !important;
        border-radius: var(--radius-sm) !important;
        padding: 6px 12px !important;
        font-size: 0.75rem !important;
        color: var(--color-low) !important;
        font-weight: 600 !important;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .ha-status-offline {
        background: rgba(249, 115, 22, 0.08) !important;
        border: 1px solid rgba(249, 115, 22, 0.2) !important;
        border-radius: var(--radius-sm) !important;
        padding: 6px 12px !important;
        font-size: 0.75rem !important;
        color: var(--color-high) !important;
        font-weight: 600 !important;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }

    /* ── Clinical Alerts & Class-Based styling ── */
    .ha-alert-item-critical {
        background: rgba(239, 68, 68, 0.08) !important;
        border-left: 4px solid var(--color-critical) !important;
        border-top: 1px solid rgba(239, 68, 68, 0.12) !important;
        border-right: 1px solid rgba(239, 68, 68, 0.12) !important;
        border-bottom: 1px solid rgba(239, 68, 68, 0.12) !important;
        padding: 1rem 1.25rem !important;
        margin-bottom: 0.75rem !important;
        border-radius: var(--radius-md) !important;
        color: var(--text-primary) !important;
    }
    .ha-alert-item-warning {
        background: rgba(249, 115, 22, 0.08) !important;
        border-left: 4px solid var(--color-high) !important;
        border-top: 1px solid rgba(249, 115, 22, 0.12) !important;
        border-right: 1px solid rgba(249, 115, 22, 0.12) !important;
        border-bottom: 1px solid rgba(249, 115, 22, 0.12) !important;
        padding: 1rem 1.25rem !important;
        margin-bottom: 0.75rem !important;
        border-radius: var(--radius-md) !important;
        color: var(--text-primary) !important;
    }
    .ha-alert-item-info {
        background: rgba(6, 182, 212, 0.08) !important;
        border-left: 4px solid var(--accent-cyan) !important;
        border-top: 1px solid rgba(6, 182, 212, 0.12) !important;
        border-right: 1px solid rgba(6, 182, 212, 0.12) !important;
        border-bottom: 1px solid rgba(6, 182, 212, 0.12) !important;
        padding: 1rem 1.25rem !important;
        margin-bottom: 0.75rem !important;
        border-radius: var(--radius-md) !important;
        color: var(--text-primary) !important;
    }
    .ha-alert-item-success {
        background: rgba(16, 185, 129, 0.08) !important;
        border-left: 4px solid var(--color-low) !important;
        border-top: 1px solid rgba(16, 185, 129, 0.12) !important;
        border-right: 1px solid rgba(16, 185, 129, 0.12) !important;
        border-bottom: 1px solid rgba(16, 185, 129, 0.12) !important;
        padding: 1rem 1.25rem !important;
        margin-bottom: 0.75rem !important;
        border-radius: var(--radius-md) !important;
        color: var(--text-primary) !important;
    }
    .ha-alert-item-critical *, .ha-alert-item-warning *, .ha-alert-item-info *, .ha-alert-item-success * {
        color: var(--text-primary) !important;
    }

    .badge {
        padding: 4px 10px;
        border-radius: 99px;
        font-size: 0.7rem;
        font-weight: 700;
        display: inline-flex;
        align-items: center;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        border: 1px solid transparent;
    }
    .badge-critical {
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border-color: rgba(239, 68, 68, 0.3);
    }
    .badge-high {
        background: rgba(249, 115, 22, 0.15);
        color: #fb923c;
        border-color: rgba(249, 115, 22, 0.3);
    }
    .badge-medium {
        background: rgba(251, 191, 36, 0.15);
        color: #fcd34d;
        border-color: rgba(251, 191, 36, 0.3);
    }
    .badge-low {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border-color: rgba(16, 185, 129, 0.3);
    }

    .ha-section-title {
        font-size: 1rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        margin: 1.25rem 0 0.75rem !important;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid rgba(6, 182, 212, 0.25);
        display: inline-block;
    }
    
    /* Triage Board cards */
    .ha-triage-card {
        border-left: 4px solid;
        padding: 1.25rem 1.5rem;
        margin-bottom: 0.75rem;
        background: rgba(17, 25, 40, 0.4) !important;
        border-radius: 0 var(--radius-md) var(--radius-md) 0;
        box-shadow: var(--shadow-sm);
        border-top: 1px solid var(--border-color) !important;
        border-right: 1px solid var(--border-color) !important;
        border-bottom: 1px solid var(--border-color) !important;
        transition: transform 0.2s ease !important;
    }
    .ha-triage-card:hover {
        transform: translateX(4px);
    }
    .ha-triage-card-title {
        margin: 0 0 8px 0 !important;
        color: #ffffff !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .ha-triage-card-text {
        margin: 4px 0 !important;
        color: var(--text-secondary) !important;
        font-size: 0.9rem !important;
    }
    .ha-triage-card-text strong {
        color: var(--text-primary) !important;
    }

    /* Export buttons */
    .ha-export-btn {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        width: 100%;
        padding: 12px 16px;
        border-radius: var(--radius-md);
        font-weight: 700;
        font-size: 0.88rem;
        cursor: text-decoration: none !important;
        margin-bottom: 12px;
        font-family: 'Plus Jakarta Sans', sans-serif;
        transition: all 0.2s ease;
        border: 1px solid transparent;
    }
    .ha-export-btn:hover {
        transform: translateY(-2px);
    }
    .ha-export-pdf {
        background: rgba(255, 255, 255, 0.05) !important;
        color: #ffffff !important;
        border-color: var(--border-color) !important;
    }
    .ha-export-pdf:hover {
        background: rgba(255, 255, 255, 0.1) !important;
        border-color: rgba(255, 255, 255, 0.25) !important;
    }
    .ha-export-xls {
        background: linear-gradient(135deg, var(--accent-cyan) 0%, var(--accent-indigo) 100%) !important;
        color: #ffffff !important;
        box-shadow: 0 4px 12px rgba(6, 182, 212, 0.25) !important;
    }
    .ha-export-xls:hover {
        box-shadow: 0 6px 16px rgba(6, 182, 212, 0.4) !important;
    }

    /* ── Patient Profile details Grid ── */
    .ha-patient-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 12px;
        margin-top: 12px;
    }
    .ha-patient-item {
        display: flex;
        flex-direction: column;
        gap: 4px;
        padding: 8px 12px;
        background: rgba(255, 255, 255, 0.02);
        border-radius: var(--radius-sm);
        border: 1px solid rgba(255, 255, 255, 0.04);
    }
    .ha-patient-label {
        font-size: 0.72rem;
        color: var(--text-muted);
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.05em;
    }
    .ha-patient-value {
        font-size: 0.9rem;
        color: var(--text-primary);
        font-weight: 500;
    }
    .ha-allergy-value {
        color: #ef4444 !important;
        font-weight: 600 !important;
    }

    /* ── Appointment cards inside Admin Workspace ── */
    .ha-appt-patient-card {
        background: rgba(6, 180, 216, 0.05) !important;
        border-left: 4px solid var(--accent-cyan) !important;
        border-top: 1px solid rgba(6, 180, 216, 0.1) !important;
        border-right: 1px solid rgba(6, 180, 216, 0.1) !important;
        border-bottom: 1px solid rgba(6, 180, 216, 0.1) !important;
        padding: 1.25rem !important;
        border-radius: var(--radius-lg) !important;
        color: var(--text-primary) !important;
    }
    .ha-appt-patient-card h4 {
        color: var(--accent-teal) !important;
        margin-top: 0 !important;
        font-family: 'Outfit', sans-serif !important;
    }
    .ha-appt-patient-card p {
        margin: 6px 0 !important;
        color: var(--text-secondary) !important;
        font-size: 0.9rem !important;
    }
    .ha-appt-patient-card p strong {
        color: var(--text-primary) !important;
    }
    
    .ha-appt-doctor-card {
        background: rgba(249, 115, 22, 0.05) !important;
        border-left: 4px solid var(--color-high) !important;
        border-top: 1px solid rgba(249, 115, 22, 0.1) !important;
        border-right: 1px solid rgba(249, 115, 22, 0.1) !important;
        border-bottom: 1px solid rgba(249, 115, 22, 0.1) !important;
        padding: 1.25rem !important;
        border-radius: var(--radius-lg) !important;
        color: var(--text-primary) !important;
    }
    .ha-appt-doctor-card h4 {
        color: #fb923c !important;
        margin-top: 0 !important;
        font-family: 'Outfit', sans-serif !important;
    }
    .ha-appt-doctor-card p {
        margin: 6px 0 !important;
        color: var(--text-secondary) !important;
        font-size: 0.9rem !important;
    }
    .ha-appt-doctor-card p strong {
        color: var(--text-primary) !important;
    }

    /* ── Doctor-specific workload cards ── */
    .ha-doctor-appointment-card {
        border-left: 4px solid;
        padding: 1.25rem;
        margin-bottom: 1rem;
        background: rgba(17, 25, 40, 0.45) !important;
        border-radius: 0 var(--radius-md) var(--radius-md) 0;
        border-top: 1px solid var(--border-color) !important;
        border-right: 1px solid var(--border-color) !important;
        border-bottom: 1px solid var(--border-color) !important;
        box-shadow: var(--shadow-sm);
        color: var(--text-primary) !important;
    }
    .ha-doctor-appointment-card h4 {
        margin: 0 0 10px 0 !important;
        color: #ffffff !important;
        font-family: 'Outfit', sans-serif !important;
        font-size: 1.05rem !important;
    }
    .ha-doctor-appointment-card p {
        margin: 5px 0 !important;
        color: var(--text-secondary) !important;
        font-size: 0.9rem !important;
    }
    .ha-doctor-appointment-card p strong {
        color: var(--text-primary) !important;
    }
    .ha-doctor-appointment-card hr {
        margin: 12px 0 !important;
        border: none !important;
        border-top: 1px solid var(--border-color) !important;
    }

    /* ── Medication Safety Reports ── */
    .ha-interaction {
        display: flex;
        gap: 1rem;
        align-items: flex-start;
        padding: 1.25rem;
        margin-bottom: 0.75rem;
        border: 1px solid var(--border-color) !important;
        border-left: 4px solid !important;
        border-radius: var(--radius-md) !important;
        box-shadow: var(--shadow-sm);
        background: rgba(17, 25, 40, 0.35) !important;
        line-height: 1.5;
    }
    .ha-interaction, .ha-interaction *, .ha-interaction div, .ha-interaction span {
        color: var(--text-secondary) !important;
    }
    .ha-interaction strong {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    .ha-interaction-safe {
        border-left-color: var(--color-low) !important;
        background: rgba(16, 185, 129, 0.05) !important;
    }
    .ha-interaction-minor {
        border-left-color: var(--color-medium) !important;
        background: rgba(251, 191, 36, 0.05) !important;
    }
    .ha-interaction-icon {
        display: grid;
        place-items: center;
        width: 28px; height: 28px;
        flex: 0 0 28px;
        border-radius: 8px;
        background: rgba(255, 255, 255, 0.06) !important;
        color: var(--text-primary) !important;
        font-weight: 700 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }

    @media (max-width: 900px) {
        .block-container {
            padding: 1rem 1rem 3rem !important;
            max-width: 100% !important;
        }
        .ha-page-hero { padding: 1.5rem 1.25rem; border-radius: var(--radius-md); }
        .ha-page-hero h1 { font-size: 1.65rem !important; }
        [data-testid="stTabContent"] {
            padding: 1.5rem 1.25rem !important;
            border-radius: var(--radius-md) !important;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 8px 12px !important;
            font-size: 0.8rem !important;
        }
        .ha-login-hero { padding-top: 1.5rem; }
        .ha-login-hero h1 { font-size: 2.5rem !important; }
        .ha-patient-grid {
            grid-template-columns: 1fr;
        }
        .ha-patient-item[style*="grid-column"] {
            grid-column: span 1 !important;
        }
    }
</style>
""", unsafe_allow_html=True)


def apply_plotly_theme(fig, title=""):
    fig.update_layout(
        font_family="Plus Jakarta Sans",
        font_color="#94a3b8",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=50 if title else 20, b=20),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.06)",
            linecolor="rgba(255,255,255,0.1)",
            zerolinecolor="rgba(255,255,255,0.1)",
            title_font=dict(size=11, color="#94a3b8"),
            tickfont=dict(size=10, color="#94a3b8")
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.06)",
            linecolor="rgba(255,255,255,0.1)",
            zerolinecolor="rgba(255,255,255,0.1)",
            title_font=dict(size=11, color="#94a3b8"),
            tickfont=dict(size=10, color="#94a3b8")
        )
    )
    if title:
        fig.update_layout(
            title={
                'text': title,
                'font': {'family': 'Outfit', 'size': 14, 'color': '#ffffff'},
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            }
        )
    fig.update_traces(marker=dict(line=dict(color='rgba(0,0,0,0)')))
    return fig


def render_section_header(title, subtitle=""):
    sub = f'<p class="ha-section-sub">{subtitle}</p>' if subtitle else ""
    st.markdown(f'<div class="ha-section-head"><h3>{title}</h3>{sub}</div>', unsafe_allow_html=True)


def render_form_header(title, subtitle=""):
    sub = f'<p>{subtitle}</p>' if subtitle else ""
    st.markdown(f'<div class="ha-form-head"><h3>{title}</h3>{sub}</div>', unsafe_allow_html=True)


def render_page_hero(title, subtitle, badge=""):
    badge_html = f'<span class="ha-hero-badge">{badge}</span>' if badge else ""
    st.markdown(f"""
    <div class="ha-page-hero">
        <h1>{title}</h1>
        <p class="ha-hero-sub">{subtitle}</p>
        {badge_html}
    </div>
    """, unsafe_allow_html=True)


def render_sidebar_brand(fullname, role):
    st.sidebar.markdown("""
    <div class="ha-sidebar-brand">
        <div class="ha-sb-icon">⚕️</div>
        <h3>HealthAI</h3>
        <p>Clinical Intelligence</p>
    </div>
    """, unsafe_allow_html=True)
    st.sidebar.markdown(f"""
    <div class="ha-user-chip">
        <div class="ha-name">{fullname}</div>
        <div class="ha-role">{role}</div>
    </div>
    """, unsafe_allow_html=True)

# Helper to check backend status & run query
def call_api(method: str, endpoint: str, data: dict = None, params: dict = None):
    """Call FastAPI backend, fallback to direct database access if offline."""
    url = f"{API_BASE_URL}{endpoint}"
    try:
        if method.lower() == "post":
            r = requests.post(url, json=data, params=params, timeout=3)
        elif method.lower() == "get":
            r = requests.get(url, params=params, timeout=3)
        elif method.lower() == "patch":
            r = requests.patch(url, params=params, json=data, timeout=3)
            
        if r.status_code == 200:
            return r.json(), False # returns data, is_fallback=False
        else:
            return None, True
    except Exception:
        # Backend is offline, use direct fallback
        return None, True

# ----------------- SESSION STATE SETUP -----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_role" not in st.session_state:
    st.session_state.user_role = None
if "username" not in st.session_state:
    st.session_state.username = None
if "user_fullname" not in st.session_state:
    st.session_state.user_fullname = None
if "user_id" not in st.session_state:
    st.session_state.user_id = None

# ----------------- DB BACKEND FALLBACK HELPERS -----------------
def local_login(username, password):
    db = SessionLocal()
    try:
        p_hash = hash_password(password)
        u = db.query(User).filter(User.username == username, User.password_hash == p_hash).first()
        if u:
            return {"id": u.id, "username": u.username, "role": u.role, "full_name": u.full_name}
        return None
    finally:
        db.close()

def local_register(username, password, role, full_name):
    db = SessionLocal()
    try:
        existing = db.query(User).filter(User.username == username).first()
        if existing:
            return None
        new_u = User(username=username, password_hash=hash_password(password), role=role, full_name=full_name)
        db.add(new_u)
        db.commit()
        db.refresh(new_u)
        return {"id": new_u.id, "username": new_u.username, "role": new_u.role, "full_name": new_u.full_name}
    finally:
        db.close()

def local_list_patients(search=None):
    db = SessionLocal()
    try:
        q = db.query(Patient)
        if search:
            q = q.filter(Patient.name.like(f"%{search}%"))
        res = []
        for p in q.all():
            res.append({
                "id": p.id, "user_id": p.user_id, "name": p.name, "age": p.age, "gender": p.gender, 
                "contact": p.contact, "medical_history": p.medical_history, "allergies": p.allergies
            })
        return res
    finally:
        db.close()

def local_get_patient_for_user(user_id, full_name=None):
    db = SessionLocal()
    try:
        p = db.query(Patient).filter(Patient.user_id == user_id).first()
        if not p and full_name:
            p = db.query(Patient).filter(Patient.name.ilike(full_name)).first()
            if p and p.user_id is None:
                p.user_id = user_id
                db.commit()
                db.refresh(p)
        if not p:
            return None
        return {
            "id": p.id, "user_id": p.user_id, "name": p.name, "age": p.age, "gender": p.gender,
            "contact": p.contact, "medical_history": p.medical_history, "allergies": p.allergies
        }
    finally:
        db.close()

def local_create_patient(name, age, gender, contact, history, allergies, user_id=None):
    db = SessionLocal()
    try:
        p = Patient(
            user_id=user_id, name=name, age=age, gender=gender, contact=contact,
            medical_history=history, allergies=allergies
        )
        db.add(p)
        db.commit()
        db.refresh(p)
        return {"id": p.id, "name": p.name, "age": p.age, "gender": p.gender}
    finally:
        db.close()

def save_uploaded_bytes(file_bytes, original_name):
    try:
        upload_dir = os.path.join(PROJECT_ROOT, "data", "uploaded_reports")
        os.makedirs(upload_dir, exist_ok=True)
        filename = f"{int(time.time())}_{original_name}"
        file_path = os.path.join(upload_dir, filename)
        with open(file_path, "wb") as f:
            f.write(file_bytes)
        return file_path
    except Exception as e:
        log_debug(f"Error saving uploaded file: {e}")
        return None

def save_uploaded_file(uploaded_file):
    if uploaded_file is not None:
        return save_uploaded_bytes(uploaded_file.getvalue(), uploaded_file.name)
    return None

def resolve_image_path(image_path):
    if not image_path:
        return None
    if os.path.isabs(image_path) and os.path.exists(image_path):
        return image_path
    abs_path = os.path.join(PROJECT_ROOT, image_path)
    if os.path.exists(abs_path):
        return abs_path
    if os.path.exists(image_path):
        return image_path
    return None

def local_get_records(patient_id):
    db = SessionLocal()
    try:
        records = db.query(HealthRecord).filter(HealthRecord.patient_id == patient_id).order_by(HealthRecord.visit_date.desc()).all()
        res = []
        for r in records:
            res.append({
                "id": r.id, "visit_date": r.visit_date.strftime("%Y-%m-%d"), "symptoms": r.symptoms,
                "systolic_bp": r.systolic_bp, "diastolic_bp": r.diastolic_bp, "heart_rate": r.heart_rate,
                "temperature": r.temperature, "lab_results": r.lab_results, "diagnosis": r.diagnosis,
                "prescription": r.prescription, "notes": r.notes, "lab_report_image": r.lab_report_image
            })
        return res
    finally:
        db.close()

def local_analyze_medications(prescription_text, allergies=None, existing_meds=None):
    return analyze_prescription(prescription_text, allergies=allergies, existing_meds=existing_meds)

def local_save_prescription(patient_id, prescription_text, prescription_image=None):
    db = SessionLocal()
    try:
        patient = db.query(Patient).filter(Patient.id == patient_id).first()
        allergies = patient.allergies if patient else ""
        analysis = analyze_prescription(prescription_text, allergies=allergies)
        presc = PatientPrescription(
            patient_id=patient_id,
            prescription_text=prescription_text,
            prescription_image=prescription_image,
            drugs_detected=json.dumps(analysis["detected_drugs"]),
            analysis_json=json.dumps(analysis),
        )
        db.add(presc)
        db.commit()
        return True, analysis, None
    except Exception as e:
        db.rollback()
        log_debug(f"Error saving prescription for patient {patient_id}: {e}")
        return False, None, str(e)
    finally:
        db.close()

def local_get_prescriptions(patient_id):
    db = SessionLocal()
    try:
        records = db.query(PatientPrescription).filter(
            PatientPrescription.patient_id == patient_id
        ).order_by(PatientPrescription.created_at.desc()).all()
        res = []
        for r in records:
            res.append({
                "id": r.id,
                "prescription_text": r.prescription_text,
                "prescription_image": r.prescription_image,
                "drugs_detected": json.loads(r.drugs_detected or "[]"),
                "analysis": json.loads(r.analysis_json or "{}"),
                "created_at": r.created_at.strftime("%Y-%m-%d %H:%M"),
            })
        return res
    finally:
        db.close()

def render_medication_analysis(analysis):
    st.info(analysis.get("summary", ""))

    if analysis.get("detected_drugs"):
        st.markdown("#### Detected Medications")
        for drug in analysis["detected_drugs"]:
            dose = drug.get("dosage", "")
            uses = drug.get("uses", "")
            line = f"**{drug['name']}** — {drug['category']}"
            if dose:
                line += f" | Prescribed: {dose}"
            st.write(line)
            if uses:
                st.caption(f"Used for: {uses}")

    if analysis.get("allergy_alerts"):
        st.markdown("#### Allergy Alerts")
        for alert in analysis["allergy_alerts"]:
            st.markdown(
                f'<div class="ha-alert-item-critical"><strong>{alert["drug"]}</strong> — Allergy: {alert["allergy"]}<br>{alert["description"]}</div>',
                unsafe_allow_html=True,
            )

    profiles = analysis.get("drug_profiles") or analysis.get("side_effects") or []
    if profiles:
        st.markdown("#### Drug Details & Side Effects")
        for drug in profiles:
            with st.expander(f"{drug['name']} — {drug['category']}", expanded=True):
                if drug.get("uses"):
                    st.write(f"**What it treats:** {drug['uses']}")
                if drug.get("dosage"):
                    st.write(f"**From your prescription:** {drug['dosage']}")
                st.write("**Common side effects:**")
                for effect in drug.get("side_effects", []):
                    st.write(f"- {effect}")
                if drug.get("warnings"):
                    st.write("**Safety warnings:**")
                    for warn in drug["warnings"]:
                        st.write(f"- {warn}")

    if analysis.get("interaction_matrix"):
        st.markdown("#### Drug-to-Drug Interaction Matrix")
        for pair in analysis["interaction_matrix"]:
            if pair["status"] == "Safe":
                st.markdown(
                    f'<div class="ha-interaction ha-interaction-safe"><span class="ha-interaction-icon">OK</span>'
                    f'<div><strong>{pair["drug_a"]} + {pair["drug_b"]}</strong><br>{pair["description"]}</div></div>',
                    unsafe_allow_html=True,
                )
            elif pair["severity"] == "Major":
                st.markdown(
                    f'<div class="ha-alert-item-critical"><strong>{pair["drug_a"]} + {pair["drug_b"]}</strong> ({pair["severity"]})<br>{pair["description"]}</div>',
                    unsafe_allow_html=True,
                )
            elif pair["severity"] == "Moderate":
                st.markdown(
                    f'<div class="ha-alert-item-warning"><strong>{pair["drug_a"]} + {pair["drug_b"]}</strong> ({pair["severity"]})<br>{pair["description"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="ha-interaction ha-interaction-minor"><span class="ha-interaction-icon">!</span>'
                    f'<div><strong>{pair["drug_a"]} + {pair["drug_b"]} ({pair["severity"]})</strong><br>'
                    f'{pair["description"]}</div></div>',
                    unsafe_allow_html=True,
                )

    elif analysis.get("interactions"):
        st.markdown("#### Drug-to-Drug Interactions")
        for inter in analysis["interactions"]:
            drugs = " + ".join(inter["drugs"]) if isinstance(inter["drugs"], list) else inter["drugs"]
            css = "ha-alert-item-critical" if inter["severity"] == "Major" else "ha-alert-item-warning" if inter["severity"] == "Moderate" else "card"
            st.markdown(
                f'<div class="{css}"><strong>{inter["severity"]}:</strong> {drugs}<br>{inter["description"]}</div>',
                unsafe_allow_html=True,
            )

    if analysis.get("guideline_notes"):
        st.markdown("#### Clinical Guideline Notes")
        for note in analysis["guideline_notes"]:
            st.caption(f"Source: {note['source']}")
            st.write(f"**{note['drug']}:** {note['guideline_note']}")

def local_create_record(patient_id, symptoms, systolic_bp, diastolic_bp, heart_rate, temperature, lab, diag, presc, notes, lab_report_image=None):
    db = SessionLocal()
    try:
        rec = HealthRecord(
            patient_id=patient_id, symptoms=symptoms, systolic_bp=systolic_bp, diastolic_bp=diastolic_bp,
            heart_rate=heart_rate, temperature=temperature, lab_results=lab, diagnosis=diag,
            prescription=presc, notes=notes, visit_date=datetime.utcnow(), lab_report_image=lab_report_image
        )
        db.add(rec)
        db.commit()
        return True, None
    except Exception as e:
        db.rollback()
        log_debug(f"Error creating health record for patient {patient_id}: {e}")
        return False, str(e)
    finally:
        db.close()

def local_list_triage():
    db = SessionLocal()
    try:
        records = db.query(TriageRecord).order_by(TriageRecord.created_at.desc()).all()
        res = []
        for r in records:
            res.append({
                "id": r.id, "patient_id": r.patient_id, "patient_name": r.patient.name if r.patient else "Unknown",
                "patient_age": r.patient.age if r.patient else 0, "priority_level": r.priority_level,
                "symptom_severity": r.symptom_severity, "recommended_department": r.recommended_department,
                "status": r.status, "created_at": r.created_at
            })
        return res
    finally:
        db.close()

def local_submit_triage(patient_id, symptoms, age, sys_bp, dia_bp, hr, temp, lab):
    db = SessionLocal()
    try:
        triage_info = classify_triage(symptoms, age, sys_bp, dia_bp, hr, temp, lab)
        t_rec = TriageRecord(
            patient_id=patient_id, priority_level=triage_info["priority_level"],
            symptom_severity=symptoms, recommended_department=triage_info["recommended_department"],
            status="Pending"
        )
        db.add(t_rec)
        db.commit()
        return triage_info
    finally:
        db.close()

def local_resolve_triage(triage_id):
    db = SessionLocal()
    try:
        rec = db.query(TriageRecord).filter(TriageRecord.id == triage_id).first()
        if rec:
            rec.status = "Checked"
            db.commit()
            return True
        return False
    finally:
        db.close()

def local_list_appointments():
    db = SessionLocal()
    try:
        appts = db.query(Appointment).order_by(Appointment.appointment_date.asc()).all()
        res = []
        for a in appts:
            doc = db.query(User).filter(User.id == a.doctor_id).first() if a.doctor_id else None
            res.append({
                "id": a.id,
                "patient_id": a.patient.id if a.patient else None,
                "patient_name": a.patient.name if a.patient else "Unknown",
                "patient_age": a.patient.age if a.patient else 0,
                "patient_gender": a.patient.gender if a.patient else "Unknown",
                "patient_contact": a.patient.contact if a.patient else "Unknown",
                "patient_history": a.patient.medical_history if a.patient else "",
                "patient_allergies": a.patient.allergies if a.patient else "",
                "doctor_name": doc.full_name if doc else "Unassigned",
                "doctor_role": doc.role if doc else "Unassigned",
                "appointment_date": a.appointment_date.strftime("%Y-%m-%d %H:%M") if hasattr(a.appointment_date, "strftime") else str(a.appointment_date),
                "reason": a.reason,
                "status": a.status
            })
        return res
    finally:
        db.close()


def local_create_appointment(patient_id, doctor_id, date_str, reason):
    db = SessionLocal()
    try:
        target_date = datetime.fromisoformat(date_str)
        # 1. Doctor conflict check
        if doctor_id:
            conflict_doc = db.query(Appointment).filter(
                Appointment.doctor_id == doctor_id,
                Appointment.appointment_date == target_date,
                Appointment.status == "Scheduled"
            ).first()
            if conflict_doc:
                return False, "The selected doctor already has a scheduled appointment at this exact date and time."
                
        # 2. Patient conflict check
        conflict_pat = db.query(Appointment).filter(
            Appointment.patient_id == patient_id,
            Appointment.appointment_date == target_date,
            Appointment.status == "Scheduled"
        ).first()
        if conflict_pat:
            return False, "This patient already has a scheduled appointment at this exact date and time."

        appt = Appointment(
            patient_id=patient_id, doctor_id=doctor_id,
            appointment_date=target_date,
            reason=reason, status="Scheduled"
        )
        db.add(appt)
        db.commit()
        return True, "Appointment request submitted successfully!"
    except Exception as e:
        return False, f"Failed to schedule appointment: {str(e)}"
    finally:
        db.close()


def local_update_appointment(appt_id, status_val):
    db = SessionLocal()
    try:
        a = db.query(Appointment).filter(Appointment.id == appt_id).first()
        if a:
            a.status = status_val
            db.commit()
            return True
        return False
    finally:
        db.close()

def local_dashboard_data():
    db = SessionLocal()
    try:
        total_p = db.query(Patient).count()
        total_r = db.query(HealthRecord).count()
        total_a = db.query(Appointment).count()
        
        # disease trends
        recs = db.query(HealthRecord.diagnosis).all()
        counts = {}
        for r in recs:
            if r.diagnosis:
                d = r.diagnosis.split('(')[0].strip()
                counts[d] = counts.get(d, 0) + 1
        disease_trends = [{"disease": k, "count": v} for k, v in counts.items()]
        disease_trends = sorted(disease_trends, key=lambda x: x["count"], reverse=True)[:5]
        
        # triage
        tc = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0}
        t_records = db.query(TriageRecord.priority_level).all()
        for t in t_records:
            if t.priority_level in tc:
                tc[t.priority_level] += 1
        triage_dist = [{"priority": k, "count": v} for k, v in tc.items()]
        
        # alerts
        alerts = []
        alert_recs = db.query(TriageRecord).filter(TriageRecord.priority_level.in_(["Critical", "High"]), TriageRecord.status == "Pending").all()
        for ar in alert_recs:
            alerts.append({
                "id": ar.id,
                "patient_id": ar.patient_id,
                "patient_name": ar.patient.name if ar.patient else "Unknown",
                "priority": ar.priority_level,
                "symptom_severity": ar.symptom_severity,
                "department": ar.recommended_department
            })
            
        return {
            "metrics": {"total_patients": total_p, "total_records": total_r, "total_appointments": total_a},
            "disease_trends": disease_trends,
            "triage_distribution": triage_dist,
            "alerts": alerts
        }
    finally:
        db.close()


# ----------------- LOGIN / REGISTER PAGE -----------------
def show_login_page():
    log_debug("show_login_page() entered...")
    st.markdown('<style>[data-testid="stSidebar"] { display: none !important; }</style>', unsafe_allow_html=True)
    st.markdown("""
    <div class="ha-login-hero">
        <div class="ha-logo-ring">⚕️</div>
        <h1>HealthAI</h1>
        <p>Intelligent clinical decision support, triage routing,<br>and patient care — powered by AI.</p>
        <div class="ha-feature-row">
            <span class="ha-pill">🧠 ML Risk Prediction</span>
            <span class="ha-pill">📚 RAG Guidelines</span>
            <span class="ha-pill">🚑 Smart Triage</span>
            <span class="ha-pill">💊 Drug Safety</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    log_debug("show_login_page() - header markdown rendered...")
    _, col_center, _ = st.columns([1, 1.1, 1])
    log_debug("show_login_page() - columns created...")
    
    with col_center:
        tab1, tab2 = st.tabs(["Sign In", "Create Account"])
        
        with tab1:
            render_form_header("Welcome back", "Sign in to access your clinical workspace")
            username = st.text_input("Username", key="login_user", placeholder="Enter your username")
            password = st.text_input("Password", type="password", key="login_pass", placeholder="Enter your password")
            
            if st.button("Sign In →", type="primary", use_container_width=True):
                # Try API login
                res, is_fallback = call_api("POST", "/auth/login", data={"username": username, "password": password})
                if is_fallback:
                    res = local_login(username, password)
                    
                if res:
                    st.session_state.logged_in = True
                    st.session_state.username = res["username"]
                    st.session_state.user_role = res["role"]
                    st.session_state.user_fullname = res["full_name"]
                    st.session_state.user_id = res["id"]
                    st.success(f"Welcome back, {res['full_name']} ({res['role']})!")
                    st.rerun()
                else:
                    st.error("Invalid credentials. Try seeding the database or register a new user.")
                    
        with tab2:
            render_form_header("Create your account", "Register as a patient to access care services")

            reg_fullname = st.text_input("Full Name", key="reg_fullname", placeholder="John Doe")
            reg_username = st.text_input("Username", key="reg_user", placeholder="Choose a username")
            reg_password = st.text_input("Password", type="password", key="reg_pass", placeholder="Create a secure password")
            
            # Public registration is restricted to Patient role only
            reg_role = "Patient"
            
            st.markdown('<p class="ha-section-title">Patient Profile</p>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                reg_age = st.number_input("Age", min_value=1, max_value=120, value=35, key="reg_age")
            with c2:
                reg_gender = st.selectbox("Gender", ["Male", "Female", "Other"], key="reg_gender")
            reg_contact = st.text_input("Contact Number", key="reg_contact", placeholder="e.g. 555-0105")
            
            if st.button("Create Account →", type="primary", use_container_width=True):
                if not reg_fullname or not reg_username or not reg_password or not reg_contact:
                    st.warning("All fields are required, including your contact number.")
                else:
                    res, is_fallback = call_api("POST", "/auth/register", data={
                        "username": reg_username, "password": reg_password, 
                        "role": reg_role, "full_name": reg_fullname
                    })
                    if is_fallback:
                        res = local_register(reg_username, reg_password, reg_role, reg_fullname)
                        
                    if res:
                        local_create_patient(
                            reg_fullname,
                            reg_age,
                            reg_gender,
                            reg_contact,
                            "None",
                            "None",
                            user_id=res["id"]
                        )
                        st.success("Registration successful! Switch to the login tab.")
                    else:
                        st.error("Username already exists.")



# ----------------- PATIENT WORKSPACE -----------------
def show_patient_workspace():
    render_page_hero(
        "Patient Workspace",
        f"Welcome, {st.session_state.user_fullname} — manage your health, appointments, and prescriptions.",
        badge="Patient Portal",
    )
    if "patient_success" in st.session_state:
        st.success(st.session_state.patient_success)
        del st.session_state.patient_success
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "Symptom & Triage",
        "Appointments",
        "Medical Reports",
        "Prescriptions & Safety",
    ])
    
    # Resolve patient profile linked to the logged-in user account
    patient_record = local_get_patient_for_user(
        st.session_state.user_id,
        st.session_state.user_fullname
    )
            
    if not patient_record:
        st.warning("⚠️ Welcome to HealthAI! Please complete your Patient Profile to unlock the workspace.")
        with st.form("complete_profile_form"):
            p_age = st.number_input("Your Age", min_value=1, max_value=120, value=35)
            p_gender = st.selectbox("Your Gender", ["Male", "Female", "Other"])
            p_contact = st.text_input("Contact Number", placeholder="e.g. 555-0199")
                
            if st.form_submit_button("Save Profile & Unlock Workspace"):
                if p_contact:
                    local_create_patient(
                        st.session_state.user_fullname,
                        p_age,
                        p_gender,
                        p_contact,
                        "None",
                        "None",
                        user_id=st.session_state.user_id
                    )
                    st.session_state.patient_success = "Profile saved successfully! Welcome to your dashboard."
                    st.rerun()
                else:
                    st.error("Contact number is required.")
        return
        
    patient_id = patient_record["id"]

    
    with tab1:
        render_section_header(
            "AI Symptom Triage",
            "Describe your symptoms and vitals. Our system will prioritize your case and route you to the right department.",
        )
        col1, col2 = st.columns(2)
        with col1:
            symptoms = st.text_area("What symptoms are you experiencing?", placeholder="e.g., Severe dry cough, high fever since yesterday, minor chest pressure.", height=120)
            age = st.number_input("Your Age", min_value=1, max_value=120, value=35)
            lab_results = st.text_area("Recent Lab Reports / Vitals Notes (Optional)", placeholder="e.g. Fasting sugar: 180", height=80)
            
        with col2:
            st.markdown('<p class="ha-vitals-label">Optional Home Vitals</p>', unsafe_allow_html=True)
            sys_bp = st.number_input("Systolic Blood Pressure (mmHg)", min_value=60, max_value=250, value=120)
            dia_bp = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=40, max_value=150, value=80)
            heart_rate = st.number_input("Heart Rate (bpm)", min_value=30, max_value=200, value=72)
            temp = st.number_input("Body Temperature (°C)", min_value=34.0, max_value=43.0, value=36.6, step=0.1)
            
        if st.button("Submit Symptoms to AI Triage", type="primary"):
            if not symptoms:
                st.warning("Please describe your symptoms.")
            else:
                # Call Triage submit
                res, is_fallback = call_api("POST", "/triage/submit", params={"patient_id": patient_id}, data={
                    "symptoms": symptoms, "age": age, "systolic_bp": sys_bp, "diastolic_bp": dia_bp,
                    "heart_rate": heart_rate, "temperature": temp, "lab_results": lab_results
                })
                if is_fallback:
                    res = local_submit_triage(patient_id, symptoms, age, sys_bp, dia_bp, heart_rate, temp, lab_results)
                
                t_result = res["triage_result"] if "triage_result" in res else res
                
                # Display result
                st.subheader("AI Patient Triage Assessment")
                priority = t_result["priority_level"]
                
                if priority == "Critical":
                    st.markdown(f'<div class="ha-alert-item-critical"><h3>🚨 Priority Level: CRITICAL</h3><p><strong>Reason:</strong> {t_result["triage_reason"]}</p><p><strong>Action Required:</strong> Please go to the nearest Emergency Room immediately. We have recommended the <strong>{t_result["recommended_department"]}</strong> department.</p></div>', unsafe_allow_html=True)
                elif priority == "High":
                    st.markdown(f'<div class="ha-alert-item-warning"><h3>⚠️ Priority Level: HIGH</h3><p><strong>Reason:</strong> {t_result["triage_reason"]}</p><p><strong>Action Required:</strong> Recommended Department: <strong>{t_result["recommended_department"]}</strong>. A nurse has been notified and you should be seen within 1 hour.</p></div>', unsafe_allow_html=True)
                else:
                    badge_class = "badge-medium" if priority == "Medium" else "badge-low"
                    st.markdown(f'<h4>Priority Level: <span class="badge {badge_class}">{priority}</span></h4>', unsafe_allow_html=True)
                    st.write(f"**Recommended Department:** {t_result['recommended_department']}")
                    st.write(f"**Reasoning:** {t_result['triage_reason']}")
                    st.write(f"**Time Frame Estimate:** {t_result['time_estimate']}")
                    st.success("Your symptoms have been logged successfully! The nursing team will review your triage level.")
                    
    with tab2:
        render_section_header("Book an Appointment", "Request a visit with a specialist at your preferred date and time.")
        reason = st.text_input("Reason for Appointment", placeholder="e.g. Follow-up checkup")
        appt_date = st.date_input("Preferred Date")
        appt_time = st.time_input("Preferred Time")
        
        # Load Doctors list
        db = SessionLocal()
        docs = db.query(User).filter(User.role == "Doctor").all()
        doc_opts = {d.full_name: d.id for d in docs}
        db.close()
        
        sel_doc_name = st.selectbox("Select Specialist (Optional)", ["Any Available"] + list(doc_opts.keys()))
        sel_doc_id = doc_opts.get(sel_doc_name)
        
        if st.button("Book Appointment"):
            full_datetime = f"{appt_date}T{appt_time}"
            success, msg = local_create_appointment(patient_id, sel_doc_id, full_datetime, reason)
            if success:
                st.session_state.patient_success = msg
                st.rerun()
            else:
                st.error(msg)

                
    with tab3:
        render_section_header("Clinical Reports", "Your visit history, diagnoses, and attached lab documents.")
        records = local_get_records(patient_id)
        if not records:
            st.info("No past clinical records found.")
        for r in records:
            with st.expander(f"📁 Visit on {r['visit_date']} - Diagnosis: {r['diagnosis']}"):
                st.write(f"**Symptoms:** {r['symptoms']}")
                st.write(f"**Vitals:** BP: {r['systolic_bp']}/{r['diastolic_bp']} mmHg | HR: {r['heart_rate']} bpm | Temp: {r['temperature']}°C")
                if r['lab_results']:
                    st.write(f"**Lab Reports:** {r['lab_results']}")
                st.write(f"**Prescription:** {r['prescription']}")
                if r['notes']:
                    st.write(f"**Doctor Notes:** {r['notes']}")
                if r.get('lab_report_image'):
                    st.write("**Attached Lab Report Photo:**")
                    image_path = resolve_image_path(r['lab_report_image'])
                    if image_path:
                        st.image(image_path, caption="Lab Report Document", use_container_width=True)
                    else:
                        st.warning("Lab report image file could not be found on the server.")

    with tab4:
        render_section_header(
            "Prescription & Drug Safety",
            "Enter or upload prescriptions to check drug interactions, side effects, and allergy warnings.",
        )

        if patient_record.get("allergies") and patient_record["allergies"].lower() not in ("none", "n/a", ""):
            st.warning(f"**Your documented allergies:** {patient_record['allergies']}")

        col_rx1, col_rx2 = st.columns(2)
        with col_rx1:
            rx_text = st.text_area(
                "Enter your medications",
                placeholder="e.g.\nMetformin 500mg twice daily\nAmlodipine 5mg once daily\nIbuprofen 400mg as needed",
                height=160,
                key="patient_rx_text",
            )
        with col_rx2:
            rx_upload = st.file_uploader(
                "Upload prescription photo (optional)",
                type=["png", "jpg", "jpeg", "pdf"],
                key="patient_rx_upload",
            )
            if rx_upload is not None:
                file_bytes = rx_upload.getvalue()
                st.session_state["patient_rx_cache"] = {
                    "bytes": file_bytes,
                    "name": rx_upload.name,
                }
                if st.session_state.get("patient_rx_ocr_name") != rx_upload.name:
                    with st.spinner("Reading prescription image (one-time)..."):
                        st.session_state["patient_rx_ocr_text"] = extract_text_from_prescription_image(
                            file_bytes, rx_upload.name
                        )
                        st.session_state["patient_rx_ocr_name"] = rx_upload.name
                st.caption(f"Attached: **{rx_upload.name}**")
                if st.session_state.get("patient_rx_ocr_text"):
                    with st.expander("Text read from prescription image"):
                        st.text(st.session_state["patient_rx_ocr_text"])

        existing_rx = []
        for r in local_get_records(patient_id):
            if r.get("prescription") and r["prescription"].lower() not in ("none", "n/a", ""):
                existing_rx.append(r["prescription"])

        if existing_rx:
            with st.expander("Medications from your doctor visits (included in analysis)"):
                for rx in existing_rx:
                    st.write(f"- {rx}")

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            analyze_clicked = st.button("🔍 Analyze Medications", type="primary", key="analyze_rx_btn")
        with btn_col2:
            save_clicked = st.button("💾 Save Prescription", key="save_rx_btn")

        combined_text = rx_text or ""
        for rx in existing_rx:
            if combined_text:
                combined_text += "\n"
            combined_text += rx

        ocr_text = st.session_state.get("patient_rx_ocr_text", "")
        if ocr_text:
            combined_text = f"{combined_text}\n{ocr_text}".strip() if combined_text else ocr_text

        if analyze_clicked or save_clicked:
            if not combined_text.strip():
                st.error("Please enter your medications or upload a prescription image.")
            else:
                cache = st.session_state.get("patient_rx_cache")
                with st.spinner("Analyzing medications..."):
                    analysis = local_analyze_medications(
                        combined_text, allergies=patient_record.get("allergies", "")
                    )
                st.session_state["last_rx_analysis"] = analysis

                if save_clicked:
                    image_path = None
                    if cache:
                        upload_dir = os.path.join(PROJECT_ROOT, "data", "uploaded_prescriptions")
                        os.makedirs(upload_dir, exist_ok=True)
                        image_path = os.path.join(upload_dir, f"{int(time.time())}_{cache['name']}")
                        with open(image_path, "wb") as f:
                            f.write(cache["bytes"])

                    success, _, err = local_save_prescription(patient_id, combined_text, image_path)
                    if success:
                        st.session_state.pop("patient_rx_cache", None)
                        st.session_state.pop("patient_rx_ocr_text", None)
                        st.session_state.pop("patient_rx_ocr_name", None)
                        st.session_state.patient_success = "Prescription saved and analyzed successfully!"
                        st.rerun()
                    else:
                        st.error(f"Failed to save prescription: {err}")

        if "last_rx_analysis" in st.session_state:
            st.markdown("---")
            st.subheader("Medication Safety Report")
            render_medication_analysis(st.session_state["last_rx_analysis"])

        st.markdown("---")
        st.subheader("Saved Prescriptions")
        saved = local_get_prescriptions(patient_id)
        if not saved:
            st.info("No saved prescriptions yet. Analyze and save your medications above.")
        else:
            for p in saved:
                drugs = ", ".join(d["name"] for d in p["drugs_detected"]) or "No drugs detected"
                with st.expander(f"📄 {p['created_at']} — {drugs}"):
                    st.write(f"**Medications:** {p['prescription_text']}")
                    if p.get("prescription_image"):
                        img_path = resolve_image_path(p["prescription_image"])
                        if img_path:
                            st.image(img_path, caption="Prescription", use_container_width=True)
                    if p.get("analysis"):
                        render_medication_analysis(p["analysis"])

# ----------------- NURSE WORKSPACE (TRIAGE) -----------------
def show_nurse_workspace():
    render_page_hero(
        "Triage & Monitoring",
        f"{st.session_state.user_fullname} — prioritize patients and log vitals in real time.",
        badge="Nurse Station",
    )
    if "nurse_success" in st.session_state:
        st.success(st.session_state.nurse_success)
        del st.session_state.nurse_success
    
    tab1, tab2 = st.tabs(["Triage Queue", "Vital Logger"])
    
    with tab1:
        st.header("Patient Triage Priority Board")
        st.write("Automatically prioritize patients based on severity. Clear patients after checking them in.")
        
        # Load Triage Queue
        t_records = local_list_triage()
        
        if not t_records:
            st.success("No active triage records in queue.")
        else:
            # Convert to DataFrame
            df = pd.DataFrame(t_records)
            # Filter pending
            df_pending = df[df["status"] == "Pending"]
            
            if df_pending.empty:
                st.success("All patient triage files checked!")
            else:
                # Custom sorting order: Critical, High, Medium, Low
                priority_weights = {"Critical": 1, "High": 2, "Medium": 3, "Low": 4}
                df_pending["weight"] = df_pending["priority_level"].map(priority_weights)
                df_pending = df_pending.sort_values("weight")
                
                for idx, row in df_pending.iterrows():
                    p_level = row["priority_level"]
                    if p_level == "Critical":
                        border_color = "#ff4b4b"
                        badge_color = "badge-critical"
                    elif p_level == "High":
                        border_color = "#e36414"
                        badge_color = "badge-high"
                    elif p_level == "Medium":
                        border_color = "#fb8b24"
                        badge_color = "badge-medium"
                    else:
                        border_color = "#2a9d8f"
                        badge_color = "badge-low"
                        
                    with st.container():
                        st.markdown(f"""
                        <div class="ha-triage-card" style="border-left-color: {border_color};">
                            <h4 class="ha-triage-card-title">Patient: {row['patient_name']} (Age: {row['patient_age']})
                                <span class="badge {badge_color}">{p_level}</span>
                            </h4>
                            <p class="ha-triage-card-text"><strong>Symptoms:</strong> {row['symptom_severity']}</p>
                            <p class="ha-triage-card-text"><strong>Department:</strong> {row['recommended_department']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if st.button("Mark as Admitted/Checked", key=f"res_{row['id']}"):
                            # Resolve triage
                            res, is_fallback = call_api("POST", f"/triage/{row['id']}/resolve")
                            if is_fallback:
                                local_resolve_triage(row["id"])
                            st.session_state.nurse_success = f"Patient {row['patient_name']} checked in!"
                            st.rerun()

    with tab2:
        st.header("Log Patient Vitals & Demographics")
        st.write("Register a patient details and vital signs to pre-triage them in the clinic.")
        
        # Select existing patient or register new
        p_list = local_list_patients()
        p_opts = {p["name"]: p["id"] for p in p_list}
        
        sel_name = st.selectbox("Select Patient Profile", ["-- Register New Patient --"] + list(p_opts.keys()))
        
        if sel_name == "-- Register New Patient --":
            st.subheader("New Patient Registration")
            new_name = st.text_input("Full Name")
            new_age = st.number_input("Age", min_value=1, max_value=120, value=35)
            new_gender = st.selectbox("Gender", ["Female", "Male", "Other"])
            new_contact = st.text_input("Contact Number")
            new_history = st.text_area("Medical History (Comorbidities)")
            new_allergies = st.text_input("Known Allergies")
            
            if st.button("Save New Patient"):
                if new_name:
                    p = local_create_patient(new_name, new_age, new_gender, new_contact, new_history, new_allergies)
                    st.session_state.nurse_success = f"Patient {new_name} registered successfully with ID {p['id']}!"
                    st.rerun()
        else:
            p_id = p_opts[sel_name]
            # Fetch patient details
            db = SessionLocal()
            patient_obj = db.query(Patient).filter(Patient.id == p_id).first()
            db.close()
            
            st.write(f"**Age:** {patient_obj.age} | **Gender:** {patient_obj.gender} | **Allergies:** {patient_obj.allergies}")
            
            st.subheader("Log Vital Signs")
            sys_bp = st.number_input("Systolic Blood Pressure (mmHg)", min_value=60, max_value=250, value=120, key="n_sys")
            dia_bp = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=40, max_value=150, value=80, key="n_dia")
            heart_rate = st.number_input("Heart Rate (bpm)", min_value=30, max_value=200, value=72, key="n_hr")
            temp = st.number_input("Temperature (°C)", min_value=34.0, max_value=43.0, value=36.6, step=0.1, key="n_temp")
            symptoms = st.text_area("Active Symptoms", key="n_symp")
            
            if st.button("Log Vitals & Run AI Triage"):
                res = local_submit_triage(p_id, symptoms, patient_obj.age, sys_bp, dia_bp, heart_rate, temp, "")
                st.success(f"Vitals Logged! AI Triage Level: {res['priority_level']} (Routed to {res['recommended_department']})")

# ----------------- DOCTOR WORKSPACE (DECISION SUPPORT) -----------------
def show_doctor_workspace():
    render_page_hero(
        "Clinical Decision Support",
        f"Dr. {st.session_state.user_fullname} — AI-assisted diagnosis, triage, and treatment planning.",
        badge="Physician Console",
    )
    if "doctor_success" in st.session_state:
        st.success(st.session_state.doctor_success)
        del st.session_state.doctor_success
    
    # 1. Search and Select Patient
    appts = local_list_appointments()
    my_fullname = st.session_state.user_fullname
    
    # Filter doctor appointments
    my_appts = [a for a in appts if a.get("doctor_name") == my_fullname]
    my_scheduled = [a for a in my_appts if a.get("status") == "Scheduled"]
    
    selected_patient_id = None
    
    tab_my_sched, tab_search_all = st.tabs(["My Schedule", "Search Patients"])
    
    with tab_my_sched:
        if not my_scheduled:
            st.info("You have no pending scheduled appointments.")
        else:
            # Show a clean table of appointments
            sched_data = []
            for a in my_scheduled:
                sched_data.append({
                    "Time": a["appointment_date"],
                    "Patient": a["patient_name"],
                    "Age/Gender": f"{a['patient_age']} / {a['patient_gender']}",
                    "Reason": a["reason"]
                })
            st.dataframe(pd.DataFrame(sched_data), use_container_width=True)
            
            # Interactive dropdown to pick a scheduled patient
            options = {f"{a['appointment_date']} - {a['patient_name']} (Reason: {a['reason']})": a["patient_id"] for a in my_scheduled}
            sel_appt_str = st.selectbox("Select patient from schedule to begin diagnosis", list(options.keys()), key="doc_sel_appt")
            selected_patient_id = options[sel_appt_str]
            
        # History of completed/cancelled appointments
        my_history = [a for a in my_appts if a.get("status") in ["Completed", "Cancelled"]]
        if my_history:
            with st.expander("📁 View Past Appointments History"):
                hist_data = [{"Date & Time": h["appointment_date"], "Patient": h["patient_name"], "Reason": h["reason"], "Status": h["status"]} for h in my_history]
                st.dataframe(pd.DataFrame(hist_data), use_container_width=True)
                
    with tab_search_all:
        patients_list = local_list_patients()
        if not patients_list:
            st.warning("No patients registered in the database.")
        else:
            p_opts = {f"{p['name']} (ID: {p['id']})": p["id"] for p in patients_list}
            sel_patient_str = st.selectbox("Search & Select Patient Profile from Database", list(p_opts.keys()), key="doc_search_all")
            selected_patient_id = p_opts[sel_patient_str]
            
    if not selected_patient_id:
        return
        
    patient_id = selected_patient_id

    
    # Fetch details
    db = SessionLocal()
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    db.close()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown(f"""
        <div class="card">
            <h3>👤 Patient Profile</h3>
            <div class="ha-patient-grid">
                <div class="ha-patient-item">
                    <span class="ha-patient-label">Name</span>
                    <span class="ha-patient-value">{patient.name}</span>
                </div>
                <div class="ha-patient-item">
                    <span class="ha-patient-label">Age & Gender</span>
                    <span class="ha-patient-value">{patient.age} / {patient.gender}</span>
                </div>
                <div class="ha-patient-item">
                    <span class="ha-patient-label">Contact</span>
                    <span class="ha-patient-value">{patient.contact}</span>
                </div>
                <div class="ha-patient-item" style="grid-column: span 2;">
                    <span class="ha-patient-label">Medical History</span>
                    <span class="ha-patient-value">{patient.medical_history}</span>
                </div>
                <div class="ha-patient-item" style="grid-column: span 2;">
                    <span class="ha-patient-label">Known Allergies</span>
                    <span class="ha-patient-value ha-allergy-value">{patient.allergies}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.subheader("📁 Patient Health Records Timeline")
        records = local_get_records(patient_id)
        if not records:
            st.info("No clinical visit history for this patient.")
        else:
            for r in records:
                with st.expander(f"📅 Visit on {r['visit_date']} - Diagnosis: {r['diagnosis']}"):
                    st.write(f"**Symptoms:** {r['symptoms']}")
                    st.write(f"**Vitals:** BP: {r['systolic_bp']}/{r['diastolic_bp']} mmHg | HR: {r['heart_rate']} bpm | Temp: {r['temperature']}°C")
                    st.write(f"**Prescription:** {r['prescription']}")
                    if r['notes']:
                        st.write(f"**Doctor Notes:** {r['notes']}")
                    if r.get('lab_report_image'):
                        st.write("**Attached Lab Report Photo:**")
                        image_path = resolve_image_path(r['lab_report_image'])
                        if image_path:
                            st.image(image_path, caption="Lab Report Document", use_container_width=True)
                        else:
                            st.warning("Lab report image file could not be found on the server.")
                        
    st.markdown("---")
    st.subheader("🔬 AI-Powered Decision Support Module")
    
    # Load last record vitals for quick seeding
    last_rec = records[0] if records else None
    
    # Input panel for current checkup
    col_v1, col_v2, col_v3 = st.columns(3)
    with col_v1:
        symptoms_in = st.text_area("Reported Symptoms", value=last_rec["symptoms"] if last_rec else "Dry cough, mild fever, fatigue")
        sys_bp = st.number_input("Systolic BP (mmHg)", min_value=60, max_value=250, value=last_rec["systolic_bp"] if last_rec else 120)
        dia_bp = st.number_input("Diastolic BP (mmHg)", min_value=40, max_value=150, value=last_rec["diastolic_bp"] if last_rec else 80)
    with col_v2:
        heart_rate = st.number_input("Heart Rate (bpm)", min_value=30, max_value=200, value=last_rec["heart_rate"] if last_rec else 72)
        temp = st.number_input("Body Temp (°C)", min_value=34.0, max_value=43.0, value=last_rec["temperature"] if last_rec else 36.6, step=0.1)
        fasting_sugar = st.number_input("Fasting Blood Sugar (mg/dL)", min_value=50, max_value=400, value=105)
    with col_v3:
        cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=500, value=190)
        hemoglobin = st.number_input("Hemoglobin (g/dL)", min_value=5.0, max_value=20.0, value=13.5, step=0.1)
        lab_results = st.text_area("Other Lab Results Description", value=last_rec["lab_results"] if last_rec else "")
        
    st.write("**AI Predictor Checkboxes:**")
    col_c1, col_c2, col_c3, col_c4 = st.columns(4)
    with col_c1:
        cough_cb = st.checkbox("Active Cough", value=True)
    with col_c2:
        chest_cb = st.checkbox("Active Chest Pain", value=False)
    with col_c3:
        dyspnea_cb = st.checkbox("Active Dyspnea (Shortness of breath)", value=False)
    with col_c4:
        fatigue_cb = st.checkbox("Active Fatigue / Weakness", value=True)
        
    if st.button("Analyze Patient & Run AI Decision Support", type="primary"):
        # 1. Run Disease Risk Prediction
        req_pred = {
            "age": patient.age,
            "gender": patient.gender,
            "systolic_bp": sys_bp,
            "diastolic_bp": dia_bp,
            "heart_rate": heart_rate,
            "temperature": temp,
            "fasting_blood_sugar": fasting_sugar,
            "cholesterol": cholesterol,
            "hemoglobin": hemoglobin,
            "has_cough": cough_cb,
            "has_chest_pain": chest_cb,
            "has_dyspnea": dyspnea_cb,
            "has_fatigue": fatigue_cb
        }
        res_pred, is_fallback = call_api("POST", "/predict", data=req_pred)
        if is_fallback:
            res_pred = predict_disease_risk(**req_pred)
            
        # 2. Run Triage Recommendation
        req_triage = {
            "symptoms": symptoms_in,
            "age": patient.age,
            "systolic_bp": sys_bp,
            "diastolic_bp": dia_bp,
            "heart_rate": heart_rate,
            "temperature": temp,
            "lab_results": lab_results
        }
        res_triage, is_fallback = call_api("POST", "/triage", data=req_triage)
        if is_fallback:
            res_triage = classify_triage(**req_triage)
            
        # 3. Run RAG Treatment Recommendation
        req_rec = {
            "diagnosis": res_pred["predicted_disease"],
            "symptoms": symptoms_in,
            "medical_history": patient.medical_history
        }
        res_rec, is_fallback = call_api("POST", "/recommendations", data=req_rec)
        if is_fallback:
            res_rec = generate_treatment_recommendations(**req_rec)
            
        # Render Results
        st.markdown("### 📊 Clinical Diagnosis Report")
        
        c_res1, c_res2 = st.columns(2)
        with c_res1:
            st.markdown("#### Disease Risk Prediction (ML)")
            st.write(f"**Primary Predicted Risk:** `{res_pred['predicted_disease']}`")
            st.write(f"**Risk Score (Confidence):** `{res_pred['risk_score'] * 100:.1f}%`")
            
            # Draw probability breakdown chart
            risks_df = pd.DataFrame(res_pred["all_risks"])
            if not risks_df.empty:
                fig = px.bar(risks_df, x="probability", y="disease", orientation='h', 
                             color="probability",
                             color_continuous_scale=[[0.0, '#1e1b4b'], [0.5, '#6366f1'], [1.0, '#0df2c9']])
                apply_plotly_theme(fig, "Risk Probabilities Breakdown")
                st.plotly_chart(fig, use_container_width=True)
                
        with c_res2:
            st.markdown("#### Patient Triage & Routing")
            priority = res_triage["priority_level"]
            badge_class = "badge-critical" if priority == "Critical" else ("badge-high" if priority == "High" else "badge-medium")
            st.markdown(f"**Triage Level:** <span class='badge {badge_class}'>{priority}</span>", unsafe_allow_html=True)
            st.write(f"**Recommended Specialist/Dept:** `{res_triage['recommended_department']}`")
            st.write(f"**Contributing Factors / Reason:** `{res_triage['triage_reason']}`")
            
            # Contributing factors list
            st.write("**Explainability & Diagnostic Drivers:**")
            for factor in res_pred["contributing_factors"]:
                st.markdown(f"- **{factor['factor']}** (importance weight: `{factor['importance_weight']:.3f}`)")
                
        st.markdown("#### 📚 Grounded Treatment Recommendations (RAG Engine)")
        st.markdown(f"*Model Grounding: {res_rec.get('model_used', 'Local RAG')} | Guidelines Sourced: {', '.join(res_rec['sources'])}*")
        st.markdown(res_rec["recommendations"])
        
        # Save to Session State to import to visit record
        st.session_state.last_analysis = {
            "diagnosis": f"{res_pred['predicted_disease']} ({res_pred['risk_score'] * 100:.1f}% AI Risk)",
            "recommendations": res_rec["recommendations"],
            "symptoms": symptoms_in,
            "sys_bp": sys_bp,
            "dia_bp": dia_bp,
            "hr": heart_rate,
            "temp": temp,
            "lab": lab_results
        }
        
    # Section to submit a visit record
    if "last_analysis" in st.session_state:
        st.markdown("---")
        st.subheader("✍️ Commit Health Record / Write Prescription")
        la = st.session_state.last_analysis
        
        # Determine dynamic default prescription based on the diagnosis
        diag_lower = la["diagnosis"].lower()
        if "diabet" in diag_lower:
            default_presc = "Advised Metformin (500mg BID) and lifestyle modifications as retrieved from guidelines."
        elif "hypertens" in diag_lower:
            default_presc = "Advised Amlodipine (5mg daily) or Lisinopril (10mg daily) as retrieved from guidelines."
        elif "pneumonia" in diag_lower:
            default_presc = "Advised empirical antibiotic regimen (Amoxicillin 1g TID or Doxycycline 100mg BID) as retrieved from guidelines."
        elif "anemia" in diag_lower:
            default_presc = "Advised Iron supplementation and dietary adjustments as retrieved from guidelines."
        elif "coronary" in diag_lower or "cad" in diag_lower:
            default_presc = "Advised Aspirin (81mg daily), Beta-blockers, and Cardiology consultation as retrieved from guidelines."
        elif "influenza" in diag_lower:
            default_presc = "Advised supportive care, rest, hydration, and antipyretics as retrieved from guidelines."
        else:
            default_presc = "Advised supportive care and clinical follow-up as retrieved from guidelines."

        doc_diag = st.text_input("Final Diagnosis", value=la["diagnosis"])
        doc_presc = st.text_area("Prescription", value=default_presc)
        doc_notes = st.text_area("Consultation Notes", value=la["recommendations"][:300] + "...")
        
        if st.button("Save Health Record & Close Visit"):
            success, _ = local_create_record(
                patient_id, la["symptoms"], la["sys_bp"], la["dia_bp"], 
                la["hr"], la["temp"], la["lab"], doc_diag, doc_presc, doc_notes
            )
            if success:
                st.session_state.doctor_success = "Health record saved successfully to the patient history!"
                del st.session_state.last_analysis
                st.rerun()

# ----------------- ADMINISTRATOR WORKSPACE (ANALYTICS) -----------------
def show_admin_workspace():
    render_page_hero(
        "Operations Dashboard",
        f"{st.session_state.user_fullname} — hospital analytics, alerts, and clinical operations.",
        badge="Administrator",
    )
    if "admin_success" in st.session_state:
        st.success(st.session_state.admin_success)
        del st.session_state.admin_success

    
    # Load dashboard data
    data = local_dashboard_data()
    metrics = data["metrics"]
    disease_trends = data["disease_trends"]
    triage_dist = data["triage_distribution"]
    alerts = data["alerts"]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="ha-metric-card">
            <div class="metric-value">{metrics['total_patients']}</div>
            <div class="metric-label">Total Patients</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="ha-metric-card">
            <div class="metric-value">{metrics['total_records']}</div>
            <div class="metric-label">Clinical Visits</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="ha-metric-card">
            <div class="metric-value">{metrics['total_appointments']}</div>
            <div class="metric-label">Appointments</div>
        </div>
        """, unsafe_allow_html=True)
        
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        if not disease_trends:
            st.subheader("🩺 Top Diagnosed Conditions")
            st.info("No diagnoses recorded yet.")
        else:
            df_disease = pd.DataFrame(disease_trends)
            fig = px.bar(df_disease, x="count", y="disease", orientation='h', color="count",
                         color_continuous_scale=[[0.0, '#3b0764'], [1.0, '#a855f7']], labels={"count": "Visit Volume", "disease": "Diagnosis"})
            apply_plotly_theme(fig, "Top Diagnosed Conditions (Disease Volume)")
            st.plotly_chart(fig, use_container_width=True)
            
    with col_chart2:
        df_triage = pd.DataFrame(triage_dist)
        if df_triage["count"].sum() == 0:
            st.subheader("🚑 Triage Priority Distribution")
            st.info("No triage logs found.")
        else:
            fig = px.pie(df_triage, values="count", names="priority", hole=0.4,
                         color="priority", color_discrete_map={"Critical": "#ef4444", "High": "#f97316", "Medium": "#fbbf24", "Low": "#10b981"})
            fig.update_layout(
                font_family="Plus Jakarta Sans",
                font_color="#94a3b8",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=50, b=20),
                legend=dict(
                    font=dict(size=10, color="#94a3b8"),
                    orientation="h",
                    yanchor="bottom",
                    y=-0.25,
                    xanchor="center",
                    x=0.5
                ),
                title={
                    'text': "Triage Priority Distribution",
                    'font': {'family': 'Outfit', 'size': 14, 'color': '#ffffff'},
                    'y': 0.95,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
            
    st.markdown("---")
    
    col_alerts, col_exports = st.columns([2, 1])
    
    with col_alerts:
        st.subheader("🚨 AI-Powered Critical & High Risk Alerts")
        st.write("These patients require immediate checkup priority based on symptoms and vitals severity:")
        if not alerts:
            st.success("No active critical/high-risk patients requiring attention.")
        else:
            for alert in alerts:
                p_level = alert["priority"]
                badge_class = "badge-critical" if p_level == "Critical" else "badge-high"
                alert_class = "ha-alert-item-critical" if p_level == "Critical" else "ha-alert-item-warning"
                
                with st.container():
                    st.markdown(f"""
                    <div class="{alert_class}">
                        <strong>Patient:</strong> {alert['patient_name']} (ID: {alert['patient_id']}) | 
                        <strong>Priority:</strong> <span class="badge {badge_class}">{p_level}</span> | 
                        <strong>Department:</strong> {alert['department']}<br/>
                        <div style="margin-top: 6px;"><strong>Symptoms:</strong> {alert['symptom_severity']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button(f"📅 Book Emergency Appointment for {alert['patient_name']}", key=f"book_{alert['id']}", use_container_width=True):
                        now_str = datetime.now().isoformat()
                        # Book appointment
                        success, msg = local_create_appointment(
                            patient_id=alert["patient_id"],
                            doctor_id=None,
                            date_str=now_str,
                            reason=f"Emergency Appointment booked by Admin (Triage Level: {p_level})"
                        )
                        if success:
                            # Clear triage alert
                            local_resolve_triage(alert["id"])
                            st.session_state.admin_success = f"Emergency appointment booked for {alert['patient_name']}!"
                            st.rerun()
                        else:
                            st.error(msg)


                
    with col_exports:
        st.subheader("📥 Export Clinical Reports")
        st.write("Download real-time operational reports for hospital administration audits.")
        
        # Link to FastAPI endpoint or generate directly if offline
        st.markdown(f'<a href="{API_BASE_URL}/analytics/export/pdf" target="_blank" class="ha-export-btn ha-export-pdf">📄 Download PDF Audit Report</a>', unsafe_allow_html=True)
        st.markdown(f'<a href="{API_BASE_URL}/analytics/export/excel" target="_blank" class="ha-export-btn ha-export-xls">📊 Download Excel Report</a>', unsafe_allow_html=True)
        
        st.info("⚠️ Note: If FastAPI is not running, PDF/Excel generation requires the FastAPI background server to be running on port 8000. Start it using the unified run script.")

    st.markdown("---")
    st.subheader("📅 Scheduled Appointments & Clinical Staff Details")
    
    # Load appointments (try API, fallback to local database query)
    appts, is_fallback = call_api("GET", "/appointments")
    if is_fallback:
        appts = local_list_appointments()
        
    tab_appts, tab_pats, tab_upload, tab_staff = st.tabs(["Appointments", "Patients", "Lab Uploads", "Clinical Staff"])
    
    with tab_appts:
        st.write("Browse and manage all patient appointments along with assigned doctors and patient vitals.")
        
        if not appts:
            st.info("No appointments scheduled.")
        else:
            # Let's convert to a DataFrame for clean listing
            appts_data = []
            for a in appts:
                appts_data.append({
                    "Date & Time": a.get("appointment_date", ""),
                    "Patient Name": a.get("patient_name", "Unknown"),
                    "Assigned Doctor": a.get("doctor_name", "Unassigned"),
                    "Reason": a.get("reason", ""),
                    "Status": a.get("status", "Scheduled")
                })
            df_appts = pd.DataFrame(appts_data)
            st.dataframe(df_appts, use_container_width=True)
            
            # Interactive search & details
            st.write("#### 🔍 Detailed View (Patient & Doctor Profiles)")
            selected_appt_str = st.selectbox("Select Appointment to view full clinical details", 
                                             [f"{a['appointment_date']} - {a['patient_name']} (Dr. {a['doctor_name']})" for a in appts],
                                             key="admin_sel_appt")
            
            # Find the selected appointment
            selected_appt = None
            for a in appts:
                appt_str = f"{a['appointment_date']} - {a['patient_name']} (Dr. {a['doctor_name']})"
                if appt_str == selected_appt_str:
                    selected_appt = a
                    break
            
            if selected_appt:
                col_appt1, col_appt2 = st.columns(2)
                with col_appt1:
                    st.markdown(f"""
                    <div class="ha-appt-patient-card">
                        <h4>👤 Patient Details</h4>
                        <p><strong>Name:</strong> {selected_appt.get('patient_name', 'Unknown')}</p>
                        <p><strong>Age:</strong> {selected_appt.get('patient_age', 'N/A')} | <strong>Gender:</strong> {selected_appt.get('patient_gender', 'N/A')}</p>
                        <p><strong>Contact:</strong> {selected_appt.get('patient_contact', 'N/A')}</p>
                        <p><strong>Medical History:</strong> {selected_appt.get('patient_history', 'None')}</p>
                        <p><strong>Known Allergies:</strong> <span class="ha-allergy-value">{selected_appt.get('patient_allergies', 'None')}</span></p>
                    </div>
                    """, unsafe_allow_html=True)
                with col_appt2:
                    st.markdown(f"""
                    <div class="ha-appt-doctor-card">
                        <h4>🩺 Doctor & Appointment Info</h4>
                        <p><strong>Assigned Doctor:</strong> {selected_appt.get('doctor_name', 'Unassigned')}</p>
                        <p><strong>Doctor Role:</strong> {selected_appt.get('doctor_role', 'Unassigned')}</p>
                        <p><strong>Date & Time:</strong> {selected_appt.get('appointment_date', '')}</p>
                        <p><strong>Reason for Visit:</strong> {selected_appt.get('reason', '')}</p>
                        <p><strong>Status:</strong> <span class="badge badge-medium">{selected_appt.get('status', 'Scheduled')}</span></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Ability to cancel or complete appointment
                    col_act1, col_act2 = st.columns(2)
                    with col_act1:
                        if st.button("✅ Mark Completed", key=f"comp_{selected_appt['id']}"):
                            local_update_appointment(selected_appt['id'], "Completed")
                            st.session_state.admin_success = "Appointment marked as Completed!"
                            st.rerun()
                    with col_act2:
                        if st.button("❌ Cancel Appointment", key=f"cancel_{selected_appt['id']}"):
                            local_update_appointment(selected_appt['id'], "Cancelled")
                            st.session_state.admin_success = "Appointment marked as Cancelled!"
                            st.rerun()

                            
    with tab_pats:
        st.write("Full Patient Registry details:")
        patients_list = local_list_patients()
        if not patients_list:
            st.info("No patients registered.")
        else:
            df_patients = pd.DataFrame(patients_list)
            # Reorder and rename columns for display
            df_patients = df_patients.rename(columns={
                "name": "Full Name",
                "age": "Age",
                "gender": "Gender",
                "contact": "Contact Info",
                "medical_history": "Pre-existing Conditions",
                "allergies": "Allergies"
            })
            st.dataframe(df_patients, use_container_width=True)
            
    with tab_upload:
        st.header("🔬 Upload Patient Lab Reports & Photos")
        st.write("Upload a scanned lab report or photo for a patient to log it directly into their clinical history.")
        
        p_list = local_list_patients()
        if not p_list:
            st.warning("No patients registered in the database.")
        else:
            p_opts = {f"{p['name']} (ID: {p['id']})": p["id"] for p in p_list}
            sel_patient_str = st.selectbox("Select Patient Profile", list(p_opts.keys()), key="admin_upload_pat")
            patient_id = p_opts[sel_patient_str]

            uploaded_file = st.file_uploader(
                "Choose Lab Report Photo (PNG, JPG, JPEG)",
                type=["png", "jpg", "jpeg"],
                key="admin_lab_report_upload"
            )
            if uploaded_file is not None:
                st.session_state["admin_upload_cache"] = {
                    "bytes": uploaded_file.getvalue(),
                    "name": uploaded_file.name,
                    "size": uploaded_file.size,
                }
                st.caption(f"Selected file: **{uploaded_file.name}** ({uploaded_file.size:,} bytes)")

            symptoms = st.text_input("Symptoms / Reason for Test", value="Routine Lab Testing", key="admin_upload_symptoms")
            diagnosis = st.text_input("Diagnosis / Test Name", value="Lab Results Uploaded", key="admin_upload_diagnosis")
            lab_desc = st.text_area(
                "Lab Results Summary / Description",
                value="Lab report photo uploaded by administrator.",
                key="admin_upload_lab_desc"
            )
            notes = st.text_area("Administrative Notes", value="Uploaded by Administrator.", key="admin_upload_notes")

            sys_bp = st.number_input("Systolic BP (mmHg)", min_value=60, max_value=250, value=120, key="admin_upload_sys_bp")
            dia_bp = st.number_input("Diastolic BP (mmHg)", min_value=40, max_value=150, value=80, key="admin_upload_dia_bp")
            heart_rate = st.number_input("Heart Rate (bpm)", min_value=30, max_value=200, value=72, key="admin_upload_hr")
            temp = st.number_input("Body Temp (°C)", min_value=34.0, max_value=43.0, value=36.6, step=0.1, key="admin_upload_temp")

            if st.button("Upload and Save Report", type="primary", key="admin_upload_btn"):
                upload_cache = st.session_state.get("admin_upload_cache")
                if not upload_cache:
                    st.error("Please select a lab report photo to upload before saving.")
                else:
                    saved_path = save_uploaded_bytes(upload_cache["bytes"], upload_cache["name"])
                    if saved_path:
                        success, err = local_create_record(
                            patient_id=patient_id,
                            symptoms=symptoms,
                            systolic_bp=sys_bp,
                            diastolic_bp=dia_bp,
                            heart_rate=heart_rate,
                            temperature=temp,
                            lab=lab_desc,
                            diag=diagnosis,
                            presc="None",
                            notes=notes,
                            lab_report_image=saved_path
                        )
                        if success:
                            st.session_state.pop("admin_upload_cache", None)
                            st.session_state.admin_success = f"Lab report uploaded for {sel_patient_str}!"
                            st.rerun()
                        else:
                            st.error(f"Failed to save health record to database: {err}")
                    else:
                        st.error("Failed to save uploaded file.")

    with tab_staff:
        db = SessionLocal()
        try:
            staff_users = db.query(User).filter(User.role.in_(["Doctor", "Nurse", "Administrator", "Super Admin"])).all()
            doctors = [u for u in staff_users if u.role == "Doctor"]
            
            st.write("### 🩺 Doctor-Specific Workloads & Patient Schedules")
            st.write("Select a doctor to view their individual scheduled appointments and details of their patients:")
            
            if not doctors:
                st.warning("No doctors registered in the system.")
            else:
                doc_names = [d.full_name for d in doctors]
                selected_doc_name = st.selectbox("Select Doctor Profile", doc_names, key="admin_filter_doc")
                
                # Filter appointments for selected doctor
                doc_appts = [a for a in appts if a.get("doctor_name") == selected_doc_name]
                
                if not doc_appts:
                    st.info(f"Dr. {selected_doc_name} has no scheduled appointments.")
                else:
                    st.write(f"#### 📅 Schedule for Dr. {selected_doc_name} ({len(doc_appts)} visits)")
                    for a in doc_appts:
                        p_status = a.get("status", "Scheduled")
                        status_color = "#2a9d8f" if p_status == "Completed" else ("#e36414" if p_status == "Scheduled" else "#6c757d")
                        
                        with st.container():
                            st.markdown(f"""
                            <div class="ha-doctor-appointment-card" style="border-left-color: {status_color};">
                                <h4>📅 {a['appointment_date']} — Patient: {a['patient_name']} (Age: {a['patient_age']} | {a['patient_gender']})</h4>
                                <p><strong>Reason for Visit:</strong> {a['reason']}</p>
                                <p><strong>Status:</strong> <span style="color:{status_color}; font-weight:bold;">{p_status}</span></p>
                                <hr/>
                                <p><strong>Patient Contact:</strong> {a.get('patient_contact', 'N/A')}</p>
                                <p><strong>Pre-existing Conditions:</strong> {a.get('patient_history', 'None')}</p>
                                <p><strong>Known Allergies:</strong> <span class="ha-allergy-value">{a.get('patient_allergies', 'None')}</span></p>
                            </div>
                            """, unsafe_allow_html=True)
                            
            st.markdown("---")
            st.write("### 🏥 System Staff Directory")
            staff_data = [{"Full Name": u.full_name, "Username": u.username, "System Role": u.role} for u in staff_users]
            st.dataframe(pd.DataFrame(staff_data), use_container_width=True)
        finally:
            db.close()



# ----------------- SUPER ADMIN WORKSPACE -----------------
def show_superadmin_workspace():
    render_page_hero(
        "System Manager",
        "Database inspection, user administration, and platform configuration.",
        badge="Super Admin",
    )
    
    tab1, tab2 = st.tabs(["Database Inspector", "User Management"])
    
    with tab1:
        st.header("Database Table Inspector")
        db = SessionLocal()
        
        table_opt = st.selectbox("Select Database Table to Inspect", ["Users", "Patients", "Health Records", "Triage Records", "Appointments"])
        
        if table_opt == "Users":
            data = db.query(User).all()
            df = pd.DataFrame([{"id": u.id, "username": u.username, "role": u.role, "full_name": u.full_name} for u in data])
            st.dataframe(df, use_container_width=True)
        elif table_opt == "Patients":
            data = db.query(Patient).all()
            df = pd.DataFrame([{"id": p.id, "name": p.name, "age": p.age, "gender": p.gender, "contact": p.contact, "medical_history": p.medical_history, "allergies": p.allergies} for p in data])
            st.dataframe(df, use_container_width=True)
        elif table_opt == "Health Records":
            data = db.query(HealthRecord).all()
            df = pd.DataFrame([{"id": h.id, "patient_id": h.patient_id, "visit_date": h.visit_date, "symptoms": h.symptoms, "systolic_bp": h.systolic_bp, "diastolic_bp": h.diastolic_bp, "heart_rate": h.heart_rate, "temperature": h.temperature, "diagnosis": h.diagnosis, "prescription": h.prescription} for h in data])
            st.dataframe(df, use_container_width=True)
        elif table_opt == "Triage Records":
            data = db.query(TriageRecord).all()
            df = pd.DataFrame([{"id": t.id, "patient_id": t.patient_id, "priority_level": t.priority_level, "symptom_severity": t.symptom_severity, "recommended_department": t.recommended_department, "status": t.status, "created_at": t.created_at} for t in data])
            st.dataframe(df, use_container_width=True)
        elif table_opt == "Appointments":
            data = db.query(Appointment).all()
            df = pd.DataFrame([{"id": a.id, "patient_id": a.patient_id, "doctor_id": a.doctor_id, "appointment_date": a.appointment_date, "reason": a.reason, "status": a.status} for a in data])
            st.dataframe(df, use_container_width=True)
            
        db.close()
        
    with tab2:
        st.header("User Roles & Accounts Administration")
        db = SessionLocal()
        users = db.query(User).all()
        
        col_list, col_add = st.columns(2)
        
        with col_list:
            st.subheader("Registered System Users")
            user_data = [{"ID": u.id, "Username": u.username, "Role": u.role, "Full Name": u.full_name} for u in users]
            st.dataframe(pd.DataFrame(user_data), use_container_width=True)
            
        with col_add:
            st.subheader("Create System Admin Account")
            new_fullname = st.text_input("Admin Full Name", key="sa_name")
            new_username = st.text_input("Admin Username", key="sa_user")
            new_password = st.text_input("Password", type="password", key="sa_pass")
            new_role = st.selectbox("Role", ["Super Admin", "Administrator", "Doctor", "Nurse"], key="sa_role")
            
            if st.button("Create Account", key="sa_btn"):
                if new_fullname and new_username and new_password:
                    u = User(username=new_username, password_hash=hash_password(new_password), role=new_role, full_name=new_fullname)
                    db.add(u)
                    db.commit()
                    st.success(f"Account for {new_fullname} created successfully!")
                    st.rerun()
                else:
                    st.warning("All fields are required.")
                    
        db.close()

# ----------------- MAIN APP ROUTING -----------------
def main():
    log_debug("main() entry...")
    if not st.session_state.logged_in:
        log_debug("main() -> not logged in, rendering login page...")
        show_login_page()
    else:
        render_sidebar_brand(st.session_state.user_fullname, st.session_state.user_role)
        
        st.sidebar.markdown("---")
        if st.sidebar.button("Sign Out", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.session_state.user_role = None
            st.session_state.user_fullname = None
            st.session_state.user_id = None
            st.rerun()
            
        # Display backend status indicator
        url = f"{API_BASE_URL}/"
        try:
            r = requests.get(url, timeout=1)
            st.sidebar.markdown('<div class="ha-status-online">● API Connected</div>', unsafe_allow_html=True)
        except Exception:
            st.sidebar.markdown('<div class="ha-status-offline">● Offline — Direct DB</div>', unsafe_allow_html=True)
            
        # Render appropriate workspace based on selected role
        user_role = st.session_state.user_role
        if user_role == "Patient":
            show_patient_workspace()
        elif user_role == "Doctor":
            show_doctor_workspace()
        elif user_role == "Nurse":
            show_nurse_workspace()
        elif user_role == "Administrator":
            show_admin_workspace()
        elif user_role == "Super Admin":
            show_superadmin_workspace()

if __name__ == "__main__":
    main()
