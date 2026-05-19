"""
CIBIL Credit Risk Predictor — FastAPI Backend
Run from inside the backend/ folder:
    uvicorn main:app --reload --port 8000
Then open: http://localhost:8000/app
"""

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from auth import (
    create_access_token, verify_token,
    hash_password, verify_password,
    get_user, create_user, UserExists, UserNotFound,
    init_db, get_all_users
)

# ── App setup ────────────────────────────────────────────────
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app):
    init_db()   # create SQLite tables + seed demo user
    yield

app = FastAPI(
    title="CIBIL Credit Risk API",
    description="ML-powered loan default prediction API",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

# ── Load model bundle ────────────────────────────────────────
MODEL_PATH = Path(__file__).parent / "model.pkl"

def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

bundle  = load_model()
model   = bundle["model"]
scaler  = bundle["scaler"]
threshold = bundle["threshold"]
features  = bundle["features"]
num_cols  = bundle["num_cols"]

# ── Pydantic schemas ─────────────────────────────────────────
class SignupRequest(BaseModel):
    email: str
    password: str = Field(min_length=6)
    full_name: str

class LoginRequest(BaseModel):
    email: str
    password: str

class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict

class PredictRequest(BaseModel):
    cibil_score: int        = Field(ge=300, le=900)
    annual_income: int      = Field(ge=100000)
    loan_amount: int        = Field(ge=10000)
    loan_tenure: int        = Field(ge=6, le=360)
    active_loans: int       = Field(ge=0, le=20)
    credit_cards: int       = Field(ge=0, le=20)
    missed_payments: int    = Field(ge=0, le=12)
    dpd: int                = Field(ge=0, le=365)
    employment_type: Literal["Salaried", "Self-Employed", "Business"]
    city_tier: Literal["Tier1", "Tier2", "Tier3"]
    credit_utilization: float = Field(ge=0.0, le=1.0, default=0.3)
    num_enquiries: int      = Field(ge=0, le=20, default=1)

class PredictResponse(BaseModel):
    probability: float
    prediction: str       # "DEFAULT" | "NO DEFAULT"
    risk_level: str       # HIGH | MEDIUM | LOW
    confidence: str
    threshold_used: float
    risk_factors: list
    recommendation: str

# ── Helper: feature engineering ──────────────────────────────
def build_features(req: PredictRequest) -> pd.DataFrame:
    emi = int(req.loan_amount * 0.01)
    lti = req.loan_amount / req.annual_income

    row = {
        "cibil_score":      req.cibil_score,
        "annual_income":    req.annual_income,
        "loan_amount":      req.loan_amount,
        "loan_tenure":      req.loan_tenure,
        "emi":              emi,
        "active_loans":     req.active_loans,
        "credit_cards":     req.credit_cards,
        "missed_payments":  req.missed_payments,
        "dpd":              req.dpd,
        "credit_utilization": req.credit_utilization,
        "num_enquiries":    req.num_enquiries,

        # Engineered
        "loan_to_income":       lti,
        "emi_to_income":        emi / (req.annual_income / 12),
        "debt_burden":          req.active_loans * emi,
        "payment_stress":       req.missed_payments * 10 + req.dpd / 6,
        "cibil_bucket":         _cibil_bucket(req.cibil_score),
        "high_risk_flag":       int(req.cibil_score < 600 or req.missed_payments >= 3 or req.dpd > 60),
        "very_high_risk":       int(req.cibil_score < 500 or req.missed_payments >= 6 or req.dpd > 90),
        "clean_record":         int(req.missed_payments == 0 and req.dpd == 0),
        "over_leveraged":       int(req.active_loans >= 4),
        "high_util":            int(req.credit_utilization > 0.7),
        "cibil_x_missed":       (req.cibil_score / 900) * (1 - req.missed_payments / 12),
        "cibil_x_dpd":          (req.cibil_score / 900) * (1 - req.dpd / 180),
        "risk_composite":       int(req.cibil_score < 600 or req.missed_payments >= 3 or req.dpd > 60)
                                + int(req.cibil_score < 500 or req.missed_payments >= 6 or req.dpd > 90)
                                + int(req.active_loans >= 4),
        "log_income":           np.log1p(req.annual_income),
        "log_loan":             np.log1p(req.loan_amount),
        "affordability":        req.annual_income / (emi + 1),
        "cibil_sq":             req.cibil_score ** 2,
        "mp_sq":                req.missed_payments ** 2,
        "dpd_sq":               req.dpd ** 2,

        # One-hot employment_type
        "employment_type_Business":     int(req.employment_type == "Business"),
        "employment_type_Salaried":     int(req.employment_type == "Salaried"),
        "employment_type_Self-Employed": int(req.employment_type == "Self-Employed"),

        # One-hot city_tier
        "city_tier_Tier1":  int(req.city_tier == "Tier1"),
        "city_tier_Tier2":  int(req.city_tier == "Tier2"),
        "city_tier_Tier3":  int(req.city_tier == "Tier3"),
    }

    df = pd.DataFrame([row])
    # Ensure all expected features present
    for feat in features:
        if feat not in df.columns:
            df[feat] = 0
    df = df[features]

    # Scale numeric columns
    df[num_cols] = scaler.transform(df[num_cols])
    return df


def _cibil_bucket(score: int) -> float:
    bins = [300, 450, 550, 620, 680, 720, 780, 841, 901]
    labels = [0, 1, 2, 3, 4, 5, 6, 7]
    for i in range(len(bins) - 1):
        if bins[i] <= score < bins[i + 1]:
            return float(labels[i])
    return 7.0


def analyze_risk_factors(req: PredictRequest) -> list:
    factors = []
    lti = req.loan_amount / req.annual_income

    if req.cibil_score >= 800:
        factors.append({"factor": "CIBIL Score", "status": "green", "detail": f"Excellent ({req.cibil_score})"})
    elif req.cibil_score >= 750:
        factors.append({"factor": "CIBIL Score", "status": "green", "detail": f"Good ({req.cibil_score})"})
    elif req.cibil_score >= 700:
        factors.append({"factor": "CIBIL Score", "status": "yellow", "detail": f"Fair ({req.cibil_score})"})
    elif req.cibil_score >= 600:
        factors.append({"factor": "CIBIL Score", "status": "orange", "detail": f"Below average ({req.cibil_score})"})
    else:
        factors.append({"factor": "CIBIL Score", "status": "red", "detail": f"Poor ({req.cibil_score}) — major risk"})

    if req.missed_payments == 0:
        factors.append({"factor": "Payment History", "status": "green", "detail": "No missed payments"})
    elif req.missed_payments <= 2:
        factors.append({"factor": "Payment History", "status": "yellow", "detail": f"{req.missed_payments} missed"})
    elif req.missed_payments <= 5:
        factors.append({"factor": "Payment History", "status": "orange", "detail": f"{req.missed_payments} missed payments"})
    else:
        factors.append({"factor": "Payment History", "status": "red", "detail": f"{req.missed_payments} missed — severe"})

    if req.dpd == 0:
        factors.append({"factor": "Days Past Due", "status": "green", "detail": "Clean DPD record"})
    elif req.dpd <= 30:
        factors.append({"factor": "Days Past Due", "status": "yellow", "detail": f"{req.dpd} days"})
    elif req.dpd <= 90:
        factors.append({"factor": "Days Past Due", "status": "orange", "detail": f"{req.dpd} days past due"})
    else:
        factors.append({"factor": "Days Past Due", "status": "red", "detail": f"{req.dpd} days — severe delinquency"})

    if lti <= 1:
        factors.append({"factor": "Loan-to-Income", "status": "green", "detail": f"{lti:.2f}x — healthy"})
    elif lti <= 3:
        factors.append({"factor": "Loan-to-Income", "status": "yellow", "detail": f"{lti:.2f}x — moderate"})
    elif lti <= 5:
        factors.append({"factor": "Loan-to-Income", "status": "orange", "detail": f"{lti:.2f}x — high"})
    else:
        factors.append({"factor": "Loan-to-Income", "status": "red", "detail": f"{lti:.2f}x — very high"})

    if req.active_loans == 0:
        factors.append({"factor": "Active Loans", "status": "green", "detail": "No other loans"})
    elif req.active_loans <= 2:
        factors.append({"factor": "Active Loans", "status": "yellow", "detail": f"{req.active_loans} active loans"})
    else:
        factors.append({"factor": "Active Loans", "status": "red", "detail": f"{req.active_loans} active — over-leveraged"})

    if req.credit_utilization <= 0.3:
        factors.append({"factor": "Credit Utilization", "status": "green", "detail": f"{req.credit_utilization:.0%}"})
    elif req.credit_utilization <= 0.6:
        factors.append({"factor": "Credit Utilization", "status": "yellow", "detail": f"{req.credit_utilization:.0%}"})
    else:
        factors.append({"factor": "Credit Utilization", "status": "red", "detail": f"{req.credit_utilization:.0%} — high"})

    return factors


def get_recommendation(prob: float, threshold: float) -> str:
    pct = prob * 100
    if pct < 20:
        return "Strong candidate — approve with standard terms."
    elif pct < threshold * 100:
        return "Borderline candidate — consider reduced amount or slightly higher rate."
    elif pct < 70:
        return "High risk — reject or require collateral / co-applicant."
    else:
        return "Very high default risk — decline this application."


# ── Auth endpoints ───────────────────────────────────────────
@app.post("/auth/signup", response_model=AuthResponse)
def signup(req: SignupRequest):
    try:
        user = create_user(req.email, hash_password(req.password), req.full_name)
    except UserExists:
        raise HTTPException(status_code=400, detail="Email already registered")
    token = create_access_token({"sub": user["email"]})
    return {"access_token": token, "token_type": "bearer",
            "user": {"email": user["email"], "full_name": user["full_name"]}}


@app.post("/auth/login", response_model=AuthResponse)
def login(req: LoginRequest):
    try:
        user = get_user(req.email)
    except UserNotFound:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    if not verify_password(req.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = create_access_token({"sub": user["email"]})
    return {"access_token": token, "token_type": "bearer",
            "user": {"email": user["email"], "full_name": user["full_name"]}}


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    payload = verify_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    try:
        return get_user(payload["sub"])
    except UserNotFound:
        raise HTTPException(status_code=401, detail="User not found")


# ── Predict endpoint ─────────────────────────────────────────
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, current_user=Depends(get_current_user)):
    df = build_features(req)
    prob = float(model.predict_proba(df)[0][1])
    is_default = prob >= threshold

    risk_level = "HIGH" if prob > 0.65 else ("MEDIUM" if prob > 0.35 else "LOW")
    confidence = "High" if abs(prob - 0.5) > 0.25 else ("Moderate" if abs(prob - 0.5) > 0.12 else "Low")

    return {
        "probability":      round(prob, 4),
        "prediction":       "DEFAULT" if is_default else "NO DEFAULT",
        "risk_level":       risk_level,
        "confidence":       confidence,
        "threshold_used":   threshold,
        "risk_factors":     analyze_risk_factors(req),
        "recommendation":   get_recommendation(prob, threshold),
    }


# ── Serve frontend ───────────────────────────────────────────
FRONTEND = Path(__file__).parent.parent / "frontend" / "index.html"


@app.get("/app", include_in_schema=False)
@app.get("/app/", include_in_schema=False)
def serve_app():
    """Serve the React frontend so there are no file:// CORS issues."""
    if FRONTEND.exists():
        return FileResponse(str(FRONTEND), media_type="text/html")
    return {"error": "frontend/index.html not found — run from project root"}


# ── Health + model info ──────────────────────────────────────
@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "CIBIL Credit Risk API v2.0",
        "frontend": "Open http://localhost:8000/app in your browser",
    }


@app.get("/model/info")
def model_info(current_user=Depends(get_current_user)):
    return {
        "roc_auc":   round(bundle["auc"], 4),
        "accuracy":  round(bundle["accuracy"], 4),
        "macro_f1":  round(bundle["f1"], 4),
        "threshold": round(threshold, 2),
        "algorithm": "XGBoost + LightGBM + RandomForest Ensemble (Soft Voting)",
        "features":  len(features),
        "balancing": "SMOTE oversampling",
    }
