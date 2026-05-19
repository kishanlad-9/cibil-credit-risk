# 🏦 CIBIL Credit Risk Predictor

> An end-to-end ML web application that predicts loan default risk using CIBIL-style Indian credit bureau data — built with XGBoost + LightGBM + Random Forest ensemble, FastAPI backend, and a React frontend with JWT authentication.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?style=flat-square&logo=fastapi)
![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **ROC-AUC** | **0.8026** ✅ |
| Accuracy | 89.4% |
| Macro F1 | 0.6843 |
| Algorithm | XGBoost + LightGBM + RandomForest (Soft Voting) |
| Balancing | SMOTE oversampling |
| Features | 36 engineered features |
| Training samples | 10,000 synthetic CIBIL-style records |

---

## ✨ Features

- **ML Ensemble** — XGBoost + LightGBM + Random Forest with soft voting and SMOTE balancing
- **CIBIL-style features** — CIBIL score, DPD, missed payments, credit utilization, loan-to-income ratio
- **Risk Factor Breakdown** — per-applicant explanation of what's driving the risk
- **JWT Authentication** — secure login/signup with bcrypt password hashing
- **Prediction History** — session-level history of all predictions
- **Model Info Dashboard** — live AUC, accuracy, F1 from the deployed model
- **One-file frontend** — React SPA served by FastAPI, no build step needed
- **Deployable** — ready for Render/Railway (backend) + Vercel/Netlify (frontend)

---

## 📁 Project Structure

```
cibil_credit_risk/
├── backend/
│   ├── main.py            ← FastAPI app (predict, auth, serve frontend)
│   ├── auth.py            ← JWT + bcrypt + SQLite via SQLAlchemy
│   ├── model.pkl          ← Trained model bundle (included)
│   └── users.db           ← SQLite user store (auto-created)
├── frontend/
│   └── index.html         ← React SPA (no build step)
├── ml_pipeline/
│   └── train_model.py     ← Full retrain script
├── notebooks/             ← EDA notebooks (optional)
├── .gitignore
├── requirements.txt
├── render.yaml            ← One-click Render deployment
└── README.md
```

---

## 🚀 Local Setup

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/cibil-credit-risk.git
cd cibil-credit-risk
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the backend

```bash
cd backend
uvicorn main:app --reload --port 8000
```

### 4. Open the app

```
http://localhost:8000/app
```

**Demo credentials:** `demo@cibil.ai` / `demo1234`

---

## 📡 API Reference

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `POST` | `/auth/signup` | ❌ | Register new user |
| `POST` | `/auth/login` | ❌ | Login → returns JWT |
| `POST` | `/predict` | ✅ JWT | Run ML prediction |
| `GET` | `/model/info` | ✅ JWT | AUC, accuracy, F1 |
| `GET` | `/app` | ❌ | Serve React frontend |

**Swagger UI:** `http://localhost:8000/docs`

---

## 🌐 Deployment

### Backend → Render

1. Push to GitHub
2. New Web Service on [render.com](https://render.com)
3. Root directory: `backend` | Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Add env var: `SECRET_KEY=your-random-secret`

### Frontend → Vercel/Netlify

1. In `frontend/index.html`, set `const API = "https://your-render-url.onrender.com";`
2. Deploy the `frontend/` folder as a static site

---

## 🧠 ML Pipeline

- 10,000 synthetic CIBIL-style records with realistic default logic
- 36 engineered features: FOIR, loan-to-income, payment stress, CIBIL buckets, interaction terms
- RobustScaler → SMOTE → Soft-voting ensemble (XGBoost + LightGBM + RF)
- Threshold tuned to maximize macro F1

---

## 📄 License

MIT License
