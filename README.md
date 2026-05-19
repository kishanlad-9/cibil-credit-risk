# 🏦 CIBIL Credit Risk Predictor

A production-ready **ML + Web Application** for predicting loan default risk.

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **ROC-AUC** | **0.8026** ✅ |
| Accuracy | 89.4% |
| Macro F1 | 0.6843 |
| Stack | XGBoost + LightGBM + RF (Soft Voting) |
| Balancing | SMOTE oversampling |
| Features | 36 engineered features |

---

## 📁 Project Structure

```
cibil_credit_risk/
├── backend/
│   ├── main.py        ← FastAPI app (predict, login, signup)
│   ├── auth.py        ← JWT + bcrypt hashing
│   ├── model.pkl      ← Trained model bundle
│   └── users.json     ← User store (auto-created)
├── frontend/
│   └── index.html     ← React SPA — open directly in browser
├── ml_pipeline/
│   └── train_model.py ← Retrain script
├── requirements.txt
└── README.md
```

---

## 🚀 Setup — Windows / Mac / Linux

### Step 1 — Install dependencies

Open a terminal **inside the `cibil_credit_risk` folder**, then run:

```bash
pip install -r requirements.txt
```

> If you get a bcrypt error: `pip install bcrypt==4.0.1`

---

### Step 2 — Train the model *(skip — model.pkl is included)*

```bash
python ml_pipeline/train_model.py
```

---

### Step 3 — Start the backend

```bash
# Windows (run inside the cibil_credit_risk folder)
cd backend
uvicorn main:app --reload --port 8000

# Mac / Linux
cd backend && uvicorn main:app --reload --port 8000
```

Expected output:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
```

---

### Step 4 — Open the frontend

**Do NOT double-click `index.html`** — browsers block requests from `file://` URLs.

Instead, open this URL in your browser after starting the backend:

```
http://localhost:8000/app
```

```
Demo login:   demo@cibil.ai  /  demo1234
```

> The backend serves the frontend at `/app` to avoid browser CORS restrictions.

---

## 📡 API Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | `/auth/signup` | ❌ | Register |
| POST | `/auth/login` | ❌ | Login → JWT |
| POST | `/predict` | ✅ JWT | ML prediction |
| GET | `/model/info` | ✅ JWT | AUC, accuracy, F1 |
| GET | `/` | ❌ | Health check |

Swagger UI → `http://localhost:8000/docs`

---

## 🌐 Deployment

**Backend → Render / Railway**
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Root directory: `backend`
- Env var: `SECRET_KEY=some-long-random-string`

**Frontend → Vercel / Netlify**
- Edit `API_BASE` in `frontend/index.html` to your deployed backend URL
- Deploy `frontend/` as a static site — no build step needed
