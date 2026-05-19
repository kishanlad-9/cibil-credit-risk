# Deployment Guide

## Step 1 — Push to GitHub

```bash
cd cibil_credit_risk

git init
git add .
git commit -m "Initial commit: CIBIL Credit Risk Predictor"

# Create a new repo on github.com named: cibil-credit-risk
# Then run:
git remote add origin https://github.com/YOUR_USERNAME/cibil-credit-risk.git
git branch -M main
git push -u origin main
```

---

## Step 2 — Deploy Backend to Render (free)

1. Go to https://render.com and sign up (free)
2. Click **New → Web Service**
3. Connect your GitHub repo
4. Set the following:
   - **Root directory:** `backend`
   - **Build command:** `pip install -r ../requirements.txt`
   - **Start command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Under **Environment Variables**, add:
   - `SECRET_KEY` → any long random string (e.g. `openssl rand -hex 32`)
6. Click **Create Web Service**

Wait ~3 minutes. Your API will be live at:
`https://cibil-credit-risk-api.onrender.com`

Test it: `https://cibil-credit-risk-api.onrender.com/`

---

## Step 3 — Deploy Frontend to Netlify (free)

1. Open `frontend/index.html`
2. Find this line near the top of the `<script>` tag:
   ```js
   const API = window.location.protocol === "file:" ? "http://localhost:8000" : "";
   ```
3. Change it to:
   ```js
   const API = "https://cibil-credit-risk-api.onrender.com";
   ```
4. Go to https://netlify.com → drag and drop your `frontend/` folder
5. Your app is live!

---

## Notes

- Render's free tier spins down after 15 minutes of inactivity — first request may take 30 seconds
- Upgrade to Render Starter ($7/mo) to keep it always-on
- For production, replace SQLite with PostgreSQL (Render provides free PostgreSQL)
- Set a strong `SECRET_KEY` — never use the default in production
