# GitHub Repo Setup — Step by Step

## 1. Create the repo on GitHub

1. Go to https://github.com/new
2. Repository name: `cibil-credit-risk`
3. Description: `🏦 ML-powered loan default predictor using CIBIL-style Indian credit data | XGBoost + LightGBM + RF | FastAPI + React | 0.80 AUC`
4. Set to **Public** (so recruiters can see it)
5. Do NOT check "Add README" — we already have one
6. Click **Create repository**

---

## 2. Push your code

Open terminal inside your `cibil_credit_risk` folder and run:

```bash
git init
git add .
git commit -m "feat: initial commit — CIBIL Credit Risk Predictor

- XGBoost + LightGBM + RF ensemble (AUC 0.80)
- FastAPI backend with JWT auth
- React SPA frontend
- SMOTE balancing, 36 engineered features
- Render + Netlify deployment config"

git remote add origin https://github.com/YOUR_USERNAME/cibil-credit-risk.git
git branch -M main
git push -u origin main
```

---

## 3. Add topics to your repo (helps with discovery)

On your GitHub repo page, click the ⚙️ next to "About" and add these topics:
```
machine-learning  credit-risk  xgboost  fastapi  python  finance  india  cibil  mlops
```

---

## 4. Pin it on your profile

1. Go to your GitHub profile
2. Click "Customize your pins"
3. Select `cibil-credit-risk`

---

## 5. Add a screenshot

Take a screenshot of the running app and save it as `screenshot.png` in the root folder.
Then add this line to your README.md right after the badges:

```markdown
![App Screenshot](screenshot.png)
```

Commit and push.

---

## Pro tips for recruiter visibility

- Star your own repo (it shows activity)
- Write a commit message history that tells a story — use feat:, fix:, docs: prefixes
- Add a LICENSE file (MIT) — shows professionalism
- Keep your README polished — it's the first thing they read
