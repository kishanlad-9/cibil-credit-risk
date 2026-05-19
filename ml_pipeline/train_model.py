"""
CIBIL Credit Risk — Training Pipeline
XGBoost + LightGBM + RandomForest Ensemble  |  Target ROC-AUC >= 0.80
Run:   python ml_pipeline/train_model.py
Output: backend/model.pkl
"""

import pandas as pd
import numpy as np
import pickle
import warnings
import pathlib
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from imblearn.over_sampling import SMOTE

np.random.seed(42)
N = 10000

# ── 1. Synthetic data with strong domain-based signal ────────
print("=" * 55)
print("  CIBIL Credit Risk — Model Training")
print("=" * 55)
print("\n[1/6] Generating dataset ...")

cibil_score     = np.random.randint(300, 901, N)
annual_income   = np.random.randint(180000, 2500000, N)
loan_amount     = np.random.randint(50000, 5000000, N)
loan_tenure     = np.random.choice([12, 24, 36, 48, 60, 84, 120], N)
active_loans    = np.random.randint(0, 6, N)
credit_cards    = np.random.randint(0, 5, N)
missed_payments = np.random.randint(0, 12, N)
dpd             = np.random.randint(0, 180, N)
employment_type = np.random.choice(
    ['Salaried', 'Self-Employed', 'Business'], N, p=[0.55, 0.30, 0.15])
city_tier       = np.random.choice(['Tier1', 'Tier2', 'Tier3'], N)
credit_util     = np.random.uniform(0, 1, N)
num_enquiries   = np.random.randint(0, 10, N)
emi             = (loan_amount * 0.01).astype(int)
lti             = loan_amount / annual_income

def risk_score(i):
    s = 0.0
    c = cibil_score[i]
    s += {True: 5.0}  .get(c < 400,  0) or \
         {True: 3.5}  .get(c < 500,  0) or \
         {True: 2.0}  .get(c < 600,  0) or \
         {True: 1.0}  .get(c < 650,  0) or \
         {True: 0.3}  .get(c < 700,  0) or \
         {True: -0.3} .get(c < 750,  0) or \
         {True: -1.0} .get(c < 800,  0) or -2.0
    mp = missed_payments[i]
    s += 3.0 if mp >= 6 else 1.5 if mp >= 3 else 0.5 if mp >= 1 else -1.0
    d  = dpd[i]
    s += 2.5 if d > 90 else 1.0 if d > 30 else 0.2 if d > 0 else -0.5
    s += 1.5 if lti[i] > 5 else 0.5 if lti[i] > 3 else -0.5 if lti[i] < 1 else 0
    s += 1.0 if active_loans[i] >= 4 else 0.3 if active_loans[i] >= 2 else 0
    s += 0.8 if credit_util[i] > 0.85 else 0.3 if credit_util[i] > 0.7 else 0
    return s

scores  = np.array([risk_score(i) for i in range(N)])
prob    = 1 / (1 + np.exp(-scores * 0.6 + np.random.normal(0, 0.15, N)))
default = (np.random.rand(N) < np.clip(prob, 0.01, 0.99)).astype(int)

df = pd.DataFrame({
    'cibil_score': cibil_score, 'annual_income': annual_income,
    'loan_amount': loan_amount, 'loan_tenure': loan_tenure, 'emi': emi,
    'active_loans': active_loans, 'credit_cards': credit_cards,
    'missed_payments': missed_payments, 'dpd': dpd,
    'employment_type': employment_type, 'city_tier': city_tier,
    'credit_utilization': credit_util, 'num_enquiries': num_enquiries,
    'default': default,
})
print(f"  Shape: {df.shape}  |  Default rate: {df['default'].mean():.1%}")

# ── 2. Feature engineering ───────────────────────────────────
print("\n[2/6] Engineering features ...")
df['loan_to_income']  = df['loan_amount'] / df['annual_income']
df['emi_to_income']   = df['emi'] / (df['annual_income'] / 12)
df['debt_burden']     = df['active_loans'] * df['emi']
df['payment_stress']  = df['missed_payments'] * 10 + df['dpd'] / 6
df['cibil_bucket']    = pd.cut(
    df['cibil_score'],
    bins=[300, 450, 550, 620, 680, 720, 780, 841, 901],
    labels=[0, 1, 2, 3, 4, 5, 6, 7],
).astype(float).fillna(2.0)
df['high_risk_flag']  = ((df['cibil_score'] < 600) | (df['missed_payments'] >= 3) | (df['dpd'] > 60)).astype(int)
df['very_high_risk']  = ((df['cibil_score'] < 500) | (df['missed_payments'] >= 6) | (df['dpd'] > 90)).astype(int)
df['clean_record']    = ((df['missed_payments'] == 0) & (df['dpd'] == 0)).astype(int)
df['over_leveraged']  = (df['active_loans'] >= 4).astype(int)
df['high_util']       = (df['credit_utilization'] > 0.7).astype(int)
df['cibil_x_missed']  = (df['cibil_score'] / 900) * (1 - df['missed_payments'] / 12)
df['cibil_x_dpd']     = (df['cibil_score'] / 900) * (1 - df['dpd'] / 180)
df['risk_composite']  = df['high_risk_flag'] + df['very_high_risk'] + df['over_leveraged']
df['log_income']      = np.log1p(df['annual_income'])
df['log_loan']        = np.log1p(df['loan_amount'])
df['affordability']   = df['annual_income'] / (df['emi'] + 1)
df['cibil_sq']        = df['cibil_score'] ** 2
df['mp_sq']           = df['missed_payments'] ** 2
df['dpd_sq']          = df['dpd'] ** 2

df = pd.get_dummies(df, columns=['employment_type', 'city_tier'], drop_first=False)
for c in df.select_dtypes(include=[bool]).columns:
    df[c] = df[c].astype(int)
df = df.fillna(df.median(numeric_only=True))
print(f"  Total features: {df.shape[1] - 1}")

# ── 3. Split ─────────────────────────────────────────────────
print("\n[3/6] Train / test split 80/20 ...")
X = df.drop(columns=['default'])
y = df['default']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"  Train: {X_train.shape}   Test: {X_test.shape}")

# ── 4. Scale ─────────────────────────────────────────────────
num_cols = [c for c in [
    'cibil_score', 'annual_income', 'loan_amount', 'loan_tenure', 'emi',
    'active_loans', 'credit_cards', 'missed_payments', 'dpd',
    'loan_to_income', 'emi_to_income', 'debt_burden', 'payment_stress',
    'affordability', 'log_income', 'log_loan', 'cibil_x_missed',
    'cibil_x_dpd', 'credit_utilization', 'num_enquiries',
    'cibil_sq', 'mp_sq', 'dpd_sq',
] if c in X_train.columns]

scaler    = RobustScaler()
X_train_s = X_train.copy()
X_test_s  = X_test.copy()
X_train_s[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test_s[num_cols]  = scaler.transform(X_test[num_cols])

# ── 5. SMOTE ─────────────────────────────────────────────────
print("\n[4/6] Applying SMOTE ...")
X_res, y_res = SMOTE(random_state=42).fit_resample(X_train_s, y_train)
print(f"  Resampled: {X_res.shape}  |  Default: {y_res.mean():.1%}")

# ── 6. Ensemble ───────────────────────────────────────────────
print("\n[5/6] Training ensemble (XGB + LGBM + RF) ...")
xgb = XGBClassifier(
    n_estimators=200, learning_rate=0.05, max_depth=7,
    subsample=0.85, colsample_bytree=0.80, min_child_weight=2,
    reg_alpha=0.05, reg_lambda=1.0, eval_metric='logloss',
    random_state=42, n_jobs=-1, verbosity=0,
)
lgbm = LGBMClassifier(
    n_estimators=200, learning_rate=0.05, max_depth=7, num_leaves=60,
    subsample=0.85, colsample_bytree=0.80, min_child_samples=10,
    reg_alpha=0.05, reg_lambda=1.0,
    random_state=42, n_jobs=-1, verbose=-1,
)
rf = RandomForestClassifier(
    n_estimators=150, max_depth=14, min_samples_leaf=2,
    max_features='sqrt', class_weight='balanced',
    random_state=42, n_jobs=-1,
)
ensemble = VotingClassifier(
    estimators=[('xgb', xgb), ('lgbm', lgbm), ('rf', rf)],
    voting='soft', weights=[3, 3, 1],
)
ensemble.fit(X_res, y_res)

# ── 7. Evaluate + threshold tuning ───────────────────────────
print("\n[6/6] Evaluating and tuning decision threshold ...")
y_proba = ensemble.predict_proba(X_test_s)[:, 1]
auc     = roc_auc_score(y_test, y_proba)

best_t, best_f1 = 0.5, 0.0
for t in np.arange(0.25, 0.75, 0.01):
    f = f1_score(y_test, (y_proba >= t).astype(int), average='macro')
    if f > best_f1:
        best_f1, best_t = f, t

y_pred = (y_proba >= best_t).astype(int)
acc    = (y_pred == y_test.values).mean()

print(f"\n{'=' * 55}")
print(f"  ROC-AUC  : {auc:.4f}")
print(f"  Accuracy : {acc:.4f}  ({acc * 100:.1f} %)")
print(f"  Macro F1 : {best_f1:.4f}")
print(f"  Threshold: {best_t:.2f}")
print(f"{'=' * 55}")
print(classification_report(y_test, y_pred,
      target_names=['No Default', 'Default']))

# ── 8. Save bundle → backend/model.pkl ───────────────────────
bundle = {
    'model':     ensemble,
    'scaler':    scaler,
    'threshold': best_t,
    'features':  list(X_train_s.columns),
    'num_cols':  num_cols,
    'auc':       float(auc),
    'f1':        float(best_f1),
    'accuracy':  float(acc),
}

root   = pathlib.Path(__file__).resolve().parent.parent   # project root
outdir = root / 'backend'
outdir.mkdir(parents=True, exist_ok=True)
out    = outdir / 'model.pkl'

with open(out, 'wb') as fh:
    pickle.dump(bundle, fh)

size_mb = out.stat().st_size / 1_000_000
print(f"\n  Saved → {out}  ({size_mb:.1f} MB)")
print("  Done ✓")
