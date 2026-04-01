"""
Credit Card Fraud Detection - ML Pipeline
Covers: Data loading, EDA, SMOTE oversampling, normalization,
        feature importance, Random Forest + Logistic Regression models
"""

import pandas as pd
import numpy as np
import json
import joblib
import warnings
import os
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, precision_recall_curve,
                              average_precision_score, roc_curve)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_FILES = [
    '/mnt/user-data/uploads/chunk_0.csv',
    '/mnt/user-data/uploads/chunk_1.csv',
    '/mnt/user-data/uploads/chunk_4.csv',
]
OUT_DIR   = '/home/claude/fraud-detection/static'
MODEL_DIR = '/home/claude/fraud-detection/models'
os.makedirs(f'{OUT_DIR}/charts', exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

PALETTE = {'bg': '#0a0f1e', 'card': '#111827', 'accent': '#00d4ff',
           'danger': '#ff4d6d', 'safe': '#00e5a0', 'text': '#e2e8f0'}

plt.rcParams.update({
    'figure.facecolor': PALETTE['bg'],
    'axes.facecolor':   PALETTE['card'],
    'axes.edgecolor':   '#2d3748',
    'axes.labelcolor':  PALETTE['text'],
    'xtick.color':      PALETTE['text'],
    'ytick.color':      PALETTE['text'],
    'text.color':       PALETTE['text'],
    'grid.color':       '#2d3748',
    'grid.linestyle':   '--',
    'grid.alpha':       0.5,
    'font.family':      'monospace',
})

# ─── 1. Load Data ─────────────────────────────────────────────────────────────
print("📦 Loading data...")
df = pd.concat([pd.read_csv(f) for f in DATA_FILES], ignore_index=True)
df.drop_duplicates(inplace=True)
print(f"   Shape: {df.shape}  |  Fraud: {df['Class'].sum()}  |  Legit: {(df['Class']==0).sum()}")

# ─── 2. EDA ───────────────────────────────────────────────────────────────────
print("📊 Generating EDA charts...")

## 2a. Class distribution (pie)
fig, ax = plt.subplots(figsize=(6, 6), facecolor=PALETTE['bg'])
counts = df['Class'].value_counts()
wedges, texts, autotexts = ax.pie(
    counts, labels=['Legitimate', 'Fraudulent'],
    autopct='%1.2f%%', startangle=140,
    colors=[PALETTE['safe'], PALETTE['danger']],
    wedgeprops=dict(width=0.55, edgecolor=PALETTE['bg'], linewidth=3),
    textprops={'color': PALETTE['text'], 'fontsize': 13})
for at in autotexts:
    at.set_fontsize(12); at.set_color('white')
ax.set_title('Transaction Class Distribution', fontsize=16, pad=20, color=PALETTE['accent'])
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/charts/class_distribution.png', dpi=150, bbox_inches='tight',
            facecolor=PALETTE['bg'])
plt.close()

## 2b. Amount distribution histogram
fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=PALETTE['bg'])
for ax, cls, col, label in zip(axes, [0, 1],
                                [PALETTE['safe'], PALETTE['danger']],
                                ['Legitimate Transactions', 'Fraudulent Transactions']):
    data = df[df['Class'] == cls]['Amount']
    ax.hist(data, bins=50, color=col, alpha=0.85, edgecolor='none')
    ax.set_title(label, fontsize=13, color=col, pad=10)
    ax.set_xlabel('Amount (USD)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    ax.text(0.97, 0.95, f'Mean: ${data.mean():.2f}\nMax: ${data.max():.2f}',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=10, color=PALETTE['text'],
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#1e293b', alpha=0.8))
fig.suptitle('Transaction Amount Distribution', fontsize=15, color=PALETTE['accent'], y=1.01)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/charts/amount_distribution.png', dpi=150, bbox_inches='tight',
            facecolor=PALETTE['bg'])
plt.close()

## 2c. Correlation heatmap (top features)
fig, ax = plt.subplots(figsize=(14, 11), facecolor=PALETTE['bg'])
corr = df.drop(columns=['Time']).corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, ax=ax,
            cmap=sns.diverging_palette(220, 10, as_cmap=True),
            center=0, linewidths=0.3, linecolor='#1e293b',
            annot=False, square=True,
            cbar_kws={'shrink': 0.7, 'label': 'Correlation'})
ax.set_title('Feature Correlation Matrix', fontsize=15, color=PALETTE['accent'], pad=15)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/charts/correlation_heatmap.png', dpi=150, bbox_inches='tight',
            facecolor=PALETTE['bg'])
plt.close()

## 2d. Boxplots for top correlated features with Class
corr_with_class = df.corr()['Class'].abs().drop('Class').nlargest(6).index.tolist()
fig, axes = plt.subplots(2, 3, figsize=(14, 8), facecolor=PALETTE['bg'])
for ax, feat in zip(axes.flatten(), corr_with_class):
    data0 = df[df['Class']==0][feat]
    data1 = df[df['Class']==1][feat]
    bp = ax.boxplot([data0, data1], patch_artist=True,
                    medianprops=dict(color='white', linewidth=2),
                    whiskerprops=dict(color=PALETTE['text']),
                    capprops=dict(color=PALETTE['text']),
                    flierprops=dict(marker='o', color=PALETTE['danger'],
                                   markersize=2, alpha=0.4))
    bp['boxes'][0].set_facecolor(PALETTE['safe'])
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor(PALETTE['danger'])
    bp['boxes'][1].set_alpha(0.7)
    ax.set_xticklabels(['Legit', 'Fraud'])
    ax.set_title(feat, fontsize=11, color=PALETTE['accent'])
    ax.yaxis.grid(True); ax.set_axisbelow(True)
fig.suptitle('Top Features: Legit vs Fraud Distribution', fontsize=14,
             color=PALETTE['accent'], y=1.01)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/charts/feature_boxplots.png', dpi=150, bbox_inches='tight',
            facecolor=PALETTE['bg'])
plt.close()
print("   EDA charts saved.")

# ─── 3. Preprocessing ─────────────────────────────────────────────────────────
print("⚙️  Preprocessing...")
features = [c for c in df.columns if c not in ['Class', 'Time']]
X = df[features].copy()
y = df['Class'].copy()

## Normalize Amount
scaler_amount = StandardScaler()
X['Amount'] = scaler_amount.fit_transform(X[['Amount']])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

## SMOTE (manual implementation — no imblearn needed)
print("   Applying SMOTE oversampling...")
def manual_smote(X_min, n_synthetic, k=5, seed=42):
    rng = np.random.RandomState(seed)
    Xm = X_min.values
    synthetic = []
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k+1).fit(Xm)
    dists, idxs = nn.kneighbors(Xm)
    for _ in range(n_synthetic):
        i = rng.randint(0, len(Xm))
        nn_idx = idxs[i, 1:][rng.randint(0, k)]
        diff = Xm[nn_idx] - Xm[i]
        synthetic.append(Xm[i] + rng.random() * diff)
    return pd.DataFrame(synthetic, columns=X_min.columns)

fraud_train = X_train[y_train == 1]
legit_train = X_train[y_train == 0]
n_needed = len(legit_train) - len(fraud_train)
synthetic = manual_smote(fraud_train, n_needed)
X_train_bal = pd.concat([X_train, synthetic], ignore_index=True)
y_train_bal = pd.concat([y_train, pd.Series([1]*len(synthetic))], ignore_index=True)
print(f"   After SMOTE — Fraud: {y_train_bal.sum()}  Legit: {(y_train_bal==0).sum()}")

## Scale all features
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_bal)
X_test_sc  = scaler.transform(X_test)

# ─── 4. Train Models ──────────────────────────────────────────────────────────
print("🤖 Training models...")

## Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=12,
                             class_weight='balanced', random_state=42, n_jobs=-1)
rf.fit(X_train_sc, y_train_bal)

## Logistic Regression
lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr.fit(X_train_sc, y_train_bal)

# ─── 5. Evaluate ──────────────────────────────────────────────────────────────
print("📈 Evaluating...")

def evaluate(model, name, color):
    y_pred  = model.predict(X_test_sc)
    y_proba = model.predict_proba(X_test_sc)[:, 1]
    auc_roc = roc_auc_score(y_test, y_proba)
    auc_pr  = average_precision_score(y_test, y_proba)
    report  = classification_report(y_test, y_pred, output_dict=True)
    cm      = confusion_matrix(y_test, y_pred)
    print(f"   {name}: AUC-ROC={auc_roc:.4f}  AUC-PR={auc_pr:.4f}")
    return {'auc_roc': auc_roc, 'auc_pr': auc_pr, 'report': report,
            'cm': cm, 'y_proba': y_proba, 'y_pred': y_pred, 'color': color}

rf_eval = evaluate(rf, 'Random Forest', PALETTE['accent'])
lr_eval = evaluate(lr, 'Logistic Regression', '#a78bfa')

## 5a. Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=PALETTE['bg'])
for ax, ev, name in zip(axes,
                         [rf_eval, lr_eval],
                         ['Random Forest', 'Logistic Regression']):
    cm_pct = ev['cm'].astype(float) / ev['cm'].sum(axis=1, keepdims=True)
    sns.heatmap(cm_pct, annot=ev['cm'], fmt='d', ax=ax,
                cmap='Blues', linewidths=2, linecolor=PALETTE['bg'],
                xticklabels=['Legit','Fraud'], yticklabels=['Legit','Fraud'],
                annot_kws={'size': 14, 'color': 'white'})
    ax.set_title(f'{name}\nAUC-ROC: {ev["auc_roc"]:.4f}', fontsize=12,
                 color=ev['color'], pad=12)
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('Actual',    fontsize=11)
fig.suptitle('Confusion Matrices', fontsize=15, color=PALETTE['accent'])
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/charts/confusion_matrices.png', dpi=150, bbox_inches='tight',
            facecolor=PALETTE['bg'])
plt.close()

## 5b. ROC curves
fig, ax = plt.subplots(figsize=(8, 6), facecolor=PALETTE['bg'])
for ev, name in [(rf_eval, 'Random Forest'), (lr_eval, 'Logistic Regression')]:
    fpr, tpr, _ = roc_curve(y_test, ev['y_proba'])
    ax.plot(fpr, tpr, color=ev['color'], lw=2,
            label=f'{name} (AUC={ev["auc_roc"]:.3f})')
ax.plot([0,1],[0,1], 'w--', lw=1, alpha=0.4, label='Random Classifier')
ax.fill_between([0,1],[0,1], alpha=0.03, color='white')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curve Comparison', fontsize=14, color=PALETTE['accent'], pad=12)
ax.legend(fontsize=11, facecolor='#1e293b', edgecolor='#2d3748')
ax.yaxis.grid(True); ax.xaxis.grid(True); ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/charts/roc_curves.png', dpi=150, bbox_inches='tight',
            facecolor=PALETTE['bg'])
plt.close()

## 5c. Precision-Recall curves
fig, ax = plt.subplots(figsize=(8, 6), facecolor=PALETTE['bg'])
for ev, name in [(rf_eval, 'Random Forest'), (lr_eval, 'Logistic Regression')]:
    prec, rec, _ = precision_recall_curve(y_test, ev['y_proba'])
    ax.plot(rec, prec, color=ev['color'], lw=2,
            label=f'{name} (AP={ev["auc_pr"]:.3f})')
baseline = y_test.mean()
ax.axhline(baseline, color='white', lw=1, ls='--', alpha=0.4,
           label=f'Baseline ({baseline:.3f})')
ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Precision-Recall Curve', fontsize=14, color=PALETTE['accent'], pad=12)
ax.legend(fontsize=11, facecolor='#1e293b', edgecolor='#2d3748')
ax.yaxis.grid(True); ax.xaxis.grid(True); ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/charts/precision_recall.png', dpi=150, bbox_inches='tight',
            facecolor=PALETTE['bg'])
plt.close()

## 5d. Feature importance
importances = pd.Series(rf.feature_importances_, index=features).nlargest(15)
fig, ax = plt.subplots(figsize=(10, 7), facecolor=PALETTE['bg'])
colors = [PALETTE['accent'] if i < 5 else '#4a9eff'
          if i < 10 else '#6b7280' for i in range(len(importances))]
bars = ax.barh(importances.index[::-1], importances.values[::-1],
               color=colors[::-1], edgecolor='none', height=0.65)
ax.set_xlabel('Feature Importance Score', fontsize=12)
ax.set_title('Top 15 Feature Importances (Random Forest)', fontsize=14,
             color=PALETTE['accent'], pad=12)
ax.xaxis.grid(True); ax.set_axisbelow(True)
for bar, val in zip(bars, importances.values[::-1]):
    ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=9, color=PALETTE['text'])
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/charts/feature_importance.png', dpi=150, bbox_inches='tight',
            facecolor=PALETTE['bg'])
plt.close()

print("   All evaluation charts saved.")

# ─── 6. Save models & metadata ────────────────────────────────────────────────
print("💾 Saving models...")
joblib.dump(rf,     f'{MODEL_DIR}/random_forest.pkl')
joblib.dump(lr,     f'{MODEL_DIR}/logistic_regression.pkl')
joblib.dump(scaler, f'{MODEL_DIR}/scaler.pkl')

# Save stats for the website
rf_report  = rf_eval['report']
lr_report  = lr_eval['report']

stats = {
    'dataset': {
        'total_rows':   int(len(df)),
        'fraud_count':  int(df['Class'].sum()),
        'legit_count':  int((df['Class']==0).sum()),
        'fraud_pct':    round(df['Class'].mean()*100, 2),
        'features':     len(features),
        'amount_mean':  round(df['Amount'].mean(), 2),
        'amount_max':   round(df['Amount'].max(), 2),
    },
    'random_forest': {
        'auc_roc':   round(rf_eval['auc_roc'], 4),
        'auc_pr':    round(rf_eval['auc_pr'], 4),
        'precision': round(rf_report['1']['precision'], 4),
        'recall':    round(rf_report['1']['recall'], 4),
        'f1':        round(rf_report['1']['f1-score'], 4),
        'accuracy':  round(rf_report['accuracy'], 4),
    },
    'logistic_regression': {
        'auc_roc':   round(lr_eval['auc_roc'], 4),
        'auc_pr':    round(lr_eval['auc_pr'], 4),
        'precision': round(lr_report['1']['precision'], 4),
        'recall':    round(lr_report['1']['recall'], 4),
        'f1':        round(lr_report['1']['f1-score'], 4),
        'accuracy':  round(lr_report['accuracy'], 4),
    },
    'top_features': importances.head(10).to_dict(),
    'smote': {
        'original_fraud': int(y_train.sum()),
        'after_smote':    int(y_train_bal.sum()),
    }
}

with open(f'{OUT_DIR}/stats.json', 'w') as f:
    json.dump(stats, f, indent=2)

print("✅ Training complete! All files saved.")
print(json.dumps({k: v for k,v in stats.items() if k != 'top_features'}, indent=2))
