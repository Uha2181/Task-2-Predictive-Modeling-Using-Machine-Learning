# ============================================================
# Task 2: Predictive Modeling Using Machine Learning
# Dataset: Titanic (via seaborn)
# Models: Logistic Regression, Decision Tree, Random Forest
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_curve, auc)
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# STEP 1: LOAD & CLEAN DATA
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Load & Preprocess Data")
print("=" * 60)

df = sns.load_dataset('titanic')

# Drop high-missing and irrelevant columns
df.drop(columns=['deck', 'embark_town', 'alive', 'who', 'adult_male', 'alone', 'class'], inplace=True)

# Fill missing values
df["age"].fillna(df["age"].median(), inplace=True)
# Also fill any remaining NaN after encoding
df.dropna(inplace=True)
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)

# Encode categorical columns
le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])           # male=1, female=0
df['embarked'] = le.fit_transform(df['embarked']) # C=0, Q=1, S=2

df.dropna(inplace=True)
print(f"Dataset shape: {df.shape}")
print(f"Columns used: {list(df.columns)}")
print(f"\nClass distribution:\n{df['survived'].value_counts()}")

# ─────────────────────────────────────────────
# STEP 2: FEATURE SELECTION & SPLITTING
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Feature Selection & Train/Test Split")
print("=" * 60)

X = df.drop('survived', axis=1)
y = df['survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set : {X_test.shape[0]} samples")
print(f"Features    : {list(X.columns)}")

# ─────────────────────────────────────────────
# STEP 3: TRAIN MODELS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Training Models")
print("=" * 60)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree':       DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
}

results = {}
for name, model in models.items():
    if name == 'Logistic Regression':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    cv  = cross_val_score(model,
                          X_train_scaled if name == 'Logistic Regression' else X_train,
                          y_train, cv=5, scoring='accuracy').mean()
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc     = auc(fpr, tpr)
    cm          = confusion_matrix(y_test, y_pred)
    report      = classification_report(y_test, y_pred, output_dict=True)

    results[name] = {
        'model': model, 'y_pred': y_pred, 'y_prob': y_prob,
        'accuracy': acc, 'cv_score': cv, 'auc': roc_auc,
        'fpr': fpr, 'tpr': tpr, 'cm': cm, 'report': report
    }
    print(f"\n✔ {name}")
    print(f"   Accuracy  : {acc*100:.2f}%")
    print(f"   CV Score  : {cv*100:.2f}%")
    print(f"   ROC-AUC   : {roc_auc:.4f}")

# ─────────────────────────────────────────────
# STEP 4: BEST MODEL DETAILS
# ─────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]['auc'])
print(f"\n🏆 Best Model: {best_name} (AUC = {results[best_name]['auc']:.4f})")
print(f"\nClassification Report — {best_name}:")
print(classification_report(y_test, results[best_name]['y_pred'],
                             target_names=['Did Not Survive', 'Survived']))

# ─────────────────────────────────────────────
# STEP 5: VISUALIZATIONS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Generating Visualizations")
print("=" * 60)

sns.set_theme(style='whitegrid')
plt.rcParams['figure.facecolor'] = '#f8f9fa'
COLORS = ['#3498db', '#e74c3c', '#2ecc71']

fig = plt.figure(figsize=(18, 14))
fig.suptitle('Predictive Modeling — Machine Learning Dashboard',
             fontsize=20, fontweight='bold', y=0.98, color='#2c3e50')
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── Plot 1: Model Accuracy Comparison ──
ax1 = fig.add_subplot(gs[0, 0])
names = list(results.keys())
accs  = [results[n]['accuracy']*100 for n in names]
short = ['Log. Reg.', 'Dec. Tree', 'Rnd. Forest']
bars  = ax1.bar(short, accs, color=COLORS, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, accs):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
             f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
ax1.set_title('Model Accuracy Comparison', fontweight='bold', fontsize=12)
ax1.set_ylabel('Accuracy (%)')
ax1.set_ylim(0, 105)

# ── Plot 2: Cross-Validation Scores ──
ax2 = fig.add_subplot(gs[0, 1])
cvs = [results[n]['cv_score']*100 for n in names]
bars2 = ax2.bar(short, cvs, color=COLORS, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars2, cvs):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
             f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
ax2.set_title('Cross-Validation Scores (5-Fold)', fontweight='bold', fontsize=12)
ax2.set_ylabel('CV Accuracy (%)')
ax2.set_ylim(0, 105)

# ── Plot 3: ROC-AUC Comparison ──
ax3 = fig.add_subplot(gs[0, 2])
aucs = [results[n]['auc'] for n in names]
bars3 = ax3.bar(short, aucs, color=COLORS, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars3, aucs):
    ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
             f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
ax3.set_title('ROC-AUC Score Comparison', fontweight='bold', fontsize=12)
ax3.set_ylabel('AUC Score')
ax3.set_ylim(0, 1.1)
ax3.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Random')

# ── Plots 4-6: Confusion Matrices ──
for i, (name, short_name, color) in enumerate(zip(names, short, COLORS)):
    ax = fig.add_subplot(gs[1, i])
    cm = results[name]['cm']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                linewidths=0.5, cbar=False,
                xticklabels=['Not Survived', 'Survived'],
                yticklabels=['Not Survived', 'Survived'])
    ax.set_title(f'Confusion Matrix\n{short_name}', fontweight='bold', fontsize=11)
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')

# ── Plot 7: ROC Curves (all 3 models) ──
ax7 = fig.add_subplot(gs[2, 0:2])
for name, color in zip(names, COLORS):
    r = results[name]
    ax7.plot(r['fpr'], r['tpr'], color=color, linewidth=2.5,
             label=f"{name} (AUC = {r['auc']:.3f})")
ax7.plot([0,1],[0,1], 'k--', linewidth=1.5, label='Random Classifier')
ax7.fill_between(results[best_name]['fpr'], results[best_name]['tpr'],
                 alpha=0.1, color='#2ecc71')
ax7.set_title('ROC Curves — All Models', fontweight='bold', fontsize=12)
ax7.set_xlabel('False Positive Rate')
ax7.set_ylabel('True Positive Rate')
ax7.legend(loc='lower right')

# ── Plot 8: Feature Importance (Random Forest) ──
ax8 = fig.add_subplot(gs[2, 2])
rf_model   = results['Random Forest']['model']
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
importances.sort_values().plot(kind='barh', ax=ax8, color='#2ecc71',
                                edgecolor='white', linewidth=1)
ax8.set_title('Feature Importance\n(Random Forest)', fontweight='bold', fontsize=12)
ax8.set_xlabel('Importance Score')

plt.savefig('/mnt/user-data/outputs/ml_dashboard.png', dpi=150,
            bbox_inches='tight', facecolor='#f8f9fa')
print("✔ Saved: ml_dashboard.png")
plt.close()

# ─────────────────────────────────────────────
# STEP 6: FINAL SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Final Summary")
print("=" * 60)
print(f"\n{'Model':<22} {'Accuracy':>10} {'CV Score':>10} {'AUC':>8}")
print("-" * 54)
for name in names:
    r = results[name]
    marker = " 🏆" if name == best_name else ""
    print(f"{name:<22} {r['accuracy']*100:>9.2f}% {r['cv_score']*100:>9.2f}% {r['auc']:>8.4f}{marker}")

print(f"\n🏆 Best Model: {best_name}")
print(f"   → Highest ROC-AUC of {results[best_name]['auc']:.4f}")
print(f"   → Accuracy: {results[best_name]['accuracy']*100:.2f}%")
print("\n✅ ALL TASKS COMPLETE!")
