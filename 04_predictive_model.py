

"""Logistic Regression Student Loss Risk Prediction Model

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, accuracy_score)
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
```

---
print("\n\n" + "=" * 60)
print("Module 04: Logistic Regression - Student Churn Prediction")
print("=" * 60)

print("\n[1/5] Loading data...")
df = pd.read_csv('student_annual_metrics.csv')
print(f"  Total students: {len(df)}")
print(f"  Withdrawn: {df['is_withdraw'].sum()} ({df['is_withdraw'].mean()*100:.1f}%)")

print("\n[2/5] Preparing features...")

feature_cols = [
    'core_gpa',
    'hw_average',
    'cw_average',
    'core_assessments',
    'behavior_points',
    'attendance_rate',
    'suspension_count',
    'risk_score',
    'gpa_change_t1_t3'
]

df_model = df[feature_cols + ['is_withdraw']].dropna()
X = df_model[feature_cols]
y = df_model['is_withdraw']

print(f"  Features: {len(feature_cols)}")
print(f"  Samples: {len(X)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"  Train set: {len(X_train)} ({y_train.sum()} withdrawn)")
print(f"  Test set: {len(X_test)} ({y_test.sum()} withdrawn)")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n[3/5] Training Logistic Regression...")

model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
model.fit(X_train_scaled, y_train)

y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)
y_test_proba = model.predict_proba(X_test_scaled)[:, 1]

print("  Training complete")

print("\n[4/5] Evaluating model...")

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
test_auc = roc_auc_score(y_test, y_test_proba)

print(f"\n  Accuracy (Train): {train_acc:.4f}")
print(f"  Accuracy (Test):  {test_acc:.4f}")
print(f"  ROC-AUC (Test):   {test_auc:.4f}")

print("\n  Classification Report (Test Set):")
print(classification_report(y_test, y_test_pred,
                           target_names=['Retained', 'Withdrawn'],
                           digits=3))

cm = confusion_matrix(y_test, y_test_pred)
print(f"  Confusion Matrix:")
print(f"    TN: {cm[0,0]:3d}  FP: {cm[0,1]:3d}")
print(f"    FN: {cm[1,0]:3d}  TP: {cm[1,1]:3d}")

print("\n[5/5] Generating visualizations...")

coefficients = pd.DataFrame({
    'Feature': feature_cols,
    'Coefficient': model.coef_[0],
    'Abs_Coefficient': np.abs(model.coef_[0])
}).sort_values('Abs_Coefficient', ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#e74c3c' if c < 0 else '#3498db' for c in coefficients['Coefficient']]
bars = ax.barh(coefficients['Feature'], coefficients['Coefficient'],
               color=colors, alpha=0.85, edgecolor='black', linewidth=1.2)
ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax.set_xlabel('Coefficient (Log-Odds)', fontsize=12, fontweight='bold')
ax.set_title('Feature Importance for Student Churn Prediction',
             fontsize=13, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

for bar, val in zip(bars, coefficients['Coefficient']):
    x_pos = val + (0.1 if val > 0 else -0.1)
    ha = 'left' if val > 0 else 'right'
    ax.text(x_pos, bar.get_y() + bar.get_height()/2, f'{val:.2f}',
            va='center', ha=ha, fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)

fig, ax = plt.subplots(figsize=(8, 7))
ax.plot(fpr, tpr, color='#3498db', linewidth=2.5,
        label=f'ROC Curve (AUC = {test_auc:.3f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier')
ax.fill_between(fpr, tpr, alpha=0.2, color='#3498db')
ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
ax.set_title('ROC Curve - Student Churn Prediction',
             fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted: Retained', 'Predicted: Withdrawn'],
            yticklabels=['Actual: Retained', 'Actual: Withdrawn'],
            linewidths=2, linecolor='black', ax=ax)
ax.set_title('Confusion Matrix', fontsize=13, fontweight='bold', pad=15)

for i in range(2):
    for j in range(2):
        text = ax.texts[i*2 + j]
        text.set_fontsize(16)
        text.set_fontweight('bold')

plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n" + "=" * 60)
print("Module 04 Complete - Results Summary")
print("=" * 60)

coefficients.to_csv('feature_importance.csv', index=False)

results_summary = pd.DataFrame({
    'Metric': ['Accuracy (Train)', 'Accuracy (Test)', 'ROC-AUC (Test)',
               'Precision (Withdrawn)', 'Recall (Withdrawn)', 'F1-Score (Withdrawn)'],
    'Value': [
        train_acc,
        test_acc,
        test_auc,
        cm[1,1] / (cm[0,1] + cm[1,1]) if (cm[0,1] + cm[1,1]) > 0 else 0,
        cm[1,1] / (cm[1,0] + cm[1,1]) if (cm[1,0] + cm[1,1]) > 0 else 0,
        2 * (cm[1,1] / (cm[0,1] + cm[1,1])) * (cm[1,1] / (cm[1,0] + cm[1,1])) /
        ((cm[1,1] / (cm[0,1] + cm[1,1])) + (cm[1,1] / (cm[1,0] + cm[1,1])))
        if (cm[0,1] + cm[1,1]) > 0 and (cm[1,0] + cm[1,1]) > 0 else 0
    ]
})
results_summary.to_csv('model_performance.csv', index=False)

print("\n【Model Performance】")
print(results_summary.to_string(index=False))

print("\n【Top 5 Churn Predictors】")
print(coefficients.head(5)[['Feature', 'Coefficient']].to_string(index=False))

print("\n【Output Files】")
print("  - feature_importance.csv")
print("  - model_performance.csv")
print("  - feature_importance.png")
print("  - roc_curve.png")
print("  - confusion_matrix.png")

print("\n" + "=" * 60)
print("✓ All modules complete!")
print("=" * 60)