"""GPA Regression Analysis"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, accuracy_score)
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
# ========================================================================
# PART 3: GPA REGRESSION ANALYSIS
# ========================================================================

print("=" * 60)
print("Module 03: Regression Analysis - Term-by-Term Analysis")
print("=" * 60)

print("\n[1/5] Loading data...")
df_annual = pd.read_csv('student_annual_metrics.csv')
df_full = pd.read_csv('cleaned_student_data.csv')
print(f"✓ Annual data: {len(df_annual)} students")
print(f"✓ Full data: {len(df_full)} records")

print("\n[2/5] Correlation analysis...")

key_vars = ['hw_average', 'cw_average', 'core_assessments',
            'behavior_points', 'attendance_rate', 'core_gpa']

correlation_matrix = df_annual[key_vars].corr()
gpa_correlations = correlation_matrix['core_gpa'].drop('core_gpa').sort_values(ascending=False)

print("\nCorrelations with Final GPA (high to low):")
for var, corr in gpa_correlations.items():
    stars = "***" if abs(corr) > 0.7 else ("**" if abs(corr) > 0.5 else "*")
    print(f"  {var:25s}: {corr:6.3f} {stars}")

fig, ax = plt.subplots(figsize=(9, 7))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f',
            cmap='RdYlGn', center=0, vmin=-1, vmax=1,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
ax.set_title('Correlation Matrix: Academic & Behavioral Factors',
             fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("\n✓ Correlation heatmap: correlation_heatmap.png")
plt.close()

print("\n[3/5] Multiple linear regression (1): Predicting Final GPA...")

X_vars = ['hw_average', 'cw_average', 'core_assessments',
          'behavior_points', 'attendance_rate']
y_var = 'core_gpa'

df_reg = df_annual[X_vars + [y_var]].dropna()
X = df_reg[X_vars]
y = df_reg[y_var]

print(f"  Sample size: {len(df_reg)}")

scaler_gpa = StandardScaler()
X_scaled = scaler_gpa.fit_transform(X)

model_gpa = LinearRegression()
model_gpa.fit(X_scaled, y)

r_squared_gpa = model_gpa.score(X_scaled, y)
rmse_gpa = np.sqrt(np.mean((y - model_gpa.predict(X_scaled)) ** 2))

print(f"\n  Model performance:")
print(f"    R² = {r_squared_gpa:.4f} (explains {r_squared_gpa*100:.1f}% of GPA variance)")
print(f"    RMSE = {rmse_gpa:.2f} points")

coefficients_gpa = pd.DataFrame({
    'Feature': X_vars,
    'Coefficient': model_gpa.coef_,
    'Abs_Coefficient': np.abs(model_gpa.coef_)
}).sort_values('Abs_Coefficient', ascending=False)

print(f"\n  Standardized regression coefficients:")
for _, row in coefficients_gpa.iterrows():
    print(f"    {row['Feature']:25s}: {row['Coefficient']:7.4f}")

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#e74c3c' if c < 0 else '#3498db' for c in coefficients_gpa['Coefficient']]
bars = ax.barh(coefficients_gpa['Feature'], coefficients_gpa['Coefficient'],
               color=colors, alpha=0.85, edgecolor='black', linewidth=1.2)
ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax.set_xlabel('Standardized Coefficient', fontsize=13, fontweight='bold')
ax.set_title(f'Factors Driving Final GPA (R² = {r_squared_gpa:.3f})',
             fontsize=14, fontweight='bold', pad=15)
ax.grid(axis='x', alpha=0.3)

for bar, val in zip(bars, coefficients_gpa['Coefficient']):
    x_pos = val + (0.02 if val > 0 else -0.02)
    ha = 'left' if val > 0 else 'right'
    ax.text(x_pos, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
            va='center', ha=ha, fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('regression_final_gpa.png', dpi=300, bbox_inches='tight')
print("\n✓ Final GPA regression chart: regression_final_gpa.png")
plt.close()

print("\n[4/5] Multiple linear regression (2): T1 data predicting T1→T2 growth...")

df_t1 = df_full[df_full['term'] == 'T1'][[
    'student_id', 'hw_average', 'cw_average', 'core_assessments',
    'behavior_points', 'attendance_rate'
]].rename(columns={
    'hw_average': 'hw_t1',
    'cw_average': 'cw_t1',
    'core_assessments': 'assessment_t1',
    'behavior_points': 'behavior_t1',
    'attendance_rate': 'attendance_t1'
})

df_growth = df_annual[['student_id', 'gpa_change_t1_t2', 'gpa_change_t2_t3']]
df_t1_merged = df_t1.merge(df_growth, on='student_id', how='inner')

X_t1 = df_t1_merged[['hw_t1', 'cw_t1', 'assessment_t1', 'behavior_t1', 'attendance_t1']].dropna()
y_t1_t2 = df_t1_merged.loc[X_t1.index, 'gpa_change_t1_t2']

print(f"  Sample size: {len(X_t1)}")

scaler_t1 = StandardScaler()
X_t1_scaled = scaler_t1.fit_transform(X_t1)

model_t1_t2 = LinearRegression()
model_t1_t2.fit(X_t1_scaled, y_t1_t2)

r2_t1_t2 = model_t1_t2.score(X_t1_scaled, y_t1_t2)
rmse_t1_t2 = np.sqrt(np.mean((y_t1_t2 - model_t1_t2.predict(X_t1_scaled)) ** 2))

print(f"\n  Model performance:")
print(f"    R² = {r2_t1_t2:.4f}")
print(f"    RMSE = {rmse_t1_t2:.2f} points")

coef_t1_t2 = pd.DataFrame({
    'Feature': ['Homework', 'Classwork', 'Assessment', 'Behavior', 'Attendance'],
    'Coefficient': model_t1_t2.coef_,
    'Abs_Coefficient': np.abs(model_t1_t2.coef_)
}).sort_values('Abs_Coefficient', ascending=False)

print(f"\n  Standardized regression coefficients:")
for _, row in coef_t1_t2.iterrows():
    print(f"    {row['Feature']:15s}: {row['Coefficient']:7.4f}")

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#e74c3c' if c < 0 else '#2ecc71' for c in coef_t1_t2['Coefficient']]
bars = ax.barh(coef_t1_t2['Feature'], coef_t1_t2['Coefficient'],
               color=colors, alpha=0.85, edgecolor='black', linewidth=1.2)
ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax.set_xlabel('Standardized Coefficient', fontsize=13, fontweight='bold')
ax.set_title(f'T1 → T2 GPA Growth Drivers (R² = {r2_t1_t2:.3f})',
             fontsize=14, fontweight='bold', pad=15)
ax.grid(axis='x', alpha=0.3)

for bar, val in zip(bars, coef_t1_t2['Coefficient']):
    x_pos = val + (0.03 if val > 0 else -0.03)
    ha = 'left' if val > 0 else 'right'
    ax.text(x_pos, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
            va='center', ha=ha, fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('regression_gpa_growth_t1_t2.png', dpi=300, bbox_inches='tight')
print("\n✓ T1→T2 growth regression chart: regression_gpa_growth_t1_t2.png")
plt.close()

print("\n[5/5] Multiple linear regression (3): T2 data predicting T2→T3 growth...")

df_t2 = df_full[df_full['term'] == 'T2'][[
    'student_id', 'hw_average', 'cw_average', 'core_assessments',
    'behavior_points', 'attendance_rate'
]].rename(columns={
    'hw_average': 'hw_t2',
    'cw_average': 'cw_t2',
    'core_assessments': 'assessment_t2',
    'behavior_points': 'behavior_t2',
    'attendance_rate': 'attendance_t2'
})

df_t2_merged = df_t2.merge(df_growth, on='student_id', how='inner')

X_t2 = df_t2_merged[['hw_t2', 'cw_t2', 'assessment_t2', 'behavior_t2', 'attendance_t2']].dropna()
y_t2_t3 = df_t2_merged.loc[X_t2.index, 'gpa_change_t2_t3']

print(f"  Sample size: {len(X_t2)}")

scaler_t2 = StandardScaler()
X_t2_scaled = scaler_t2.fit_transform(X_t2)

model_t2_t3 = LinearRegression()
model_t2_t3.fit(X_t2_scaled, y_t2_t3)

r2_t2_t3 = model_t2_t3.score(X_t2_scaled, y_t2_t3)
rmse_t2_t3 = np.sqrt(np.mean((y_t2_t3 - model_t2_t3.predict(X_t2_scaled)) ** 2))

print(f"\n  Model performance:")
print(f"    R² = {r2_t2_t3:.4f}")
print(f"    RMSE = {rmse_t2_t3:.2f} points")

coef_t2_t3 = pd.DataFrame({
    'Feature': ['Homework', 'Classwork', 'Assessment', 'Behavior', 'Attendance'],
    'Coefficient': model_t2_t3.coef_,
    'Abs_Coefficient': np.abs(model_t2_t3.coef_)
}).sort_values('Abs_Coefficient', ascending=False)

print(f"\n  Standardized regression coefficients:")
for _, row in coef_t2_t3.iterrows():
    print(f"    {row['Feature']:15s}: {row['Coefficient']:7.4f}")

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#e74c3c' if c < 0 else '#2ecc71' for c in coef_t2_t3['Coefficient']]
bars = ax.barh(coef_t2_t3['Feature'], coef_t2_t3['Coefficient'],
               color=colors, alpha=0.85, edgecolor='black', linewidth=1.2)
ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax.set_xlabel('Standardized Coefficient', fontsize=13, fontweight='bold')
ax.set_title(f'T2 → T3 GPA Growth Drivers (R² = {r2_t2_t3:.3f})',
             fontsize=14, fontweight='bold', pad=15)
ax.grid(axis='x', alpha=0.3)

for bar, val in zip(bars, coef_t2_t3['Coefficient']):
    x_pos = val + (0.03 if val > 0 else -0.03)
    ha = 'left' if val > 0 else 'right'
    ax.text(x_pos, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
            va='center', ha=ha, fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('regression_gpa_growth_t2_t3.png', dpi=300, bbox_inches='tight')
print("\n✓ T2→T3 growth regression chart: regression_gpa_growth_t2_t3.png")
plt.close()

print("\n" + "=" * 60)
print("Module 03 Complete - Analysis Results")
print("=" * 60)

comparison = pd.DataFrame({
    'Factor': ['Homework', 'Classwork', 'Assessment', 'Behavior', 'Attendance'],
    'T1→T2_Coef': model_t1_t2.coef_,
    'T2→T3_Coef': model_t2_t3.coef_,
})
comparison['Difference'] = comparison['T2→T3_Coef'] - comparison['T1→T2_Coef']
comparison = comparison.sort_values('T1→T2_Coef', ascending=False, key=abs)

comparison.to_csv('regression_comparison.csv', index=False)
print("\n✓ Coefficient comparison table saved: regression_comparison.csv")

print("\n【Final GPA Driving Factors - Standardized Coefficients】")
print(coefficients_gpa[['Feature', 'Coefficient']].to_string(index=False))

print(f"\n【GPA Growth Driving Factors Comparison】")
print(comparison.to_string(index=False))

print(f"\n【Model Performance Metrics】")
print(f"  Final GPA:   R² = {r_squared_gpa:.4f}, RMSE = {rmse_gpa:.2f}")
print(f"  T1→T2 growth:   R² = {r2_t1_t2:.4f}, RMSE = {rmse_t1_t2:.2f}")
print(f"  T2→T3 growth:   R² = {r2_t2_t3:.4f}, RMSE = {rmse_t2_t3:.2f}")

print("\n" + "=" * 60)
print("✓ Module 03 Complete!")
print("  Output files:")
print("    - correlation_heatmap.png")
print("    - regression_final_gpa.png")
print("    - regression_gpa_growth_t1_t2.png")
print("    - regression_gpa_growth_t2_t3.png")
print("    - regression_comparison.csv")
print("=" * 60)

