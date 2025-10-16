# Student Performance Analysis and Retention Risk Modeling
This quantitative project analyzes academic and behavioral data from 347 middle school students to identify factors driving academic success and predict students at risk of withdrawal.

---

## Project Overview

**Context**: Analysis of Grade 6-7 student performance data from a middle school cohort (N=347 students, 1,047 student-term observations across three terms).

**Objective**: 
- Identify factors that significantly impact student academic performance
- Develop a systematic framework for early identification of at-risk students
- Provide empirical evidence to support resource allocation decisions

**Scope**: Independent project encompassing data collection, preprocessing, statistical analysis, and predictive modeling.

---

## Methodology

### 1. Data Preparation
**File**: `01_data_cleaning.py`

- Integrated six data sources: term-level academic records (T1, T2, T3), annual GPA data, and withdrawal records
- Addressed data quality issues including missing values, inconsistent grade labels, and duplicate records
- Created derived features: attendance rate, GPA change metrics (term-over-term and cumulative)
- Final dataset: 347 students with complete academic, behavioral, and attendance records

### 2. Risk Assessment Framework
**File**: `02_kpi_analysis.py`

Developed a five-factor risk scoring system (0-5 scale) based on:
- Academic performance: GPA below 25th percentile, consecutive GPA decline
- Behavioral indicators: low behavior points, suspension records
- Attendance: attendance rate below 25th percentile

Risk classification:
- Level 0 (0-1 points): Normal
- Level 1 (2 points): Requires monitoring
- Level 2 (3 points): Moderate risk
- Level 3 (4-5 points): High risk

### 3. Factor Analysis
**File**: `03_regression_analysis.py`

**Objective**: Identify which factors drive academic performance and GPA growth across different time periods.

**Approach**: Multiple linear regression with standardized coefficients to enable direct comparison of factor importance.

**Three models**:
1. Annual model: Predicting final GPA using full-year averages of homework, classwork, assessments, behavior, and attendance
2. T1→T2 model: Predicting Term 1 to Term 2 GPA change using Term 1 metrics
3. T2→T3 model: Predicting Term 2 to Term 3 GPA change using Term 2 metrics

**Key finding**: Factor importance shifts across time periods—homework becomes increasingly important in later terms, while assessment scores remain consistently significant.

### 4. Withdrawal Prediction Model
**File**: `04_predictive_model.py`

**Model**: Logistic regression with class balancing to address low withdrawal rate (5.8%)

**Features**: 9 variables including GPA, homework/classwork averages, assessment scores, behavioral metrics, attendance rate, and the risk score from the KPI framework

**Evaluation**: Train-test split (75-25), stratified sampling to preserve class distribution

---

## Results

### Factor Analysis (Standardized Coefficients)

**Final GPA Model (R² = 0.92)**:
- Assessment scores: β = 0.85 (strongest predictor)
- Homework average: β = 0.32
- Classwork average: β = 0.28
- Behavior points: β = 0.15
- Attendance rate: β = 0.08

**Interpretation**: Assessment performance explains the majority of variance in final GPA. Attendance shows minimal impact, likely due to high baseline (95% average attendance rate).

### Risk Distribution

| Risk Level | Count | Percentage | Avg GPA | Withdrawal Rate |
|------------|-------|------------|---------|-----------------|
| Level 0    | 270   | 77.8%      | 88.3    | 6.3%            |
| Level 1    | 42    | 12.1%      | 78.5    | 0.0%            |
| Level 2    | 29    | 8.4%       | 74.5    | 10.3%           |
| Level 3    | 6     | 1.7%       | 71.0    | 0.0%            |

### Predictive Model Performance

- **Accuracy**: 85.2%
- **ROC-AUC**: 0.74
- **Precision (Withdrawn)**: 0.25
- **Recall (Withdrawn)**: 0.50

**Note**: Low precision reflects severe class imbalance (5.8% withdrawal rate). The model correctly identifies 50% of withdrawn students while maintaining 85% overall accuracy.

**Top predictors** (by absolute coefficient):
1. Risk score (β = +2.97)
2. GPA (β = -2.15)
3. Attendance rate (β = -1.08)

---

## Technical Implementation

**Language**: Python 3.x

**Libraries**:
- Data manipulation: pandas, numpy
- Statistical modeling: scikit-learn (LinearRegression, LogisticRegression, StandardScaler)
- Visualization: matplotlib, seaborn

**Environment**: Google Colab

**Key techniques**:
- Feature standardization for coefficient comparison
- Stratified sampling to handle class imbalance
- Cross-sectional and longitudinal analysis to capture time-varying effects

---

## Limitations and Future Improvements

**Current limitations**:
- Small sample size limits generalizability
- Class imbalance reduces precision for minority class (withdrawn students)
- Limited feature set—lacks data on family background, prior academic history, or external factors
- Static model does not account for within-year changes in student status

**Potential enhancements**:
- Collect additional predictive features (parent engagement, disciplinary history, prior school records)
- Apply resampling techniques (SMOTE) or ensemble methods to address class imbalance
- Develop time-series or survival analysis models for dynamic risk assessment
- Incorporate interaction terms to capture non-linear relationships

---

## Files and Outputs

**Code**:
- `01_data_cleaning.py` - Data integration and preprocessing
- `02_kpi_analysis.py` - Risk scoring and descriptive statistics
- `03_regression_analysis.py` - Factor analysis and regression modeling
- `04_predictive_model.py` - Logistic regression for withdrawal prediction

**Outputs**:
- `student_annual_metrics.csv` - Student-level aggregated data
- `kpi_analysis_reports.xlsx` - Risk distribution and subgroup comparisons
- `regression_comparison.csv` - Coefficient comparison across models
- `feature_importance.png`, `roc_curve.png`, `confusion_matrix.png` - Model diagnostics

---

## Conclusion

This project demonstrates the application of statistical methods to education data for early risk identification. The analysis reveals that academic performance metrics—particularly assessment scores—are the most reliable indicators of student outcomes, while behavioral and attendance factors play supporting roles. The predictive model achieves moderate discrimination (AUC = 0.74), though precision remains limited by class imbalance.

The framework provides a systematic, data-driven approach to identifying students who may benefit from early intervention, enabling more efficient resource allocation in educational settings.

---

**Note**: All data has been anonymized to protect student privacy. This project was conducted for educational purposes as part of coursework/independent study.
