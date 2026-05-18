# Assignment Report Example

---

**Course:** Advanced Python (ICS0019)

**Team members:** Austin Robert Derck, Sebastian Alberto Parra Pinto

**Date:** 25.05.2026

Repository link: https://gitlab.cs.ttu.ee/austid/ml_training

---

## 1. Approach

### 1.1 Strategy Overview

Briefly describe your overall strategy. What was your plan before you started experimenting? Probably none, but you may write so as well.

[Your strategy — e.g., "We decided to focus on improving R2L/U2R detection using SMOTE and gradient boosting."]

### 1.2 Preprocessing

Describe any changes you made to the data beyond the starter code:

- **Feature engineering:** [Did you create new features? Which ones and why?]
- **Feature selection:** [Did you remove any features? Based on what criteria?]
- **Scaling:** [Did you apply StandardScaler, MinMaxScaler, or none?]
- **Other:** [Any other preprocessing steps?]

### 1.3 Class Imbalance Handling

How did you address the imbalance between classes?

- **Method used:** [SMOTE / class_weight / undersampling / combination / none]
- **Parameters:** [e.g., SMOTE with k_neighbors=5, or class_weight='balanced']
- **Effect on training set distribution:** [how did the class distribution change?]

---

## 2. Experiments

Always document experiments you ran. Fill in the summary table will all the experiments. Add the descriptions for the ones you find important.

### Total number of experiments:

### Experiment 1: [Name / description]

- **Algorithm:** [e.g., Random Forest with default parameters]
- **What changed from baseline:** [e.g., "Nothing — this is our own baseline run"]
- **Macro F1 (CV):** [cross-validation score]
- **Macro F1 (test):** [KDDTest+ score]
- **Observation:** [what did you learn from this experiment?]

### Experiment 2: [Name / description]

- **Algorithm:**
- **What changed:** [e.g., "Added SMOTE before training"]
- **Macro F1 (CV):**
- **Macro F1 (test):**
- **Observation:**

### Experiment 3: [Name / description]

- **Algorithm:**
- **What changed:**
- **Macro F1 (CV):**
- **Macro F1 (test):**
- **Observation:**

### Experiments Summary

| # | Description | Algorithm | Imbalance Handling | Macro F1 (CV) | Macro F1 (test) |
|---|-------------|-----------|--------------------|---------------|-----------------|
| 1 |             |           |                    |               |                 |
| 2 |             |           |                    |               |                 |
| 3 |             |           |                    |               |                 |
| 4 |             |           |                    |               |                 |
| 5 |             |           |                    |               |                 |

---

## 3. Final Results

### 3.1 Best Model

- **Algorithm:** [e.g., XGBoost]
- **Key parameters:** [e.g., n_estimators=200, max_depth=6, learning_rate=0.1, scale_pos_weight=...]
- **Imbalance handling:** [e.g., SMOTE + class_weight]
- **Feature engineering:** [e.g., added src_bytes/dst_bytes ratio]

### 3.2 Final Macro F1-Score

| Metric              | Score |
|---------------------|-------|
| **Macro F1 (test)** |       |
| Macro F1 (CV)       |       |

### 3.3 Classification Report

| Category | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Normal   |           |        |          |         |
| DoS      |           |        |          |         |
| Probe    |           |        |          |         |
| R2L      |           |        |          |         |
| U2R      |           |        |          |         |

### 3.4 Confusion Matrix

[Generate this image using the code from Section 9 of the guidebook. For Markdown report save it as `confusion_matrix.png` in the same folder as this report and add link `![Confusion Matrix](confusion_matrix.png)`. Don’t forget the exclamation mark].

## 4. Cross-Validation vs. Test Score

- **CV macro F1:** [score ± std]
- **Test macro F1:** [score]
- **Gap:** [CV − test]

**Analysis:** [Explain the gap. Is it expected? Is it due to unseen attack types in KDDTest+? Does it indicate overfitting?]

---

## 5. What Worked and What Didn't

### What had the biggest positive impact?

[e.g., "SMOTE increased R2L recall from 0.00 to 0.12, which raised macro F1 by 0.08"]

### What surprisingly didn't help?

[e.g., "Feature selection with SelectKBest removed 15 features but macro F1 dropped — the removed features contained information useful for rare classes"]

### What would you try with more time?

[e.g., "Stacking ensemble, more aggressive hyperparameter tuning, deeper feature engineering"]

---

## Appendix: Environment

- **Hardware:** i7-11800H, 16GB, NVIDIA T1200
- **Python version:** 3.14.5
- **Key libraries:** [scikit-learn version, xgboost version, etc.]
- **Random seed:** 42