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

- **Feature engineering:** None applied beyond starter code
- **Feature selection:** Dropped num_outbound_cmds 
- **Scaling:** None applied,XGBoost does not require feature scaling
- **Other:** Categorical features (protocol_type, service, flag) were label encoded. Attack types were mapped to 5 numeric categories (0=Normal, 1=DoS, 2=Probe, 3=R2L, 4=U2R)

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

- **Algorithm:**  XGBoost
- **What changed from baseline:** Our own baseline run with default parameters (n_estimators=100, max_depth=6, learning_rate=0.3, subsample=1)
- **Macro F1 (CV):**  0.9447 (± 0.0127)
- **Macro F1 (test):**  0.5600200543620495
- **Observation:**  Already beats the Random Forest baseline of 0.47. However R2L and U2R detection is still poor due to class imbalance.

### Experiment 2: [Name / description]

- **Algorithm:** XGBoost
- **What changed:** Increased n_estimators to 200, reduced learning_rate to 0.1, reduced subsample to 0.8
- **Macro F1 (CV):** 0.9433 (± 0.0180)
- **Macro F1 (test):** 0.5741628324208399
- **Observation:** Small but consistent improvement. Lower learning rate with more trees generalizes better.

### Experiment 3: [Name / description]

- **Algorithm:** XGBoost + SMOTE
- **What changed:** Applied SMOTE oversampling to the training data before training (default model params: n_estimators=100, max_depth=6, learning_rate=0.3, subsample=1). SMOTE raised every class to 67,343 training examples.
- **Macro F1 (CV):** 0.9998 (± 0.0001)
- **Macro F1 (test):** 0.6320434249728675
- **Observation:** Best result so far. R2L recall improved from 0.05 to 0.16 and U2R recall from 0.16 to 0.25. The near-perfect CV score (0.9998) is misleading, it happens because SMOTE's synthetic samples leak across cross-validation folds, so CV is not a reliable estimate here.

### Experiments Summary

| # | Description            | Algorithm       | Imbalance Handling | Macro F1 (CV) | Macro F1 (test)    |
|---|------------------------|-----------------|--------------------|---------------|--------------------|
| 1 | Default params         | XGBoost         |                    | 0.9447        | 0.5600200543620495 |
| 2 | Tuned params           | XGBoost         |                    | 0.9433        | 0.5741628324208399 |
| 3 | SMOTE + default params | XGBoost + SMOTE |                    | 0.9998        | 0.6320434249728675 |
| 4 |                        |                 |                    |               |                    |
| 5 |                        |                 |                    |               |                    |

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

- **Hardware:** i7-11800H, 16GB, NVIDIA T1200 - Ryzen-9 9955HX, RTX 5060 8GB
- **Python version:** 3.14.5
- **Key libraries:** [scikit-learn version, xgboost version, etc.]
- **Random seed:** 42