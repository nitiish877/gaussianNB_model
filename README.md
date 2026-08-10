# Patient Risk Classification — Gaussian Naive Bayes

This project builds a **multiclass classifier** that predicts a patient's health risk level — **Low, Medium, or High** — from six clinical measurements. The model used is **Gaussian Naive Bayes**, which is well suited to this problem since all input features are continuous numeric values (as opposed to word counts or categories, where a different variant of Naive Bayes would be used).

## Dataset

- **File:** `patient_risk_raw.csv` — 610 synthetic patient records
- **Features (all continuous):** `age`, `blood_pressure`, `cholesterol`, `bmi`, `glucose_level`, `heart_rate`
- **Target:** `risk_class` → `Low_Risk`, `Medium_Risk`, `High_Risk`

The dataset was deliberately generated to be messy, so it would resemble real-world data rather than a clean textbook dataset. Specifically, it contains:
- Missing values scattered across all numeric columns (~4–8% each)
- A few extreme outlier values
- Physiologically impossible negative readings (simulating data-entry errors)
- Inconsistently formatted labels — `"Low_Risk"`, `"low risk"`, `"  LOW_RISK  "`, etc. all referring to the same class
- Some duplicate patient IDs and duplicate rows

## Project Workflow

**1. Load the data**
The raw CSV is read into a pandas DataFrame.

**2. Handle missing values**
Rather than filling every missing value with a single global median (which ignores context), values are imputed using **group-wise medians** — a more context-aware approach:
- `age`, `blood_pressure`, `cholesterol`, and `heart_rate` are filled using the median within the patient's `age_bins` (age bucketed into ranges like 20–35, 35–50, etc.)
- `bmi` is filled using the median within `glucose_bins`
- `glucose_level` is filled using the median grouped by `cholesterol`
- If a group-level median isn't available for a row, it falls back to the overall column median

**3. Clean the target labels**
Since the raw labels have inconsistent casing and spacing, string matching (`.str.contains`, case-insensitive) is used to normalize every variant into one of three clean categories, which are then mapped to integers:
`Low_Risk → 0`, `Medium_Risk → 1`, `High_Risk → 2`

**4. Train/test split**
The cleaned data is split 80/20 into training and test sets (`random_state=42` for reproducibility).

**5. Train the model**
A `GaussianNB` model is fit on the training data (unscaled features first).

**6. Evaluate**
The model is scored on the held-out test set using accuracy, macro-averaged precision/recall/F1, a confusion matrix, and a full classification report.

**7. Cross-validation**
5-fold cross-validation is run on the training set to check how consistent the model's performance is across different data splits, rather than relying on a single train/test split.

**8. Scaling experiment**
To check whether feature scaling improves performance, `StandardScaler` is fit on the training data and applied to the test data (avoiding data leakage), and the model is retrained and re-evaluated on the scaled features for comparison.

**9. Blind test on unseen data**
Finally, the trained model is tested on 100 brand-new samples (`patient_risk_TEST_100.csv`) that it had never seen during training, with the true labels provided separately (`patient_risk_TEST_100_ANSWERS.csv`) to verify real-world generalization.

## Results

| Metric | Test set (unscaled) | Test set (scaled) | 5-fold CV (mean) | Blind test (100 new samples) |
|---|---|---|---|---|
| Accuracy | 0.926 | 0.926 | 0.863 | 0.91 |
| Precision (macro) | 0.919 | 0.949 | — | 0.92 |
| Recall (macro) | 0.949 | 0.919 | — | 0.906 |
| F1 (macro) | 0.929 | 0.929 | — | 0.912 |

**Blind test confusion matrix** (rows = true class, columns = predicted class; order is Low, Medium, High):
```
[[24  3  0]
 [ 3 41  0]
 [ 0  3 26]]
```

## Key Insights

- **Scaling made almost no difference.** This is expected behavior for Gaussian Naive Bayes — it estimates a separate mean and variance for each feature within each class internally, so it already adapts to each feature's own scale. Algorithms like Logistic Regression or KNN, which rely on distances or gradient-based optimization, would be far more sensitive to unscaled features.
- **The model generalizes well.** The blind-test accuracy (0.91) is close to the original test accuracy (0.926) and actually higher than the 5-fold CV average (0.863), suggesting the model isn't overfitting to the training data.
- **Errors only happen between adjacent risk levels** (Low↔Medium or Medium↔High) — the model never confuses Low with High. This is a good sign, since risk level is a continuous spectrum in reality, so occasional confusion at the boundary is expected and low-severity.

## Tech Stack

Python, pandas, scikit-learn (`GaussianNB`, `StandardScaler`, `train_test_split`, `cross_val_score`, evaluation metrics)
