# Credit Risk Scoring (Loan Default Prediction)

This project builds a machine learning pipeline to estimate the probability that a loan applicant will default. It uses real credit application data and walks through the full workflow you'd expect on an applied data science / risk team:

* data cleaning and preprocessing
* class imbalance handling
* model training with XGBoost / LightGBM / CatBoost
* model evaluation (ROC-AUC, precision/recall)
* model export for deployment

The final model can be used as the core of an internal credit decisioning tool or scored live via an API / Streamlit app.
---

## 🔍 Problem Definition

**Goal:** Predict whether a client will have difficulty repaying a loan.

**Target variable:**

* `TARGET = 1` → client had repayment issues (high risk)
* `TARGET = 0` → client repaid as expected (low risk)

In real lending, this type of model is used to:

* approve / decline applications
* price the loan (higher risk → higher interest)
* assign customers to manual review / enhanced KYC

---

📂 Data

This project uses the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk/data)

Download and place it in the `data/` folder before running.

---

## 🧠 Methodology / Pipeline

### 1. Data Loading & Inspection

* Load raw application data.
* Inspect shape, column types, and missingness.
* Review target balance (`TARGET` values).
* High-level sanity check (duplicates, ranges, distributions).

### 2. Data Cleaning

* Drop columns with more than 40% missing values.
* Impute remaining missing values:

  * Numeric columns → median
  * Categorical columns → mode
* Confirm no missing values remain in the modeling dataset.

Why: Credit datasets are notoriously sparse (external bureau info, derived ratios, etc.). High-NA features are usually volatile or leaky and can't be relied on in production.

### 3. Feature Engineering / Preprocessing

* Split features/target: `X` (all predictors), `y` (`TARGET`).
* Train/test split with `stratify=y` to preserve minority class %.
* Scale numeric features using `StandardScaler`.
* One-hot encode categorical features using `OneHotEncoder` with `handle_unknown='ignore'`.
* Concatenate numeric + encoded categorical into a single design matrix.
* Clean column names to remove characters that break LightGBM / CatBoost.

After this step we have:

* `X_train_final`
* `X_test_final`

Both are fully numeric, aligned, and model-ready.

### 4. Class Imbalance Handling (SMOTE)

The dataset is imbalanced (defaults are the minority class).
Most naive models would just predict "no default" for everyone and look accurate but be useless.

We fix this with **SMOTE (Synthetic Minority Oversampling Technique)**:

* Generates synthetic minority-class samples in the training set feature space.
* Balances the training data without touching the test set.

Result:

* The model is forced to learn patterns of risky borrowers instead of just memorizing the majority class.

We visualize before/after class counts to confirm SMOTE worked.

### 5. Modeling

We train and compare 3 gradient boosting methods:

* **XGBoost**

  * Strong baseline for tabular credit scoring.
  * Tuned with basic defaults + `eval_metric='logloss'`.

* **LightGBM**

  * Fast histogram-based learner.
  * Great for large, sparse feature spaces.

* **CatBoost**

  * Handles categorical structure effectively.
  * We also include light regularization (`depth=6`, `l2_leaf_reg=5`) to reduce overfitting.

All models are trained on the SMOTE-balanced training data.

### 6. Evaluation

We evaluate each model on the untouched test set using:

* Confusion matrix
* Precision / recall / F1 (via `classification_report`)
* ROC-AUC (core metric for ranking credit risk)

Why ROC-AUC?
In lending, you're ranking who is more/less risky. You don't always hard-reject at 0.5 — you choose cutoffs. ROC-AUC measures how well the model separates good vs bad borrowers across all thresholds.

We specifically care about recall on the minority class (`TARGET=1`), because missing a risky borrower is expensive.

---

## 📈 Results

Performance summary (test set):

| Model    | ROC-AUC | Notes                                 |
| -------- | ------- | ------------------------------------- |
| XGBoost  | ~0.74   | Best overall ranking of risky clients |
| CatBoost | ~0.73   | Strong, stable alternative            |
| LightGBM | ~0.71   | Slightly weaker recall on defaulters  |

**Takeaway:**
XGBoost provided the best tradeoff between catching high-risk borrowers and avoiding too many false alarms.

---

## 🔍 Feature Importance

We plot the top 15 most important features for each model.

Consistently high-signal features include:

* `EXT_SOURCE_2`, `EXT_SOURCE_3`
  → External risk scores / credit bureau style variables
* `DAYS_BIRTH`
  → Age proxy (negative number of days since birth)
* `DAYS_EMPLOYED`
  → Employment stability / tenure
* `AMT_CREDIT`, `AMT_ANNUITY`, `AMT_INCOME_TOTAL`
  → Affordability and leverage

Interpretation:

* The model is learning what real credit teams care about:

  * stability,
  * debt load relative to income,
  * bureau-like risk scores.

These are exactly the variables credit policy teams look at in production.

---

## 💾 Model Export / Deployment Notes

We export:

* `xgb_model.pkl` → trained XGBoost classifier
* `feature_columns.pkl` → the exact column order after preprocessing

Why this matters:

* In deployment you *must* feed features in the same order with the same encoded column names.
* Having both the model and feature list makes it easy to plug into:

  * a Streamlit app,
  * a Flask/FastAPI scoring API,
  * or batch inference in production.

Minimal loading example:

```python
import joblib
import pandas as pd

model = joblib.load("xgb_model.pkl")
feature_cols = joblib.load("feature_columns.pkl")

# new_applicant_df = ...  # build a single-row dataframe
# new_applicant_df = new_applicant_df.reindex(columns=feature_cols, fill_value=0)

# proba = model.predict_proba(new_applicant_df)[:, 1][0]
# print("Estimated default probability:", proba)
```

---

## ⚠️ Notes on Explainability

Regulated credit scoring normally requires explainability (for compliance and fairness review).
We attempted SHAP for per-feature attributions but ran into environment/version issues (not unusual on Windows without build tools).

Planned next step:

* Add SHAP summary plots and per-row force plots for "reason codes" (e.g. *"High credit amount vs income"*).
* Add a fairness slice analysis (compare false positive rate by gender / age bucket).

---

## 📌 Key Takeaways

* Built a realistic credit default prediction workflow end-to-end.
* Cleaned and imputed messy, high-missing financial data.
* Handled extreme class imbalance with SMOTE.
* Trained and compared 3 industry-grade gradient boosting models.
* Evaluated with ROC-AUC instead of just accuracy (important in imbalanced credit problems).
* Exported the production model and its schema for deployment.

This notebook represents the kind of work done in credit risk modeling / fintech underwriting / lending analytics teams.

---

## 🛠 Tech Stack

* Python
* pandas, numpy, matplotlib, seaborn
* scikit-learn
* imbalanced-learn (SMOTE)
* XGBoost
* LightGBM
* CatBoost
* joblib

---

## 👤 Author

**Fares Jony**
Data Science & Machine Learning
M.Sc. (Data Science)

If you'd like to discuss credit risk modeling / fraud scoring / underwriting automat
