<p align="center">
  <img src="https://media.giphy.com/media/LMcB8XospGZO8UQq87/giphy.gif" width="480px"/>
</p>

<h1 align="center">Applied Machine Learning Projects</h1>

<p align="center">
  End-to-end, production-style data science work:
  churn prevention, credit risk, time-series forecasting, customer segmentation, and large-scale NLP.
</p>

<p align="center">
  <b>All projects here are built like products, not class assignments.</b>
</p>

---

## 📂 Project Index

| #  | Folder                                                                 | Problem it Solves                                                                 | Highlights                                                                                          | Stack / Methods                                                                                  |
|----|------------------------------------------------------------------------|-----------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| 01 | `01_Customer_Churn_Prediction_XGBoost_Streamlit`                      | Who is about to leave and how do we keep them?                                   | SMOTE + XGBoost (F1 on churners ≈ 0.88), Streamlit dashboard, what-if retention simulator           | Scikit-Learn, XGBoost, SMOTE, SHAP, Streamlit                                                   |
| 02 | `02_Credit_Risk_Default-Pipeline`                                     | Will this borrower default, and can we explain that decision?                    | Gradient boosted default model, SHAP driver explanations, human-readable risk tiers                | XGBoost / LightGBM, SHAP, Pandas                                                                 |
| 03 | `03_HybridTimeFusion_Forecasting_LSTM_Prophet_XGBoost`                | How much will each store sell next week?                                         | Hybrid LSTM + Prophet + XGBoost on ~400K+ rows, leakage-safe temporal CV, promo/holiday signals     | Keras (LSTM), Prophet, XGBoost, Pandas, NumPy                                                    |
| 04 | `04_RFM_OnlineRetail_UnSupervised_Learning`                           | Which customers are high value, discount-driven, or at churn risk?               | RFM feature engineering, unsupervised clustering, customer segmentation for targeting & retention  | RFM scoring, K-Means / clustering, Cohort-style segmentation, Pandas                             |
| 05 | `05_7MillionSteamReviews_NLP_UnSupervised`                            | What are players actually complaining about at scale?                            | ~6M+ Steam reviews, TF-IDF → NMF topics, SentenceTransformer embeddings → clustering of personas   | NLP pipeline, TF-IDF, NMF, SentenceTransformer embeddings, clustering, topic mining             |

> Click into each folder to see notebooks / code / requirements and (where relevant) app code.

---

## 🔍 Project Summaries

### 01_Customer_Churn_Prediction_XGBoost_Streamlit
**Goal:** Predict who’s about to churn, and simulate how to save them.  
**What’s inside:**
- Data prep, imbalance handling with SMOTE  
- XGBoost churn classifier with ~0.88 F1 on the churn class  
- SHAP explanations so you can see *why* a customer is high risk  
- Streamlit dashboard where a non-technical user can test:  
  “What if I offer 10% off?” / “What if I upgrade their plan?”

**Business value:**  
This is built like a retention tool, not just a model accuracy flex.

**Key tech:** `Python`, `Pandas`, `Scikit-Learn`, `XGBoost`, `SMOTE`, `SHAP`, `Streamlit`  
**Folder:** `01_Customer_Churn_Prediction_XGBoost_Streamlit/`

---

### 02_Credit_Risk_Default-Pipeline
**Goal:** Score borrower default risk and explain it to underwriting/compliance.  
**What’s inside:**
- Gradient boosting model on borrower features  
- SHAP feature attributions per applicant (“top 3 risk drivers”)  
- Risk buckets (Low / Medium / High) you could actually hand to a credit analyst

**Business value:**  
Moves the convo from “the model said no” → “here are the drivers of risk.”

**Key tech:** `LightGBM / XGBoost`, `Pandas`, `NumPy`, `SHAP`  
**Folder:** `02_Credit_Risk_Default-Pipeline/`

---

### 03_HybridTimeFusion_Forecasting_LSTM_Prophet_XGBoost
**Goal:** Forecast weekly sales per store / department so planners don’t stock out on promo weeks.  
**Data scale:** ~400K+ rows (multi-store, multi-department, multi-week retail signal).  
**What’s inside:**
- Hybrid forecasting stack:
  - LSTM (sequence patterns / short-term memory)
  - Prophet (seasonality, holidays)
  - XGBoost (price, promo, macro covariates)
- Leakage-safe time-aware cross validation (no “future info” cheating)
- Rolling and lag features, promo intensity, holiday effects

**Business value:**  
Answering: “How much inventory do we need next week, not last week?”

**Key tech:** `Keras (LSTM)`, `Prophet`, `XGBoost`, `Pandas`, `NumPy`, temporal CV  
**Folder:** `03_HybridTimeFusion_Forecasting_LSTM_Prophet_XGBoost/`

---

### 04_RFM_OnlineRetail_UnSupervised_Learning
**Goal:** Segment customers for retention and targeting without labels.  
**What’s inside:**
- RFM scoring (Recency, Frequency, Monetary value) from transaction data  
- Clustering to group customers (loyal whales, high spend but at-risk, discount hunters, etc.)  
- Playbook-level interpretation: which cluster should marketing touch, and how?

**Business value:**  
Tells the business “who matters, who’s fading, and who’s only here for discounts.”

**Key tech:** `Pandas`, `RFM features`, `Unsupervised clustering (e.g. K-Means)`, `Matplotlib/Seaborn` visuals  
**Folder:** `04_RFM_OnlineRetail_UnSupervised_Learning/`

---

### 05_7MillionSteamReviews_NLP_UnSupervised
**Goal:** Turn millions of raw player reviews into usable product feedback.  
**Scale:** ~7M Steam reviews (massive, noisy, emotional text).  
**What’s inside:**
- Text cleaning & normalization at scale  
- TF-IDF + NMF to find major complaint themes / feature requests  
- SentenceTransformer embeddings + clustering to group player “personas”:  
  - performance complainers  
  - balance grinders  
  - content-hunters / replayability people  
- Surfaced what actually drives negative sentiment

**Business value:**  
This is automated voice-of-customer. You can aim this at any product with reviews.

**Key tech:** `NLP`, `TF-IDF`, `NMF`, `SentenceTransformer`, clustering, `Pandas`  
**Folder:** `05_7MillionSteamReviews_NLP_UnSupervised/`

---

## ⚙ How to Run Any Project Locally

All projects follow roughly the same pattern:

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO/01_Customer_Churn_Prediction_XGBoost_Streamlit

# 2. Create / activate an environment (conda or venv)
conda create -n dsproj python=3.11 -y
conda activate dsproj
# OR: python -m venv .venv && source .venv/bin/activate (on Windows: .venv\Scripts\activate)

# 3. Install requirements
pip install -r requirements.txt

# 4. (If the project has an app) Run it
streamlit run app.py
