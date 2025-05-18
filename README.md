# 🕵️‍♂️ Fraud Detection System — Predicting Financial Transaction Fraud with Machine Learning

This project builds a fraud detection pipeline using a real-world-style financial dataset to identify potentially fraudulent transactions. The model is optimized for high recall to ensure that most fraudulent transactions are caught — even at the cost of some false positives.

---

## 📁 Project Structure

| Phase | Description |
|-------|-------------|
| **Phase 1** | Data cleaning and preprocessing (see `clean_data.py`) |
| **Phase 2** | Exploratory Data Analysis (see `explore_data_analysis.ipynb`) |
| **Phase 3** | Feature engineering for behavioral and temporal signals (see `feature_engineering.ipynb`) |
| **Phase 4** | Training and tuning a Random Forest classifier using `RandomizedSearchCV` (see `train_model.ipynb`) |
| **Final**  | Tableau dashboard for showcasing predictions visually (CSV exported)

---

## 📊 Dataset Overview

- Transactions from a fictional bank over several years
- Includes merchant codes, chip usage, credit info, and fraud labels
- Highly imbalanced (fraud ~0.1%)

This project uses publicly available financial transaction data from Kaggle:

Kaggle Dataset: Credit Card Fraud Detection
📎 [Link to Dataset](https://www.kaggle.com/datasets/computingvictor/transactions-fraud-datasets?select=cards_data.csv)

---

## ✅ Project Goals

- Detect fraud with high **recall** (minimize false negatives)
- Extract **user-level** and **behavioral** features (e.g., chip use, spending z-scores)
- Compare original and refined models with visualizations
- Provide an interactive Tableau dashboard for stakeholders

---

## 📊 EDA Results (Exploratory Data Analysis)
### Charts showing inital data analysis for Phase 2
The charts generated are the amount distributions, correlation matrix, credit score, fraud by day, fraud by hour, fraud distributions, top merchant categories, and total debt distributions. By creating EDA Results, it helps visualize patterns for the feature engineering phase.

<img src="./outputs/eda_results/amount_distributions.png" width="500">
<img src="./outputs/eda_results/correlation_matrix.png" width="500">
<img src="./outputs/eda_results/credit_score.png" width="500">
<img src="./outputs/eda_results/fraud_by_day.png" width="500">
<img src="./outputs/eda_results/fraud_by_hour.png" width="500">
<img src="./outputs/eda_results/fraud_distribution.png" width="500">
<img src="./outputs/eda_results/top_merchant_category.png" width="500">
<img src="./outputs/eda_results/total_debt_distribution.png" width="500">


---

## 📈 Model Results

### 🔹 Original Model (No Feature Engineering)
- **Precision (fraud):** 0.5672  
- **Recall (fraud):** 0.1496  
- **F1 (fraud):** 0.2367  
- **Accuracy:** 99.90%

<p>📌 <strong>Confusion Matrix:</strong></p>
<img src="./outputs/train_model_result/original_model_confusion_matrix.png" width="500">

<p>📌 <strong>Feature Importance:</strong></p>
<img src="./outputs/train_model_result/original_model_feature_importance.png" width="500">


<sub>📄 Source: [`original_model_snapshot.txt`](./outputs/train_model_result/original_model_snapshot.txt)</sub>

---

### 🔹 Refined Model (Feature Engineered + Tuned)
- **Precision (fraud):** 0.0064  
- **Recall (fraud):** 0.8316 ✅  
- **F1 (fraud):** 0.0127  
- **Accuracy:** 86.95%

<p>📌 <strong>Confusion Matrix:</strong></p>
<img src="./outputs/train_model_result/refined_model_confusion_matrix.png" width="500">

<p>📌 <strong>Feature Importance:</strong></p>
<img src="./outputs/train_model_result/refined_model_feature_importance.png" width="500">


<sub>📄 Source: [`refined_model_snapshot.txt`](./outputs/train_model_result/refined_model_snapshot.txt)</sub>

---

## 🛠 Key Features Engineered

| Feature | Description |
|--------|-------------|
| `use_chip` | Binary flag for chip usage during transaction |
| `chip_usage_rate` | Historical chip usage frequency per card |
| `chip_deviation` | Whether this transaction deviates from user’s chip norm |
| `amount_zscore_user` | How anomalous the amount is vs. user's history |
| `ratio_zscore` | Z-score of amount-to-credit ratio by card |
| `mcc_code` | Merchant category classification |
| `transaction_hour` | Transaction time to capture night fraud patterns |

---

## 📊 Tableau Dashboard

A fraud monitoring dashboard built with Tableau shows:
- Predicted fraud probability for each transaction
- Risk tiering (High/Medium/Low)
- Merchant trends and fraud hotspots
- Time-of-day and location-based insights

📎 [Link to Dashboard](https://public.tableau.com/views/FraudAnalysis_17475417268400/Dashboard1?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)

---

## 💡 How to Run This Project

1. Clone this repo
2. Install dependencies Using Requirements.txt file (Python ≥ 3.8, pandas, scikit-learn, matplotlib, seaborn)
3. Run the notebooks:
   - `clean_data.py`
   - `explore_data_analysis.ipynb`
   - `feature_engineering.ipynb`
   - `train_model.ipynb`
4. Load `fraud_predictions_for_tableau.csv` into Tableau

---

## 📁 File Guide

| File | Purpose |
|------|---------|
| `clean_data.py` | Loads and cleans raw CSV and JSON files |
| `explore_data_analysis.ipynb` | Explores data distributions, imbalance, early fraud insights |
| `feature_engineering.ipynb` | Adds behavioral and temporal fraud features |
| `train_model.ipynb` | Trains and tunes a random forest model with `RandomizedSearchCV` |
| `fraud_predictions_for_tableau.csv` | CSV used for Tableau dashboard |
| `outputs` | Visualizations and confusion matrices for README |
| `refined_model_snapshot.txt` | Final model performance snapshot |
| `original_model_snapshot.txt` | Baseline model metrics |

