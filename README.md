# 🏦 Credit Risk Analysis — Koperasi Simpan Pinjam

End-to-end machine learning project analyzing loan applicants to **(1) segment borrower profiles** and **(2) predict loan default risk**, framed around the cooperative (koperasi simpan pinjam) lending domain.

> Built by a Legal Analyst working in a koperasi, combining domain knowledge with data science.

---

## 📌 Problem Statement

A cooperative lending institution needs to answer two questions from its borrower data:
1. **"What types of borrowers do we have?"** → to design targeted lending strategies
2. **"Who is likely to default?"** → to mitigate credit risk before approval

This project answers both using **unsupervised** and **supervised** machine learning.

---

## 📊 Dataset

- **32,581** loan records, **12** features
- Target: `loan_status` (0 = repaid, 1 = default)
- **Imbalanced:** only ~22% of loans defaulted
- Features: age, income, employment length, loan amount, interest rate, loan-to-income ratio, home ownership, loan intent, loan grade, credit history

---

## 🔧 Workflow

### 1. Data Cleaning
- Imputed missing values (`person_emp_length`, `loan_int_rate`) using **median** (robust to outliers)
- Removed invalid records (e.g., age > 100, employment length > 60 years — data entry errors)

### 2. Exploratory Data Analysis (EDA)
- Borrowers who **default** tend to have **higher interest rates** (median ~13% vs ~11%)
- **Loan-to-income ratio** is strongly linked to default (median ~24% vs ~13%)

### 3. Unsupervised Learning — Borrower Segmentation
- **K-Means (k=3)** identified 3 segments:
  | Segment | Profile | Default Rate |
  |---|---|---|
  | Established & Safe | older, high income, low ratio | 15.5% |
  | **High Risk** | large loans, high loan/income ratio | **39%** |
  | Young & Cautious | low income but conservative borrowing | 14.6% |
- **DBSCAN** flagged **104 anomalies** (~0.3%) — high-income outliers (~9x avg income), likely premium/VIP members with low risk
- Cluster count chosen via **Elbow Method** + **Silhouette Score**

### 4. Supervised Learning — Default Prediction
- Compared **Decision Tree, SVM, Random Forest**
- **Random Forest** performed best:
  | Metric | Score |
  |---|---|
  | Accuracy | 93% |
  | Precision (default) | 95% |
  | Recall (default) | 74% |
- ⚠️ Since data is imbalanced, focused on **precision/recall**, not just accuracy. Recall of 74% means ~26% of risky borrowers still go undetected — an area for improvement (e.g., SMOTE, class weighting).

### 5. Feature Importance
Top 3 drivers of default (Random Forest):
1. **Loan-to-income ratio (21%)**
2. **Income (15%)**
3. **Interest rate (12%)**

---

## 🎯 Key Insight

Three independent approaches — **EDA, clustering, and feature importance** — all converged on the same finding:

> **Loan-to-income ratio is the dominant predictor of default risk.**

**Business recommendation:** the cooperative should prioritize screening on loan-to-income ratio and consider a maximum ratio policy (e.g., < 25%) as a risk-mitigation measure.

---

## 🛠️ Tech Stack
`Python` · `pandas` · `scikit-learn` · `matplotlib` · `seaborn`

## 📂 Files
- `credit_risk_analysis.ipynb` — full analysis notebook
- `credit_risk_dataset.csv` — dataset
- `README.md` — this file

---

## 🚀 How to Run
```bash
pip install pandas scikit-learn matplotlib seaborn
jupyter notebook credit_risk_analysis.ipynb
```
