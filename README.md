# Telco Customer Churn Analysis

## 📌 Project Overview

Customer churn is a critical problem for telecom companies. This project analyzes customer data to identify key factors that influence churn and provides actionable insights to reduce customer loss.

The analysis includes data cleaning, exploratory data analysis (EDA), feature engineering, and churn prediction using machine learning. A Power BI dashboard is also created to visualize churn patterns.

---

## 📊 Dataset Description

* **Dataset Name:** Telco Customer Churn
* **Source:** IBM Sample Dataset
* **Records:** ~7,000 customers
* **Target Variable:** `Churn` (Yes / No)

### Key Features:

* Customer demographics (gender, senior citizen, dependents)
* Account information (tenure, contract, payment method)
* Services subscribed (internet, phone, streaming, security)
* Charges (monthly and total charges)

---

## 🧹 Data Cleaning & Preprocessing

* Handled missing and blank values in `TotalCharges`
* Converted categorical variables using One-Hot Encoding
* Scaled numerical features using StandardScaler
* Removed irrelevant columns (Customer ID)

---

## 📈 Exploratory Data Analysis (EDA)

* Churn distribution analysis
* Relationship between tenure, charges, and churn
* Service-based churn comparison (Tech Support, Online Security)
* Correlation heatmap to identify important features

---

## 🤖 Machine Learning Model

* **Model Used:** Logistic Regression
* **Train-Test Split:** 80% / 20%
* **Evaluation Metric:** Accuracy

The model provides a baseline approach for predicting customer churn.

---

## 📊 Power BI Dashboard

A Power BI dashboard (`telco_churn.pbix`) is included to visualize:

* Churn rate by contract type
* Churn vs tenure
* Monthly charges vs churn
* Service usage impact on churn

---

## 🛠️ Tools & Technologies

* Python (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn)
* Power BI
* GitHub

---

## 📂 Repository Structure

* `WA_Fn-UseC_-Telco-Customer-Churn.csv` – Dataset
* `telco_churn_analysis.py` – Python analysis & ML code
* `telco_churn.pbix` – Power BI dashboard

---

## 🎯 Key Insights

* Customers with short tenure are more likely to churn
* Higher monthly charges increase churn probability
* Lack of Tech Support and Online Security increases churn risk
* Long-term contracts significantly reduce churn

---

## 👩‍💻 Author

**Suvethaa B**
Aspiring Data Analyst | Data Science Enthusiast

---
