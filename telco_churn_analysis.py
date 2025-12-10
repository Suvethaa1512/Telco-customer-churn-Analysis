# -------------------------------
# TELCO CUSTOMER CHURN ANALYSIS
# -------------------------------

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------------------
# 1. LOAD DATA
# ------------------------------------
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

print(df.shape)
print(df.info())
print(df.describe().T)
print("Null values:\n", df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())

# ------------------------------------
# 2. CLEANING & PREPROCESSING
# ------------------------------------

# Fix TotalCharges mixed type issue
df['TotalCharges'] = df['TotalCharges'].str.strip()           # remove spaces
df['TotalCharges'] = df['TotalCharges'].replace('', np.nan)   # replace empty with NaN
df['TotalCharges'] = df['TotalCharges'].astype(float)         # convert to float
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Drop customerID
df.drop('customerID', axis=1, inplace=True)

# Encode target variable
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Encode categorical features using LabelEncoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

print(df.info())

# ------------------------------------
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ------------------------------------

# Boxplot
sns.boxplot(x='tenure', y='Churn', data=df)
plt.show()

# Correlation heatmap
plt.figure(figsize=(18,10))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.show()

# Scatterplots
sns.scatterplot(x='tenure', y='TotalCharges', hue='Churn', data=df)
plt.show()

sns.scatterplot(x='MonthlyCharges', y='TotalCharges', hue='Churn', data=df)
plt.show()

# Countplots
sns.countplot(x='OnlineSecurity', hue='Churn', data=df)
plt.show()

sns.countplot(x='TechSupport', hue='Churn', data=df)
plt.show()

# Histograms
sns.histplot(df['tenure'], kde=True)
plt.show()

sns.histplot(df['MonthlyCharges'], kde=True)
plt.show()

sns.histplot(df['TotalCharges'], kde=True)
plt.show()

# ------------------------------------
# 4. MODEL TRAINING (LOGISTIC REGRESSION)
# ------------------------------------

X = df.drop('Churn', axis=1)
y = df['Churn']

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Identify column types
cat_cols = X.select_dtypes(include='object').columns.tolist()
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Scaling
scaler = StandardScaler()
X_num_scaled = scaler.fit_transform(X[num_cols])

# One-hot encoding
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_cat_encoded = ohe.fit_transform(X[cat_cols])

# Combine
X_preprocessed = np.hstack((X_num_scaled, X_cat_encoded))
print("Final shape:", X_preprocessed.shape)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_preprocessed, y, test_size=0.2, random_state=42
)

# Model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Predictions + accuracy
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)
