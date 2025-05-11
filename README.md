# 🛡️ FinFraudGuard

**FinFraudGuard** is a machine learning–powered system for real-time detection of fraudulent credit card transactions.  
It includes a full ML lifecycle: data exploration, model training, feature engineering, deployment via FastAPI, and integration testing with real transaction samples.

---

## 🔍 Project Overview

This system addresses the challenge of fraud detection in highly imbalanced datasets by combining advanced preprocessing techniques with interpretable ML models.

### ✅ Features:
- Machine Learning model trained on the [ULB Credit Card Fraud Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- Custom feature engineering: time-based and amount-based fraud signals
- API serving with `FastAPI`
- Prediction endpoint: `POST /predict`
- Full test suite with real-world transaction data
- SHAP analysis for feature importance (in notebook)

---

## 🧠 Workflow Summary

### 1. [`notebooks/FinFraudGuard.ipynb`](notebooks/FinFraudGuard.ipynb)
- Exploratory Data Analysis (EDA)
- Handling of class imbalance (undersampling, stratification)
- Feature engineering:
  - `Hour` extracted from `Time`
  - `is_night`: flag for night-time transactions
  - `log_Amount`, `is_high_amount`: transformations for suspicious values
- Model training:
  - Logistic Regression (baseline)
  - Random Forest (final)
- SHAP interpretability visualization
- Export of trained model + feature list

### 2. [`src/predict.py`](src/predict.py)
- Loads trained Random Forest model
- Reproduces all preprocessing from training
- Makes predictions from raw JSON input

### 3. [`api/main.py`](api/main.py)
- FastAPI app with:
  - `POST /predict` — fraud classification endpoint
  - Swagger UI enabled at `/docs`

### 4. [`tests/test_requests.py`](tests/test_requests.py)
- Sends real transaction payloads to the API
- Validates predicted fraud flag against expected
- Shows `✅ PASSED / ❌ FAILED` with returned probability

---

## 📦 Project Structure

```text
FinFraudGuard/
├── api/
│   └── main.py                 # FastAPI app
├── src/
│   └── predict.py              # Preprocessing + model loading
├── models/
│   ├── model_randomforest.pkl  # Trained model (joblib)
│   └── feature_columns.json    # Feature order used in training
├── notebooks/
│   └── FinFraudGuard.ipynb     # Full EDA and training pipeline
├── tests/
│   └── test_requests.py        # Integration tests
├── requirements.txt
└── README.md                   # You're here