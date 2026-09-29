# E-Commerce Customer Churn Prediction & Retention Intelligence System

## 1. Project Overview

The E-Commerce Customer Churn Prediction & Retention Intelligence System is an end-to-end Machine Learning project designed to predict whether an e-commerce customer is likely to churn.

The system analyzes customer behavior, ordering activity, satisfaction, complaints, payment preferences, app usage, and other customer characteristics to identify potential churn risk.

The project covers the complete Machine Learning lifecycle:

Business Understanding → Data Collection → Data Understanding → Data Cleaning → Exploratory Data Analysis → Business Insights → Feature Engineering → Machine Learning → Model Evaluation → Hyperparameter Tuning → Model Interpretation → FastAPI → Testing → Docker Deployment

---

## 2. Business Problem

Customer churn occurs when customers stop purchasing from an e-commerce company and become inactive.

Customer acquisition can be expensive, so identifying customers who may be at risk of churn can help the retention team take preventive actions.

### Objective

The objective of this project is to develop a Machine Learning classification system that predicts whether a customer is likely to churn based on customer characteristics and behavioral information.

### Target Variable

```text
Churn = 0 → Active Customer
Churn = 1 → Churned Customer
```

---

## 3. Project Goals

- Understand customer churn behavior.
- Identify patterns associated with customer churn.
- Clean and prepare customer data.
- Perform exploratory data analysis.
- Generate meaningful business insights.
- Engineer useful customer behavior features.
- Train multiple Machine Learning classification models.
- Compare model performance using multiple evaluation metrics.
- Perform hyperparameter tuning.
- Interpret important churn-related features.
- Build a production-oriented Machine Learning pipeline.
- Expose predictions through FastAPI.
- Validate API inputs using Pydantic.
- Test the API.
- Containerize the application using Docker.

---

## 4. Dataset

### Dataset Name

Ecommerce Customer Churn Analysis and Prediction

### Dataset Source

Kaggle

### Dataset File

```text
data/raw/E Commerce Dataset.xlsx
```

### Dataset File

```text
data/raw/E Commerce Dataset.xlsx
```

Then copy-paste **everything below this point** as one block:

### Dataset Size

```text
Rows    : 5,630
Columns : 20
```

```

Then copy-paste **everything below this point** as one block:

Churn

Churn = 0 → Active Customer
Churn = 1 → Churned Customer
```

ecommerce-churn-prediction/
│
├── data/
│ ├── raw/
│ │ └── E Commerce Dataset.xlsx
│ │
│ └── processed/
│ ├── ecommerce_churn_cleaned.csv
│ ├── model_comparison.csv
│ └── feature_importance.csv
│
├── notebooks/
│ ├── 01_data_understanding.ipynb
│ ├── 02_data_cleaning.ipynb
│ ├── 03_eda.ipynb
│ └── 05_model_experiments.ipynb
│
├── src/
│ ├── data_processing.py
│ ├── feature_engineering.py
│ ├── train.py
│ ├── evaluate.py
│ └── predict.py
│
├── api/
│ ├── main.py
│ └── schemas.py
│
├── artifacts/
│ └── churn_pipeline.joblib
│
├── tests/
│ └── test_api.py
│
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── README.md
└── .gitignore
