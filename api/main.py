from pathlib import Path
import sys

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException

# --------------------------------------------------
# Project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

SRC_DIR = BASE_DIR / "src"
MODEL_PATH = BASE_DIR / "artifacts" / "churn_pipeline.joblib"

# Make src available
sys.path.insert(0, str(SRC_DIR))

from feature_engineering import create_features
from schemas import CustomerInput, PredictionResponse


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="E-Commerce Customer Churn Prediction API",
    description="API for predicting customer churn risk",
    version="1.0.0"
)


# --------------------------------------------------
# Load ML model
# --------------------------------------------------

try:
    model = joblib.load(MODEL_PATH)
    MODEL_LOAD_ERROR = None

except Exception as exc:
    model = None
    MODEL_LOAD_ERROR = repr(exc)


# --------------------------------------------------
# Root endpoint
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "E-Commerce Customer Churn Prediction API",
        "version": "1.0.0"
    }


# --------------------------------------------------
# Health endpoint
# --------------------------------------------------

@app.get("/health")
def health():

    if model is None:
        return {
            "status": "unhealthy",
            "model_loaded": False,
            "error": MODEL_LOAD_ERROR
        }

    return {
        "status": "healthy",
        "model_loaded": True
    }


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(customer: CustomerInput):

    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Prediction model could not be loaded."
        )

    customer_data = pd.DataFrame([{
        "Tenure": customer.tenure,
        "PreferredLoginDevice": customer.preferred_login_device,
        "CityTier": customer.city_tier,
        "WarehouseToHome": customer.warehouse_to_home,
        "PreferredPaymentMode": customer.preferred_payment_mode,
        "Gender": customer.gender,
        "HourSpendOnApp": customer.hour_spend_on_app,
        "NumberOfDeviceRegistered": customer.number_of_device_registered,
        "PreferedOrderCat": customer.prefered_order_cat,
        "SatisfactionScore": customer.satisfaction_score,
        "MaritalStatus": customer.marital_status,
        "NumberOfAddress": customer.number_of_address,
        "Complain": customer.complain,
        "OrderAmountHikeFromlastYear": (
            customer.order_amount_hike_from_last_year
        ),
        "CouponUsed": customer.coupon_used,
        "OrderCount": customer.order_count,
        "DaySinceLastOrder": customer.day_since_last_order,
        "CashbackAmount": customer.cashback_amount
    }])

    try:
        prediction = int(
            model.predict(customer_data)[0]
        )

        probability = float(
            model.predict_proba(customer_data)[0][1]
        )

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {exc}"
        )

    churn_status = (
        "High Risk"
        if prediction == 1
        else "Low Risk"
    )

    return PredictionResponse(
        prediction=prediction,
        churn_status=churn_status,
        churn_probability=round(probability, 4)
    )