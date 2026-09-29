import os
import pandas as pd
import joblib


# ============================================================
# MODEL PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "artifacts",
    "churn_pipeline.joblib"
)


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(
    MODEL_PATH
)


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_churn(customer_data: dict) -> dict:
    """
    Predict churn for a single customer.
    """

    # Convert dictionary to DataFrame
    customer_df = pd.DataFrame(
        [customer_data]
    )

    # Prediction
    prediction = model.predict(
        customer_df
    )[0]

    # Probability
    probability = model.predict_proba(
        customer_df
    )[0][1]

    # Churn status
    if prediction == 1:
        status = "High Risk"
    else:
        status = "Low Risk"

    return {
        "prediction": int(prediction),
        "churn_status": status,
        "churn_probability": round(
            float(probability),
            4
        )
    }


# ============================================================
# EXAMPLE CUSTOMER
# ============================================================

if __name__ == "__main__":

    customer = {

        "Tenure": 17.0,

        "PreferredLoginDevice": "Computer",

        "CityTier": 3,

        "WarehouseToHome": 9.0,

        "PreferredPaymentMode": "Debit Card",

        "Gender": "Male",

        "HourSpendOnApp": 4.0,

        "NumberOfDeviceRegistered": 4,

        "PreferedOrderCat": "Laptop & Accessory",

        "SatisfactionScore": 1,

        "MaritalStatus": "Married",

        "NumberOfAddress": 3,

        "Complain": 0,

        "OrderAmountHikeFromlastYear": 21.0,

        "CouponUsed": 6.0,

        "OrderCount": 8.0,

        "DaySinceLastOrder": 8.0,

        "CashbackAmount": 181.75
    }


    result = predict_churn(
        customer
    )


    print("\nPrediction Result")
    print("=" * 40)

    print(
        "Prediction:",
        result["prediction"]
    )

    print(
        "Churn Status:",
        result["churn_status"]
    )

    print(
        "Churn Probability:",
        result["churn_probability"]
    )