import requests


BASE_URL = "http://127.0.0.1:8000"


valid_customer = {
    "tenure": 17.0,
    "preferred_login_device": "Computer",
    "city_tier": 3,
    "warehouse_to_home": 9.0,
    "preferred_payment_mode": "Debit Card",
    "gender": "Male",
    "hour_spend_on_app": 4.0,
    "number_of_device_registered": 4,
    "prefered_order_cat": "Laptop & Accessory",
    "satisfaction_score": 1,
    "marital_status": "Married",
    "number_of_address": 3,
    "complain": 0,
    "order_amount_hike_from_last_year": 21.0,
    "coupon_used": 6.0,
    "order_count": 8.0,
    "day_since_last_order": 8.0,
    "cashback_amount": 181.75
}


def test_root():
    response = requests.get(
        f"{BASE_URL}/"
    )

    assert response.status_code == 200


def test_health():
    response = requests.get(
        f"{BASE_URL}/health"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_valid_prediction():
    response = requests.post(
        f"{BASE_URL}/predict",
        json=valid_customer
    )

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] in [0, 1]
    assert data["churn_status"] in [
        "High Risk",
        "Low Risk"
    ]
    assert 0 <= data["churn_probability"] <= 1


def test_negative_tenure():
    customer = valid_customer.copy()
    customer["tenure"] = -5

    response = requests.post(
        f"{BASE_URL}/predict",
        json=customer
    )

    assert response.status_code == 422


def test_invalid_satisfaction():
    customer = valid_customer.copy()
    customer["satisfaction_score"] = 8

    response = requests.post(
        f"{BASE_URL}/predict",
        json=customer
    )

    assert response.status_code == 422


def test_invalid_complain():
    customer = valid_customer.copy()
    customer["complain"] = 2

    response = requests.post(
        f"{BASE_URL}/predict",
        json=customer
    )

    assert response.status_code == 422


def test_missing_tenure():
    customer = valid_customer.copy()
    del customer["tenure"]

    response = requests.post(
        f"{BASE_URL}/predict",
        json=customer
    )

    assert response.status_code == 422