from pydantic import BaseModel, Field


class CustomerInput(BaseModel):
    tenure: float = Field(..., ge=0)
    preferred_login_device: str
    city_tier: int = Field(..., ge=1)
    warehouse_to_home: float = Field(..., ge=0)
    preferred_payment_mode: str
    gender: str
    hour_spend_on_app: float = Field(..., ge=0)
    number_of_device_registered: int = Field(..., ge=0)
    prefered_order_cat: str
    satisfaction_score: int = Field(..., ge=1, le=5)
    marital_status: str
    number_of_address: int = Field(..., ge=0)
    complain: int = Field(..., ge=0, le=1)
    order_amount_hike_from_last_year: float = Field(..., ge=0)
    coupon_used: float = Field(..., ge=0)
    order_count: float = Field(..., ge=0)
    day_since_last_order: float = Field(..., ge=0)
    cashback_amount: float = Field(..., ge=0)


class PredictionResponse(BaseModel):
    prediction: int
    churn_status: str
    churn_probability: float