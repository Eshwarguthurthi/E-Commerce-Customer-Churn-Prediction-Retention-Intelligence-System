import pandas as pd
import numpy as np


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create customer behavior features for churn prediction.
    """

    data = df.copy()

    # Avoid division by zero
    tenure_safe = data["Tenure"].replace(0, np.nan)
    order_count_safe = data["OrderCount"].replace(0, np.nan)

    # 1. Order frequency relative to customer tenure
    data["OrderFrequency"] = (
        data["OrderCount"] / tenure_safe
    )

    # 2. Coupon usage relative to order count
    data["CouponUsageRate"] = (
        data["CouponUsed"] / order_count_safe
    )

    # 3. Customer recency category
    data["RecencyCategory"] = pd.cut(
        data["DaySinceLastOrder"],
        bins=[-np.inf, 7, 30, 60, np.inf],
        labels=[
            "0-7 days",
            "8-30 days",
            "31-60 days",
            "60+ days"
        ]
    )

    # 4. Registered devices relative to tenure
    data["DevicePerTenure"] = (
        data["NumberOfDeviceRegistered"] / tenure_safe
    )

    # Replace any values created by division
    data.replace([np.inf, -np.inf], np.nan, inplace=True)

    return data