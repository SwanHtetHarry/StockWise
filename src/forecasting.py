import pandas as pd
import numpy as np


def forecast_next_days(
    model,
    df,
    feature_columns,
    store_id,
    item_id,
    days=7
):
    history = (
        df[
            (df["store"] == store_id) &
            (df["item"] == item_id)
        ]
        .sort_values("date")
        [["date", "sales"]]
        .copy()
    )

    sales_history = (
        history["sales"]
        .astype(float)
        .tolist()
    )

    last_date = history["date"].max()

    forecasts = []

    for i in range(1, days + 1):

        future_date = (
            last_date
            + pd.Timedelta(days=i)
        )

        year = future_date.year
        month = future_date.month
        day = future_date.day

        day_of_week = future_date.dayofweek
        day_of_year = future_date.dayofyear

        week_of_year = int(
            future_date.isocalendar().week
        )

        is_weekend = int(
            day_of_week >= 5
        )

        lag_1 = sales_history[-1]
        lag_7 = sales_history[-7]
        lag_14 = sales_history[-14]
        lag_28 = sales_history[-28]

        rolling_mean_7 = np.mean(
            sales_history[-7:]
        )

        rolling_mean_30 = np.mean(
            sales_history[-30:]
        )

        future_features = pd.DataFrame(
            [{
                "store": store_id,
                "item": item_id,
                "year": year,
                "month": month,
                "day": day,
                "day_of_week": day_of_week,
                "day_of_year": day_of_year,
                "week_of_year": week_of_year,
                "is_weekend": is_weekend,
                "lag_1": lag_1,
                "lag_7": lag_7,
                "lag_14": lag_14,
                "lag_28": lag_28,
                "rolling_mean_7": rolling_mean_7,
                "rolling_mean_30": rolling_mean_30
            }]
        )

        future_features = future_features[
            feature_columns
        ]

        prediction = model.predict(
            future_features
        )[0]

        prediction = max(
            0,
            prediction
        )

        forecasts.append({
            "date": future_date,
            "store": store_id,
            "item": item_id,
            "predicted_sales": prediction
        })

        sales_history.append(
            prediction
        )

    return pd.DataFrame(
        forecasts
    )