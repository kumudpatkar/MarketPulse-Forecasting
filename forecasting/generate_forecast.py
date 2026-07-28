import pandas as pd
import numpy as np
import os

def generate_future_forecast(df, months=6):

    df = df.copy()

    df["Date"] = pd.to_datetime(df["Date"])

    monthly = (
        df.groupby(pd.Grouper(key="Date", freq="MS"))["Sales"]
        .sum()
        .reset_index()
    )

    last_date = monthly["Date"].max()

    future_dates = pd.date_range(
        last_date + pd.offsets.MonthBegin(1),
        periods=months,
        freq="MS"
    )

    last_sales = monthly["Sales"].iloc[-1]

    growth = 0.03

    forecast = []

    for i, date in enumerate(future_dates):

        prediction = last_sales * ((1 + growth) ** (i + 1))

        forecast.append({
            "Date": date,
            "ForecastedSales": round(prediction, 2),
            "LowerBound": round(prediction * 0.95, 2),
            "UpperBound": round(prediction * 1.05, 2)
        })

    forecast_df = pd.DataFrame(forecast)

    os.makedirs(
        "powerbi_export",
        exist_ok=True
    )

    forecast_df.to_csv(
        "powerbi_export/forecast.csv",
        index=False
    )

    return forecast_df