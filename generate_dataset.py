"""
=============================================================================
  RETAIL SALES DATASET GENERATOR
=============================================================================
  Generates 3+ years of realistic daily retail sales data with:
    - Upward growth trend
    - Monthly / weekly seasonality
    - Holiday spikes (Black Friday, Christmas, Back-to-School)
    - Weekend vs weekday patterns
    - Random noise
    - Multiple product categories

  Run this FIRST:  python -X utf8 generate_dataset.py
=============================================================================
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

# ---------- Configuration ----------
START_DATE = "2021-01-01"
END_DATE = "2024-03-31"       # ~3.25 years of data
CATEGORIES = ["Electronics", "Clothing", "Groceries", "Home & Garden"]


def generate_dataset():
    print("[*] Generating synthetic retail sales dataset...")

    dates = pd.date_range(start=START_DATE, end=END_DATE, freq="D")
    records = []

    for cat in CATEGORIES:
        # Base daily revenue for each category
        base = {"Electronics": 4200, "Clothing": 3100,
                "Groceries": 5500, "Home & Garden": 2800}[cat]

        for i, date in enumerate(dates):
            day_of_week = date.dayofweek          # 0=Mon ... 6=Sun
            month       = date.month
            day_of_year = date.dayofyear

            # 1) Growth trend: ~0.08% daily compound growth
            trend = base * (1 + 0.0008) ** i

            # 2) Monthly seasonality (peak in Nov-Dec, dip in Jan-Feb)
            month_factor = {
                1: 0.80, 2: 0.82, 3: 0.90, 4: 0.93, 5: 0.95, 6: 1.00,
                7: 0.97, 8: 1.05, 9: 1.02, 10: 1.00, 11: 1.20, 12: 1.35
            }[month]

            # 3) Day-of-week effect (weekends are higher for retail)
            dow_factor = [0.85, 0.82, 0.88, 0.92, 1.05, 1.25, 1.15][day_of_week]

            # 4) Holiday spikes
            holiday_factor = 1.0
            # Black Friday region (last week of November)
            if month == 11 and date.day >= 24 and date.day <= 30:
                holiday_factor = 1.6 if cat == "Electronics" else 1.35
            # Christmas week
            if month == 12 and date.day >= 20 and date.day <= 26:
                holiday_factor = 1.5
            # Back-to-School (August)
            if month == 8 and date.day >= 10 and date.day <= 25:
                holiday_factor = 1.15 if cat in ["Electronics", "Clothing"] else 1.05
            # Valentine's Day
            if month == 2 and date.day >= 12 and date.day <= 14:
                holiday_factor = 1.20

            # 5) Random noise (+/- 12%)
            noise = np.random.normal(1.0, 0.12)

            sales = trend * month_factor * dow_factor * holiday_factor * noise
            quantity = max(1, int(sales / (base * 0.015)))  # rough unit count

            records.append({
                "Date": date,
                "Category": cat,
                "Sales": round(max(0, sales), 2),
                "Quantity": quantity,
                "DayOfWeek": date.strftime("%A"),
                "Month": date.strftime("%B"),
                "Year": date.year,
            })

    df = pd.DataFrame(records)

    # Inject ~0.5% missing values for realism
    missing_idx = np.random.choice(df.index, size=int(len(df) * 0.005), replace=False)
    df.loc[missing_idx, "Sales"] = np.nan

    # Save
    out_path = os.path.join(os.path.dirname(__file__), "data", "retail_sales.csv")
    df.to_csv(out_path, index=False)
    print(f"[+] Dataset saved to {out_path}")
    print(f"    Shape : {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"    Range : {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"    Categories: {CATEGORIES}")
    return df


if __name__ == "__main__":
    generate_dataset()
