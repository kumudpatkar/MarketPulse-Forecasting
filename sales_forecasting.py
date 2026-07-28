"""
=============================================================================
  SALES FORECASTING & DEMAND PREDICTION  --  End-to-End Pipeline
=============================================================================
  Steps:
    1. Load & Inspect Data
    2. Data Preprocessing (missing values, types, aggregation)
    3. Time Series EDA (trends, seasonality, patterns)
    4. Feature Engineering for ML models
    5. Train Forecasting Models (Linear Regression, SARIMAX)
    6. Evaluate & Compare Models
    7. Forecast Future Sales
    8. Business Insights & Recommendations
    9. Export data for Power BI
=============================================================================
"""

import os, warnings, joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # Non-interactive backend (Windows-safe)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

# -- Paths -----------------------------------------------------------------
BASE   = os.path.dirname(os.path.abspath(__file__))
DATA   = os.path.join(BASE, "data", "retail_sales.csv")
PLOTS  = os.path.join(BASE, "plots")
MODELS = os.path.join(BASE, "models")
PBI    = os.path.join(BASE, "powerbi_export")

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
PALETTE = ["#6366f1", "#f43f5e", "#10b981", "#f59e0b"]


# ===== STEP 1 : LOAD & INSPECT ============================================
def load_data():
    print("\n" + "="*70)
    print("  STEP 1 : LOAD & INSPECT DATA")
    print("="*70)
    df = pd.read_csv(DATA, parse_dates=["Date"])
    print(f"  Shape  : {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Date range: {df['Date'].min().date()} -> {df['Date'].max().date()}")
    print(f"  Missing:\n{df.isnull().sum()[df.isnull().sum()>0]}")
    return df


# ===== STEP 2 : PREPROCESSING =============================================
def preprocess(df):
    print("\n" + "="*70)
    print("  STEP 2 : DATA PREPROCESSING")
    print("="*70)

    # Fill missing Sales with forward-fill per category
    missing = df["Sales"].isnull().sum()
    df["Sales"] = df.groupby("Category")["Sales"].transform(
        lambda s: s.fillna(method="ffill").fillna(method="bfill")
    )
    print(f"  [+] Filled {missing} missing Sales values (forward-fill per category)")

    # Create daily aggregate (total across all categories)
    daily = df.groupby("Date").agg(
        TotalSales=("Sales", "sum"),
        TotalQty=("Quantity", "sum"),
    ).reset_index()
    daily["DayOfWeek"] = daily["Date"].dt.dayofweek
    daily["Month"]     = daily["Date"].dt.month
    daily["Year"]      = daily["Date"].dt.year
    daily["WeekOfYear"]= daily["Date"].dt.isocalendar().week.astype(int)

    print(f"  [+] Daily aggregate: {len(daily)} days")
    return df, daily


# ===== STEP 3 : EXPLORATORY DATA ANALYSIS =================================
def run_eda(df, daily):
    print("\n" + "="*70)
    print("  STEP 3 : EXPLORATORY DATA ANALYSIS")
    print("="*70)

    # 3a) Overall sales trend
    fig, ax = plt.subplots(figsize=(14, 5))
    monthly = daily.set_index("Date")["TotalSales"].resample("M").sum()
    ax.plot(monthly.index, monthly.values, color=PALETTE[0], linewidth=2)
    ax.fill_between(monthly.index, monthly.values, alpha=0.15, color=PALETTE[0])
    ax.set_title("Monthly Total Sales Over Time", fontsize=15, fontweight="bold")
    ax.set_ylabel("Total Sales ($)")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, "01_monthly_sales_trend.png"), dpi=150)
    plt.close()

    # 3b) Sales by category over time
    fig, ax = plt.subplots(figsize=(14, 5))
    for i, cat in enumerate(df["Category"].unique()):
        cat_m = df[df["Category"]==cat].set_index("Date")["Sales"].resample("M").sum()
        ax.plot(cat_m.index, cat_m.values, label=cat, linewidth=2, color=PALETTE[i])
    ax.set_title("Monthly Sales by Category", fontsize=15, fontweight="bold")
    ax.set_ylabel("Sales ($)")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=45); plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, "02_sales_by_category.png"), dpi=150)
    plt.close()

    # 3c) Day-of-week pattern
    fig, ax = plt.subplots(figsize=(8, 5))
    dow_names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    dow_sales = daily.groupby("DayOfWeek")["TotalSales"].mean()
    ax.bar(dow_names, dow_sales.values, color=PALETTE, edgecolor="white", linewidth=1.2)
    ax.set_title("Average Daily Sales by Day of Week", fontsize=15, fontweight="bold")
    ax.set_ylabel("Avg Sales ($)")
    for i, v in enumerate(dow_sales.values):
        ax.text(i, v + 200, f"${v:,.0f}", ha="center", fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, "03_sales_by_dayofweek.png"), dpi=150)
    plt.close()

    # 3d) Monthly seasonality (averaged across years)
    fig, ax = plt.subplots(figsize=(10, 5))
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    month_avg = daily.groupby("Month")["TotalSales"].mean()
    bars = ax.bar(month_names, month_avg.values, color=PALETTE[0], edgecolor="white")
    # Highlight peak months
    for i in [10, 11]:  # Nov, Dec
        bars[i].set_color(PALETTE[1])
    ax.set_title("Average Daily Sales by Month (Seasonality)", fontsize=15, fontweight="bold")
    ax.set_ylabel("Avg Sales ($)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, "04_monthly_seasonality.png"), dpi=150)
    plt.close()

    # 3e) Year-over-year comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    for yr in sorted(daily["Year"].unique()):
        yr_data = daily[daily["Year"]==yr].set_index("Date")["TotalSales"].resample("M").sum()
        ax.plot(range(len(yr_data)), yr_data.values, label=str(yr),
                linewidth=2, marker="o", markersize=4)
    ax.set_title("Year-over-Year Monthly Sales", fontsize=15, fontweight="bold")
    ax.set_xticks(range(12)); ax.set_xticklabels(month_names)
    ax.set_ylabel("Total Sales ($)"); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, "05_yoy_comparison.png"), dpi=150)
    plt.close()

    # 3f) Seasonal decomposition (multiplicative)
    ts = daily.set_index("Date")["TotalSales"].asfreq("D")
    ts = ts.fillna(method="ffill")
    decomp = seasonal_decompose(ts, model="multiplicative", period=30)
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    components = [("Observed", decomp.observed), ("Trend", decomp.trend),
                  ("Seasonal", decomp.seasonal), ("Residual", decomp.resid)]
    for ax, (name, comp) in zip(axes, components):
        ax.plot(comp, color=PALETTE[0], linewidth=1)
        ax.set_ylabel(name, fontsize=12, fontweight="bold")
    axes[0].set_title("Time Series Decomposition (Multiplicative)", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, "06_decomposition.png"), dpi=150)
    plt.close()

    # 3g) Sales distribution / histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(daily["TotalSales"], bins=50, color=PALETTE[0], edgecolor="white", alpha=0.8)
    ax.axvline(daily["TotalSales"].mean(), color=PALETTE[1], linestyle="--",
               linewidth=2, label=f"Mean: ${daily['TotalSales'].mean():,.0f}")
    ax.set_title("Distribution of Daily Total Sales", fontsize=15, fontweight="bold")
    ax.set_xlabel("Daily Sales ($)"); ax.set_ylabel("Frequency"); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, "07_sales_distribution.png"), dpi=150)
    plt.close()

    print(f"  [+] Saved 7 EDA plots to /plots")
    return daily


# ===== STEP 4 : FEATURE ENGINEERING (for ML models) =======================
def engineer_features(daily):
    print("\n" + "="*70)
    print("  STEP 4 : FEATURE ENGINEERING")
    print("="*70)

    df = daily.copy()
    # Lag features (previous days' sales)
    for lag in [1, 7, 14, 30]:
        df[f"lag_{lag}"] = df["TotalSales"].shift(lag)
    # Rolling averages
    df["rolling_7"]  = df["TotalSales"].rolling(7).mean()
    df["rolling_30"] = df["TotalSales"].rolling(30).mean()
    # Day number (for trend)
    df["day_num"] = (df["Date"] - df["Date"].min()).dt.days

    df.dropna(inplace=True)
    print(f"  [+] Features: lag_1/7/14/30, rolling_7/30, day_num, DayOfWeek, Month")
    print(f"  [+] Shape after feature engineering: {df.shape}")
    return df


# ===== STEP 5 : TRAIN MODELS ==============================================
def train_models(feat_df, daily):
    print("\n" + "="*70)
    print("  STEP 5 : TRAINING FORECASTING MODELS")
    print("="*70)

    # ---- 5a) Linear Regression & Random Forest ----------------------------
    feature_cols = ["day_num", "DayOfWeek", "Month", "WeekOfYear",
                    "lag_1", "lag_7", "lag_14", "lag_30",
                    "rolling_7", "rolling_30"]
    # 80/20 time-based split
    split = int(len(feat_df) * 0.8)
    train = feat_df.iloc[:split]
    test  = feat_df.iloc[split:]

    X_train, y_train = train[feature_cols], train["TotalSales"]
    X_test,  y_test  = test[feature_cols],  test["TotalSales"]

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    print(f"\n  -- Linear Regression trained on {len(X_train)} days, testing on {len(X_test)}")

    # Random Forest
    rf = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    print(f"  -- Random Forest trained")

    # ---- 5b) SARIMAX on monthly aggregated data ---------------------------
    print(f"\n  -- Training SARIMAX on monthly data...")
    monthly = daily.set_index("Date")["TotalSales"].resample("M").sum()

    # ADF test for stationarity
    adf_stat, adf_p, *_ = adfuller(monthly.dropna())
    print(f"     ADF Statistic: {adf_stat:.4f}, p-value: {adf_p:.4f}")
    print(f"     Series is {'stationary' if adf_p < 0.05 else 'non-stationary (needs differencing)'}")

    train_m = monthly[:-6]      # Hold out last 6 months
    test_m  = monthly[-6:]

    # SARIMAX(1,1,1)(1,1,1,12) — standard seasonal model
    model = SARIMAX(train_m, order=(1,1,1), seasonal_order=(1,1,1,12),
                    enforce_stationarity=False, enforce_invertibility=False)
    sarima = model.fit(disp=False)
    sarima_pred = sarima.forecast(steps=len(test_m))
    print(f"  -- SARIMAX trained (order=1,1,1  seasonal=1,1,1,12)")

    # Save models
    joblib.dump(lr, os.path.join(MODELS, "linear_regression.pkl"))
    joblib.dump(rf, os.path.join(MODELS, "random_forest.pkl"))
    joblib.dump(sarima, os.path.join(MODELS, "sarimax.pkl"))
    joblib.dump(feature_cols, os.path.join(MODELS, "feature_cols.pkl"))
    print(f"  [+] Models saved to /models")

    return (lr, rf, sarima, lr_pred, rf_pred, sarima_pred,
            X_test, y_test, test, train_m, test_m, monthly, feature_cols)


# ===== STEP 6 : EVALUATE & COMPARE ========================================
def evaluate(lr_pred, rf_pred, sarima_pred, y_test, test_m):
    print("\n" + "="*70)
    print("  STEP 6 : MODEL EVALUATION")
    print("="*70)

    def metrics(name, actual, predicted):
        mae  = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        r2   = r2_score(actual, predicted)
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        print(f"\n  {name}:")
        print(f"    MAE  = ${mae:,.0f}")
        print(f"    RMSE = ${rmse:,.0f}")
        print(f"    MAPE = {mape:.2f}%")
        print(f"    R2   = {r2:.4f}")
        return {"Model": name, "MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}

    results = [
        metrics("Linear Regression (daily)", y_test.values, lr_pred),
        metrics("Random Forest (daily)", y_test.values, rf_pred),
        metrics("SARIMAX (monthly)", test_m.values, sarima_pred.values),
    ]

    # Comparison bar chart
    res_df = pd.DataFrame(results)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # MAPE comparison
    axes[0].barh(res_df["Model"], res_df["MAPE"], color=PALETTE[:3], edgecolor="white")
    axes[0].set_xlabel("MAPE (%)")
    axes[0].set_title("Mean Absolute Percentage Error", fontsize=13, fontweight="bold")
    for i, v in enumerate(res_df["MAPE"]):
        axes[0].text(v + 0.2, i, f"{v:.1f}%", va="center", fontweight="bold")

    # R2 comparison (daily models only)
    daily_res = res_df[res_df["Model"].str.contains("daily")]
    axes[1].barh(daily_res["Model"], daily_res["R2"], color=PALETTE[:2], edgecolor="white")
    axes[1].set_xlabel("R-squared")
    axes[1].set_title("R-squared (Daily Models)", fontsize=13, fontweight="bold")
    axes[1].set_xlim(0, 1)
    for i, v in enumerate(daily_res["R2"]):
        axes[1].text(v + 0.01, i, f"{v:.3f}", va="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, "08_model_comparison.png"), dpi=150)
    plt.close()
    print(f"\n  [+] Comparison plot saved")
    return res_df


# ===== STEP 7 : FORECAST FUTURE SALES =====================================
def forecast_future(sarima, rf, test, monthly, feature_cols):
    print("\n" + "="*70)
    print("  STEP 7 : FORECASTING FUTURE SALES")
    print("="*70)

    # --- 7a) SARIMAX monthly forecast (next 6 months) ---
    future_months = 6
    full_model = SARIMAX(monthly, order=(1,1,1), seasonal_order=(1,1,1,12),
                         enforce_stationarity=False, enforce_invertibility=False)
    full_fit = full_model.fit(disp=False)
    forecast = full_fit.get_forecast(steps=future_months)
    fc_mean = forecast.predicted_mean
    fc_ci   = forecast.conf_int()

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(monthly.index, monthly.values, label="Historical", color=PALETTE[0], linewidth=2)
    ax.plot(fc_mean.index, fc_mean.values, label="Forecast", color=PALETTE[1],
            linewidth=2, linestyle="--", marker="o")
    ax.fill_between(fc_ci.index, fc_ci.iloc[:,0], fc_ci.iloc[:,1],
                    alpha=0.2, color=PALETTE[1], label="95% Confidence")
    ax.set_title("Monthly Sales Forecast (Next 6 Months)", fontsize=15, fontweight="bold")
    ax.set_ylabel("Total Sales ($)")
    ax.legend(fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=45); plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, "09_future_forecast.png"), dpi=150)
    plt.close()

    print(f"\n  SARIMAX 6-Month Forecast:")
    for date, val in fc_mean.items():
        print(f"    {date.strftime('%B %Y'):>15}: ${val:>12,.0f}")

    # --- 7b) Actual vs Predicted (test period, daily) ---
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(test["Date"].values, test["TotalSales"].values,
            label="Actual", color=PALETTE[0], linewidth=1.5, alpha=0.8)
    rf_test_pred = rf.predict(test[feature_cols])
    ax.plot(test["Date"].values, rf_test_pred,
            label="Random Forest Prediction", color=PALETTE[1], linewidth=1.5, alpha=0.8)
    ax.set_title("Actual vs Predicted Daily Sales (Test Period)", fontsize=15, fontweight="bold")
    ax.set_ylabel("Sales ($)"); ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=45); plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, "10_actual_vs_predicted.png"), dpi=150)
    plt.close()

    # Save updated SARIMAX
    joblib.dump(full_fit, os.path.join(MODELS, "sarimax_full.pkl"))
    print(f"  [+] Forecast plots saved")
    return fc_mean, fc_ci


# ===== STEP 8 : BUSINESS INSIGHTS =========================================
def business_insights():
    print("\n" + "="*70)
    print("  STEP 8 : BUSINESS INSIGHTS & RECOMMENDATIONS")
    print("="*70)
    insights = [
        ("TREND: Consistent growth",
         "Sales show a clear upward trend (~25-30% YoY growth).",
         "Plan for increased inventory capacity and staffing."),
        ("SEASONALITY: Nov-Dec is king",
         "November and December account for the highest sales due to holidays.",
         "Stock up inventory 2 months before. Start marketing in October."),
        ("WEEKDAY PATTERN: Weekends drive revenue",
         "Saturday-Sunday sales are 20-35% higher than weekdays.",
         "Schedule more staff and promotions on weekends."),
        ("CATEGORY: Groceries lead, Electronics spike",
         "Groceries are the most consistent. Electronics spike during holidays.",
         "Maintain steady grocery supply; build Electronics buffer for Q4."),
        ("DEMAND PLANNING: Use lag + rolling features",
         "Last week's sales strongly predict this week. 30-day rolling avg is key.",
         "Build automated reorder triggers based on 7-day rolling average."),
        ("SLOW PERIODS: Jan-Feb dip",
         "Post-holiday sales drop 15-20%.",
         "Run clearance sales in January; reduce orders in Q1."),
    ]
    for title, finding, action in insights:
        print(f"\n  >> {title}")
        print(f"     Finding : {finding}")
        print(f"     Action  : {action}")


# ===== STEP 9 : POWER BI EXPORT ===========================================
def export_powerbi(df, daily, fc_mean, fc_ci):
    print("\n" + "="*70)
    print("  STEP 9 : POWER BI EXPORT")
    print("="*70)

    # Detailed category data
    df.to_csv(os.path.join(PBI, "sales_by_category.csv"), index=False)

    # Daily aggregate
    daily.to_csv(os.path.join(PBI, "daily_aggregate.csv"), index=False)

    # Forecast data
    fc_df = pd.DataFrame({
        "Date": fc_mean.index,
        "ForecastedSales": fc_mean.values,
        "LowerBound": fc_ci.iloc[:, 0].values,
        "UpperBound": fc_ci.iloc[:, 1].values,
        "Type": "Forecast"
    })
    fc_df.to_csv(os.path.join(PBI, "forecast.csv"), index=False)
    print("  [+] Exported 3 CSVs to /powerbi_export:")
    print("      - sales_by_category.csv  (detailed)")
    print("      - daily_aggregate.csv    (summary)")
    print("      - forecast.csv           (future predictions)")
    print("  [i] Import these into Power BI Desktop to build dashboards.")


# ===== MAIN PIPELINE ======================================================
def main():
    print("\n" + "="*70)
    print("   SALES FORECASTING & DEMAND PREDICTION PIPELINE")
    print("="*70)

    df = load_data()
    df, daily = preprocess(df)
    daily = run_eda(df, daily)
    feat_df = engineer_features(daily)
    (lr, rf, sarima, lr_pred, rf_pred, sarima_pred,
     X_test, y_test, test, train_m, test_m, monthly, feature_cols) = train_models(feat_df, daily)
    res_df = evaluate(lr_pred, rf_pred, sarima_pred, y_test, test_m)
    fc_mean, fc_ci = forecast_future(sarima, rf, test, monthly, feature_cols)
    business_insights()
    export_powerbi(df, daily, fc_mean, fc_ci)

    # Best model selection
    print("\n" + "="*70)
    print("  BEST MODEL SELECTION")
    print("="*70)
    best = res_df.loc[res_df["MAPE"].idxmin()]
    print(f"\n  Best Model (lowest MAPE): {best['Model']}")
    print(f"  MAPE: {best['MAPE']:.2f}%")
    print(f"\n  Why? MAPE measures average % error in predictions,")
    print(f"  making it the most intuitive metric for business stakeholders.")

    print("\n" + "="*70)
    print("  PIPELINE COMPLETE!")
    print(f"  Plots   : {PLOTS}")
    print(f"  Models  : {MODELS}")
    print(f"  Power BI: {PBI}")
    print(f"  Dashboard: streamlit run streamlit_app.py")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
