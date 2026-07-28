"""
====================================================
AI Forecast Model Training Pipeline
====================================================
Models:
- Linear Regression
- Random Forest
- XGBoost
- SARIMAX
====================================================
"""

import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from statsmodels.tsa.statespace.sarimax import SARIMAX


# ====================================================
# MODEL PATH
# ====================================================

MODEL_DIR = "models"

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_model.pkl"
)


# ====================================================
# DATA PREPARATION
# ====================================================

def prepare_data(df):

    df = df.copy()

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day


    category_encoded = pd.get_dummies(
        df["Category"],
        drop_first=True
    )


    df = pd.concat(
        [
            df,
            category_encoded
        ],
        axis=1
    )


    features = [
        "Year",
        "Month",
        "Day"
    ] + list(category_encoded.columns)


    X = df[features]

    y = df["Sales"]


    return X, y



# ====================================================
# SARIMAX MODEL
# ====================================================

def train_sarimax(df):

    data = df.copy()

    data["Date"] = pd.to_datetime(
        data["Date"]
    )


    ts = (
        data.groupby("Date")["Sales"]
        .sum()
        .sort_index()
    )


    train_size = int(
        len(ts) * 0.8
    )


    train = ts[:train_size]

    test = ts[train_size:]


    model = SARIMAX(

        train,

        order=(1,1,1),

        seasonal_order=(1,1,1,7),

        enforce_stationarity=False,

        enforce_invertibility=False

    )


    fitted_model = model.fit(
        disp=False
    )


    forecast = fitted_model.forecast(
        steps=len(test)
    )


    mape = (

        mean_absolute_percentage_error(
            test,
            forecast
        )

        * 100

    )


    r2 = r2_score(
        test,
        forecast
    )


    return {

        "Model": "SARIMAX",

        "MAPE": round(
            mape,
            2
        ),

        "R2 Score": round(
            r2,
            3
        )

    }



# ====================================================
# TRAIN ALL MODELS
# ====================================================

def train_models(df):


    X, y = prepare_data(df)


    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.2,

        random_state=42

    )


    models = {

        "Linear Regression":
            LinearRegression(),


        "Random Forest":
            RandomForestRegressor(
                n_estimators=200,
                random_state=42
            ),


        "XGBoost":
            XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=5,
                random_state=42
            )

    }


    results = []


    for name, model in models.items():


        model.fit(
            X_train,
            y_train
        )


        # ===========================================
        # Feature Importance
        # ===========================================

        if name in [
            "Random Forest",
            "XGBoost"
        ]:

            importance = pd.DataFrame(
                {
                    "Feature": X.columns,
                    "Importance": model.feature_importances_
                }
            )


            importance = importance.sort_values(
                by="Importance",
                ascending=False
            )


            importance.to_csv(

                os.path.join(
                    MODEL_DIR,
                    f"{name.replace(' ','_')}_importance.csv"
                ),

                index=False
            )



        prediction = model.predict(
            X_test
        )


        mape = (

            mean_absolute_percentage_error(

                y_test,

                prediction

            )

            * 100

        )


        r2 = r2_score(

            y_test,

            prediction

        )


        results.append(

            {

                "Model": name,

                "MAPE": round(
                    mape,
                    2
                ),

                "R2 Score": round(
                    r2,
                    3
                )

            }

        )



    # ===========================================
    # Add SARIMAX
    # ===========================================

    sarimax_result = train_sarimax(df)


    results.append(
        sarimax_result
    )


    results_df = pd.DataFrame(
        results
    )



    # ===========================================
    # Save Best Model
    # ===========================================

    best_model_name = (

        results_df

        .sort_values(
            "MAPE"
        )

        .iloc[0]["Model"]

    )


    if best_model_name in models:


        best_model = models[
            best_model_name
        ]


        joblib.dump(

            best_model,

            MODEL_PATH

        )



    return results_df