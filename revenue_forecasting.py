from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "transactions_clean.csv"
OUT = ROOT / "outputs"
MODELS = ROOT / "models"
OUT.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

def make_features(daily):
    x = daily.copy()
    x["dayofweek"] = x["Date"].dt.dayofweek
    x["month"] = x["Date"].dt.month
    x["day"] = x["Date"].dt.day
    x["weekofyear"] = x["Date"].dt.isocalendar().week.astype(int)
    for lag in [1, 7, 14, 28]:
        x[f"lag_{lag}"] = x["Revenue"].shift(lag)
    # Shift before rolling to avoid including the target day's revenue.
    prior = x["Revenue"].shift(1)
    x["rolling_mean_7"] = prior.rolling(7).mean()
    x["rolling_mean_28"] = prior.rolling(28).mean()
    return x.dropna().copy()

def main():
    if not DATA.exists():
        raise FileNotFoundError("Run src/clean_data.py first.")
    df = pd.read_csv(DATA, parse_dates=["InvoiceDate"])
    daily = df.set_index("InvoiceDate").resample("D")["Revenue"].sum().rename("Revenue").reset_index()
    daily = daily.rename(columns={"InvoiceDate": "Date"}).sort_values("Date")
    # Include zero-sales days in the calendar.
    full_dates = pd.date_range(daily["Date"].min(), daily["Date"].max(), freq="D")
    daily = daily.set_index("Date").reindex(full_dates, fill_value=0).rename_axis("Date").reset_index()

    model_data = make_features(daily)
    feature_cols = [
        "dayofweek", "month", "day", "weekofyear",
        "lag_1", "lag_7", "lag_14", "lag_28",
        "rolling_mean_7", "rolling_mean_28"
    ]
    # Chronological holdout: last 60 eligible days are test data.
    split = len(model_data) - 60
    if split <= 0:
        raise ValueError("Not enough observations for a 60-day holdout.")
    train = model_data.iloc[:split]
    test = model_data.iloc[split:]

    model = RandomForestRegressor(
        n_estimators=250, max_depth=12, min_samples_leaf=2,
        random_state=42, n_jobs=-1
    )
    model.fit(train[feature_cols], train["Revenue"])
    pred = model.predict(test[feature_cols])

    rmse = mean_squared_error(test["Revenue"], pred) ** 0.5
    metrics = {
        "MAE": mean_absolute_error(test["Revenue"], pred),
        "RMSE": rmse,
        "R2": r2_score(test["Revenue"], pred)
    }
    result = test[["Date", "Revenue"]].copy()
    result["PredictedRevenue"] = pred
    result.to_csv(OUT / "revenue_forecast_holdout.csv", index=False)
    pd.DataFrame([metrics]).to_csv(OUT / "forecast_metrics.csv", index=False)
    joblib.dump({"model": model, "features": feature_cols}, MODELS / "daily_revenue_rf.joblib")

    print("Chronological holdout metrics (last 60 eligible days):")
    for key, value in metrics.items():
        print(f"{key}: {value:,.3f}")
    print("\nDo not report these numbers until this script has run successfully.")
    print("This is a one-step-ahead historical evaluation, not a validated live forecast.")

if __name__ == "__main__":
    main()
