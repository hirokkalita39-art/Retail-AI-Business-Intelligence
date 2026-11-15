from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "transactions_clean.csv"
OUT = ROOT / "outputs"
MODELS = ROOT / "models"
OUT.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

def main():
    if not DATA.exists():
        raise FileNotFoundError("Run src/clean_data.py first.")
    df = pd.read_csv(DATA, parse_dates=["InvoiceDate"])
    df = df.dropna(subset=["CustomerID"]).copy()

    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("Revenue", "sum")
    ).reset_index()

    # Log transform skewed positive values, then standardize.
    features = rfm[["Recency", "Frequency", "Monetary"]].clip(lower=0)
    X = np.log1p(features)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Compare a small range of cluster counts; avoid claiming K=4 is universally best.
    scores = []
    candidates = range(2, 7)
    for k in candidates:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)
        scores.append((k, silhouette_score(X_scaled, labels)))

    best_k = max(scores, key=lambda item: item[1])[0]
    final_model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    rfm["Cluster"] = final_model.fit_predict(X_scaled)

    summary = rfm.groupby("Cluster").agg(
        Customers=("CustomerID", "count"),
        AvgRecencyDays=("Recency", "mean"),
        AvgOrders=("Frequency", "mean"),
        AvgRevenue=("Monetary", "mean"),
        MedianRevenue=("Monetary", "median")
    ).round(2).reset_index()

    rfm.to_csv(OUT / "customer_segments.csv", index=False)
    summary.to_csv(OUT / "segment_summary.csv", index=False)
    joblib.dump({"model": final_model, "scaler": scaler}, MODELS / "rfm_kmeans.joblib")

    print("Silhouette scores:", scores)
    print("Selected cluster count:", best_k)
    print("\nInterpret clusters using their averages; cluster IDs are arbitrary:")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
