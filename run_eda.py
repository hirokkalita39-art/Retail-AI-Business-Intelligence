from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "transactions_clean.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

def main():
    if not DATA.exists():
        raise FileNotFoundError("Run src/clean_data.py first.")
    df = pd.read_csv(DATA, parse_dates=["InvoiceDate"])
    df["Month"] = df["InvoiceDate"].dt.to_period("M").astype(str)

    monthly = df.groupby("Month", as_index=False).agg(
        Revenue=("Revenue", "sum"),
        Orders=("InvoiceNo", "nunique"),
        Units=("Quantity", "sum")
    )
    monthly.to_csv(OUT / "monthly_kpis.csv", index=False)

    top_products = df.groupby("Description", dropna=False)["Revenue"].sum().nlargest(15).sort_values()
    plt.figure(figsize=(10, 7))
    top_products.plot(kind="barh")
    plt.title("Top 15 Products by Revenue")
    plt.xlabel("Revenue (£)")
    plt.tight_layout()
    plt.savefig(OUT / "top_products.png", dpi=150)
    plt.close()

    plt.figure(figsize=(12, 5))
    sns.lineplot(data=monthly, x="Month", y="Revenue", marker="o")
    plt.xticks(rotation=60)
    plt.title("Monthly Revenue Trend")
    plt.tight_layout()
    plt.savefig(OUT / "monthly_revenue.png", dpi=150)
    plt.close()

    print("\nMonthly KPI sample:")
    print(monthly.head())
    print("\nTop 10 products:")
    print(top_products.tail(10).sort_values(ascending=False))
    print(f"\nCharts and monthly KPIs saved in {OUT}")

if __name__ == "__main__":
    main()
