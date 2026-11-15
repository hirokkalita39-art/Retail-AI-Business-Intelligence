from pathlib import Path
import sqlite3
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "processed" / "transactions_clean.csv"
DB = ROOT / "data" / "retail_analytics.sqlite"

def main():
    if not CSV.exists():
        raise FileNotFoundError("Run src/clean_data.py first.")
    DB.parent.mkdir(parents=True, exist_ok=True)
    # Load in chunks so SQLite creation is more memory-friendly.
    if DB.exists():
        DB.unlink()
    with sqlite3.connect(DB) as conn:
        first = True
        for chunk in pd.read_csv(CSV, chunksize=100_000, parse_dates=["InvoiceDate"]):
            chunk.to_sql("sales", conn, if_exists="replace" if first else "append", index=False)
            first = False
        conn.execute("CREATE INDEX idx_sales_date ON sales(InvoiceDate)")
        conn.execute("CREATE INDEX idx_sales_customer ON sales(CustomerID)")
        conn.execute("CREATE INDEX idx_sales_product ON sales(StockCode)")
    print(f"SQLite database created: {DB}")

if __name__ == "__main__":
    main()
