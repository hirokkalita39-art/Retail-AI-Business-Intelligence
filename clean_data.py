from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "online_retail_II.xlsx"
PROCESSED = ROOT / "data" / "processed"
EXPORTS = ROOT / "data" / "exports"

def main():
    if not RAW.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW}\nDownload Online Retail II from UCI and place it there."
        )

    PROCESSED.mkdir(parents=True, exist_ok=True)
    EXPORTS.mkdir(parents=True, exist_ok=True)

    # Read one worksheet at a time to reduce peak memory use.
    book = pd.ExcelFile(RAW)
    frames = []
    for sheet in book.sheet_names:
        part = pd.read_excel(RAW, sheet_name=sheet, engine="openpyxl")
        part.columns = [str(c).strip().replace(" ", "") for c in part.columns]
        frames.append(part)
        print(f"Read {sheet}: {len(part):,} rows")

    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={
        "Invoice": "InvoiceNo",
        "Price": "UnitPrice",
        "CustomerID": "CustomerID"
    })

    required = ["InvoiceNo", "StockCode", "Quantity", "InvoiceDate", "UnitPrice"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}. Found: {list(df.columns)}")

    df["InvoiceNo"] = df["InvoiceNo"].astype(str).str.strip()
    df["StockCode"] = df["StockCode"].astype(str).str.strip()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")

    # Keep an audit summary before filtering.
    audit = {
        "rows_loaded": len(df),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_customer_id": int(df["CustomerID"].isna().sum()) if "CustomerID" in df else None,
        "missing_invoice_date": int(df["InvoiceDate"].isna().sum()),
        "cancellation_rows": int(df["InvoiceNo"].str.upper().str.startswith("C").sum())
    }

    df = df.drop_duplicates().copy()
    df = df.dropna(subset=["InvoiceNo", "StockCode", "InvoiceDate", "Quantity", "UnitPrice"])
    df = df[~df["InvoiceNo"].str.upper().str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)].copy()

    if "Description" in df:
        df["Description"] = df["Description"].fillna("Unknown product")
    if "Country" in df:
        df["Country"] = df["Country"].fillna("Unknown")
    if "CustomerID" in df:
        df["CustomerID"] = df["CustomerID"].astype("string")

    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["OrderDate"] = df["InvoiceDate"].dt.date.astype(str)

    out = PROCESSED / "transactions_clean.csv"
    df.to_csv(out, index=False)
    (PROCESSED / "cleaning_audit.txt").write_text(
        "\n".join(f"{k}: {v}" for k, v in audit.items()) +
        f"\nrows_after_cleaning: {len(df)}\n",
        encoding="utf-8"
    )

    # Power BI-ready dimension and fact exports.
    products = df[["StockCode"] + (["Description"] if "Description" in df else [])].drop_duplicates("StockCode")
    customers = df[["CustomerID"] + (["Country"] if "Country" in df else [])]
    customers = customers.dropna(subset=["CustomerID"]).drop_duplicates("CustomerID")
    daily = df.assign(OrderDate=df["InvoiceDate"].dt.date).groupby("OrderDate", as_index=False).agg(
        Revenue=("Revenue", "sum"),
        UnitsSold=("Quantity", "sum"),
        Orders=("InvoiceNo", "nunique")
    )
    df.to_csv(EXPORTS / "fact_sales.csv", index=False)
    products.to_csv(EXPORTS / "dim_product.csv", index=False)
    customers.to_csv(EXPORTS / "dim_customer.csv", index=False)
    daily.to_csv(EXPORTS / "daily_sales.csv", index=False)

    print(f"\nCleaned rows: {len(df):,}")
    print(f"Saved: {out}")
    print("Audit:", audit)

if __name__ == "__main__":
    main()
