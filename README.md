# 🛍️ Retail AI Business Intelligence & Demand Forecasting System

An end-to-end retail analytics and machine learning project designed to transform historical retail transaction data into actionable business insights.

The project combines **Python, SQL, Machine Learning and Power BI** to analyze sales performance, segment customers and evaluate revenue forecasting.

## 🎯 Project Objectives

- Analyze historical retail sales performance.
- Identify valuable and inactive customer segments.
- Understand monthly revenue and purchasing trends.
- Build a machine learning model for daily revenue forecasting.
- Present business insights through an interactive Power BI dashboard.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Data processing and analysis |
| Pandas, NumPy | Data cleaning and manipulation |
| Matplotlib | Data visualization |
| Scikit-learn | Machine learning |
| SQLite | Data storage and SQL analysis |
| Power BI | Interactive business intelligence dashboard |

## 📊 Project Modules

### 1. Data Cleaning and Preparation
- Loaded historical retail transaction data.
- Handled duplicate records and missing customer information.
- Processed cancellation transactions.
- Prepared cleaned data for analysis.

### 2. Exploratory Data Analysis
- Monthly revenue analysis.
- Order and unit performance analysis.
- Product sales exploration.
- Identification of business trends.

### 3. Customer Segmentation
Used RFM-style customer features and K-Means clustering to identify customer groups.

**Customer segments:**

| Metric | Segment 0 | Segment 1 |
|---|---:|---:|
| Customers | 3,558 | 2,320 |
| Average Revenue | £614.34 | £6,546.97 |
| Average Orders | 2.10 | 12.72 |
| Average Recency | 299.65 days | 50.56 days |

### 4. Revenue Forecasting
Developed a Random Forest regression model to predict daily revenue.

**Historical holdout evaluation:**

| Metric | Result |
|---|---:|
| MAE | £12,349.54 |
| RMSE | £23,255.87 |
| R² | 0.469 |

The model was evaluated using a chronological holdout of the final 60 eligible days. These results describe historical test performance, not guaranteed future forecasting accuracy.

### 5. Power BI Dashboard

The Power BI report presents:

- Executive Overview
- Customer Insights
- Product Analytics (planned)
- AI Revenue Forecasting
- Business Insights & Recommendations

## 💡 Business Recommendations

- Introduce loyalty programs for high-value customers.
- Use targeted campaigns to re-engage less-active customers.
- Monitor monthly revenue trends.
- Use revenue forecasts as supporting information for business planning.
- Improve future predictions by incorporating additional demand-related factors.

## 📁 Project Files

- `src/` – Python scripts
- `outputs/` – Analysis and model output files
- `models/` – Trained machine learning models
- `requirements.txt` – Python dependencies
- `retail_ai_business_intelligence.pbix` – Power BI report

## 🚀 How to Run

1. Clone this repository.
2. Install the required Python packages:

```bash
pip install -r requirements.txt
```

3. Obtain the Online Retail II dataset and place it in the expected raw data directory.
4. Run the data cleaning script:

```bash
python src/clean_data.py
```

5. Run the analysis and modeling scripts as required.

## 📚 Dataset

Dataset: Online Retail II

The dataset contains historical online retail transactions from 2009–2011.

Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii)

## 👨‍💻 Author

**B.Tech Computer Science Engineering Student**

Interested in Data Analytics, Machine Learning, SQL and Full-Stack Development.

---

*This project was developed as a practical application of data analytics, machine learning and business intelligence techniques.*
