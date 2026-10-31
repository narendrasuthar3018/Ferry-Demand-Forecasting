# Ferry Demand Forecasting & Intelligence Dashboard

An end-to-end machine learning project for analyzing historical ferry ticket activity and forecasting future ticket sales and redemptions using time-series data.

## Project Overview

Ferry Demand Forecasting is a data-driven analytics and forecasting system developed using Python, Machine Learning, and Streamlit.

The project analyzes historical Toronto Island Ferry ticket records to identify demand patterns, prepare time-series features, train forecasting models, evaluate predictions, and present insights through an interactive dashboard.

The system focuses on **ticket sales and ticket redemptions**. These should not be interpreted as verified passenger boarding counts or actual vessel occupancy.

## Key Features

* Historical ticket sales and redemption analysis.
* Dataset validation and data quality checks.
* Time-series preparation and feature engineering.
* Multiple forecasting horizons: 15, 30, 60, and 120 minutes.
* Machine learning model training and comparison.
* Baseline forecasting.
* Forecast uncertainty analysis.
* Demand insights and trend analysis.
* Pipeline monitoring and logging.
* Interactive Streamlit dashboard.
* Forecast reports and data exports.

## Technologies Used

* Python
* Streamlit
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Joblib

## Machine Learning Models

The project includes the following forecasting models:

1. Ridge Regression
2. Random Forest
3. Hist Gradient Boosting

The models are used to forecast ticket sales and redemptions across different forecasting horizons.

## Dataset

**Dataset:** Toronto Island Ferry Tickets

**Original columns:**

* `_id`
* `Timestamp`
* `Sales Count`
* `Redemption Count`

**Dataset details:**

* Records: 261,538
* Columns: 4
* Date range: May 2015 – December 2025

The dataset is used for historical demand analysis and ticket activity forecasting.

## Project Structure

```text
Ferry-Demand-Forecasting/
│
├── app.py
├── analyze_dataset.py
├── check_dataset.py
├── baseline_models.py
├── feature_engineering.py
├── forecasting_dataset.py
├── forecasting_engine.py
├── forecasting_models.py
├── forecast_uncertainty.py
├── model_evaluation.py
├── demand_insights.py
├── pipeline_monitor.py
├── time_series_preparation.py
│
├── data/
│   └── Toronto Island Ferry Tickets (1).csv
│
├── models/
│   └── Trained model files
│
├── outputs/
│   └── Generated reports and forecasting outputs
│
├── requirements.txt
└── README.md
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Open the Project Folder

```bash
cd Ferry-Demand-Forecasting
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Environment

**Windows:**

```bash
venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Dashboard

```bash
streamlit run app.py
```

The dashboard will open in your browser.

## Dashboard

The Streamlit dashboard provides an interactive interface for exploring historical ticket activity, forecasting results, model performance, demand insights, and generated reports.

## Project Objective

The main objective is to transform historical ferry ticket records into useful forecasting insights that can support demand analysis and data-driven planning.

## Future Scope

* Integration of additional operational and weather data.
* Improved forecasting accuracy through further model experimentation.
* Real-time data integration.
* More advanced forecasting and monitoring capabilities.
* Deployment and scalability improvements.

## Author

**Narendra Suthar**

B.Tech – Artificial Intelligence and Data Science

## Disclaimer

This project uses historical ferry ticket sales and redemption records. Forecasts represent ticket activity and should not be treated as direct measurements of passenger occupancy, vessel capacity, or actual boarding demand.
