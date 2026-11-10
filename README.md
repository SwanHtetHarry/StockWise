# StockWise AI

StockWise AI is a demand forecasting and inventory planning system designed to help retail businesses estimate future product demand and make better inventory decisions.

The project uses historical sales data to forecast demand for the next 7 days and then recommends how much inventory should be reordered based on current stock and safety stock.

---

## Problem

Retail businesses often face two common inventory problems:

- **Stockouts** — not having enough inventory to meet customer demand
- **Overstocking** — keeping more inventory than necessary

Both problems can lead to financial loss.

StockWise AI aims to support inventory planning by forecasting future demand and converting that forecast into a simple reorder recommendation.

---

## Main Features

### 7-Day Demand Forecast

StockWise predicts demand for the next seven days for a selected store and product.

The forecast uses:

- Historical sales
- Month
- Day of week
- Weekend information
- Lag features
- Rolling averages

---

### Inventory Recommendation

The system compares predicted demand with current inventory.

Example:

```text
7-Day Predicted Demand: 115 units
Current Stock: 100 units
Safety Stock: 6 units
Required Stock: 121 units

Recommended Reorder: 21 units
Status: Stockout Risk
```

---

### Stock Status

StockWise classifies the inventory situation into:

- Inventory Sufficient
- Low Stock
- Stockout Risk

---

### Historical Sales Visualization

Users can view recent historical sales for the selected product and store.

This helps compare previous demand patterns with the future forecast.

---

## Dataset

This project uses the **Store Item Demand Forecasting Challenge** dataset.

The training dataset contains approximately:

- 913,000 sales records
- 10 stores
- 50 products
- 5 years of daily sales history

Main columns:

```text
date   - Date of the sales record
store  - Store ID
item   - Product ID
sales  - Units sold
```

The dataset uses numeric store and product IDs instead of real business names.

---

## Exploratory Data Analysis

Before training the models, the sales data was analyzed for seasonal patterns.

The analysis showed both:

### Monthly Seasonality

Demand changes throughout the year, with some months showing consistently higher sales than others.

### Weekly Seasonality

Sales also vary depending on the day of the week.

These patterns were later included as machine learning features.

---

## Feature Engineering

The forecasting model uses several types of features.

### Calendar Features

```text
year
month
day
day_of_week
day_of_year
week_of_year
is_weekend
```

### Lag Features

Lag features represent previous sales values.

```text
lag_1   - sales 1 day ago
lag_7   - sales 7 days ago
lag_14  - sales 14 days ago
lag_28  - sales 28 days ago
```

### Rolling Features

```text
rolling_mean_7
rolling_mean_30
```

These represent average recent demand.

The rolling features were shifted before calculation so the model does not accidentally use future information.

---

## Preventing Data Leakage

Because this is a time-series forecasting problem, the data was not randomly split.

Instead, the model was trained using earlier dates and evaluated using later dates.

```text
2013 ---------------- Sep 2017 | Oct-Dec 2017
         Training               Validation
```

This better represents a real forecasting situation where future sales are unknown.

---

## Baseline Model

Before training machine learning models, a simple baseline was created.

The baseline assumes:

```text
Today's demand = demand from 7 days ago
```

This provides a benchmark that machine learning models must outperform.

Baseline performance:

| Metric | Score |
|---|---:|
| MAE | 9.076 |
| RMSE | 12.051 |

---

## Model Comparison

Three forecasting approaches were compared.

| Model | MAE | RMSE |
|---|---:|---:|
| 7-Day Baseline | 9.076 | 12.051 |
| Linear Regression | 7.088 | 9.311 |
| Gradient Boosting | **5.967** | **7.724** |

Lower values indicate better forecasting performance.

Gradient Boosting achieved the best results.

Compared with the baseline, Gradient Boosting reduced the average forecasting error significantly.

---

## Why Gradient Boosting?

Gradient Boosting builds multiple small decision trees sequentially.

Each new tree focuses on correcting errors made by the previous trees.

This allows the model to learn more complex relationships between:

- recent sales
- weekly patterns
- monthly seasonality
- product behavior
- store behavior

This performed better than Linear Regression for this dataset.

---

## Error Analysis

Model evaluation was not limited to MAE and RMSE.

Additional analysis included:

- Actual vs predicted sales visualization
- Largest forecasting errors
- Error by product
- Error by store
- Relative MAE

Relative MAE was useful because products have different sales volumes.

For example, a prediction error of 8 units may be small for a product selling 100 units per day but large for a product selling only 20 units per day.

---

## Recursive Forecasting

The final system generates a 7-day forecast recursively.

For the first future day, the model uses real historical sales.

For later days, previous predictions are added back into the sales history and used to generate the next prediction.

```text
Historical Sales
       ↓
Predict Day 1
       ↓
Use Day 1 Prediction
       ↓
Predict Day 2
       ↓
Use Day 2 Prediction
       ↓
Continue for 7 Days
```

This prevents the system from using future sales that would not be available in a real application.

---

## Inventory Logic

After generating the forecast, StockWise calculates inventory needs.

```text
Predicted Demand
       +
Safety Stock
       =
Required Stock
```

Then:

```text
Required Stock
       -
Current Stock
       =
Recommended Reorder
```

This transforms the machine learning prediction into a business decision.

---

## Streamlit Application

The Streamlit dashboard allows users to select:

- Store ID
- Product ID
- Current stock
- Safety stock

The application then displays:

- Average historical daily sales
- 7-day predicted demand
- Current inventory
- Required inventory
- Recommended reorder quantity
- Inventory status
- 7-day forecast chart
- Recent sales history

---

## Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- HistGradientBoostingRegressor
- Matplotlib
- Joblib
- Streamlit

---

## Project Structure

```text
StockWise-AI/
│
├── data/
│   └── train.csv
│
├── models/
│   ├── stockwise_gradient_boosting.pkl
│   └── feature_columns.pkl
│
├── notebooks/
│   └── 01_data_exploration.ipynb
│
├── src/
│   ├── __init__.py
│   ├── forecasting.py
│   └── inventory.py
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/SwanHtetHarry/StockWise-AI.git
cd StockWise-AI
```

Create a virtual environment:

```bash
python -m venv .venv
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the Application

Start the Streamlit dashboard:

```bash
streamlit run app.py
```

---

## Current Limitations

- Store and product names are represented only by numeric IDs.
- External factors such as promotions, holidays, pricing, and weather are not included.
- The current inventory recommendation uses a simplified safety-stock approach.
- Recursive forecasting can accumulate prediction errors over longer forecasting horizons.
- The system is designed as an AI/ML portfolio project rather than a production inventory management platform.

---

## Future Improvements

Possible improvements include:

- Dynamic safety stock based on forecast uncertainty
- Supplier lead-time integration
- Service-level based inventory planning
- XGBoost or LightGBM model comparison
- Holiday and promotion features
- Weather features
- Product category information
- Multi-store inventory optimization
- Longer forecasting horizons
- Automated low-stock alerts
- Database integration
- Real-time sales updates

---

## What I Learned

Through this project I practiced:

- Time-series forecasting
- Feature engineering
- Lag features
- Rolling statistics
- Seasonal analysis
- Data leakage prevention
- Time-based validation
- Baseline modeling
- Linear Regression
- Gradient Boosting
- MAE and RMSE
- Error analysis
- Recursive forecasting
- Inventory decision logic
- Model serialization
- Streamlit application development
- Building an end-to-end ML system

---

## Author

**Swan Htet (Harry Li)**

B.Sc. Information Technology Student  
AI / Machine Learning Project

GitHub:  
https://github.com/SwanHtetHarry