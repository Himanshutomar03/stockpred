# StockEdge Pro 

StockEdge Pro is an interactive Streamlit app for **stock analysis and portfolio optimization**.  
It combines technical analysis, machine learning models, and modern visualizations in a single dashboard.

# Deployment Link

https://stockpred-w.streamlit.app/

##  Features

### 1.  Breakout Strategy
- Support & resistance detection using **local extrema**.
- Automatic **BUY / SELL / HOLD** signals.
- Beautiful dark‑themed charts with:
  - Support / resistance bands
  - Buy / sell markers
  - Signal distribution summary.

### 2.  Portfolio Optimization
- Fetches historical data via **yfinance**.
- **XGBoost** models to estimate expected returns.
- **PyPortfolioOpt** efficient frontier:
  - Maximizes Sharpe ratio.
  - Shows optimal weights, discrete allocation (shares to buy), and remaining cash.
- Multiple plots:
  - Allocation pie chart\
  - Efficient frontier with optimal point
  - Historical portfolio performance
  - Weight distribution bar chart.

### 3. 🕒 Intraday Price Prediction
- Intraday data from **yfinance** (1m–1h intervals).
- **Random Forest** model to predict the next bar’s close price.
- Metrics: R², RMSE, MAE.
- Next‑bar **BUY / SELL** signal.
- Feature importance and recent price table.

>  **Disclaimer:** This app is for **educational purposes only** and is **not financial advice**.

---

##  Tech Stack

- [Streamlit](https://streamlit.io/)
- [yfinance](https://github.com/ranaroussi/yfinance)
- [pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [matplotlib](https://matplotlib.org/)
- [scikit‑learn](https://scikit-learn.org/)
- [XGBoost](https://xgboost.readthedocs.io/)
- [PyPortfolioOpt](https://pyportfolioopt.readthedocs.io/)
- [SciPy](https://scipy.org/)

---

##  Running Locally

```bash
# 1. Clone the repo
git clone https://github.com/Himanshutomar03/stockpred.git
cd stockpred

# 2. (Optional but recommended) Create a virtual environment
python -m venv venv
venv\Scripts\activate  # on Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
