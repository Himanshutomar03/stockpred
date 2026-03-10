# stock_predictor_app.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from pypfopt import EfficientFrontier, risk_models, expected_returns, plotting
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.set_page_config(layout="wide")
st.title("📈 Unified Stock Market Prediction App")

MODE = st.sidebar.selectbox("Select Mode", ["Breakout Strategy", "Portfolio Forecasting", "Intraday Prediction"])

# ---------- Breakout Strategy Module ----------
if MODE == "Breakout Strategy":
    st.header("🔍 Breakout Strategy with Support/Resistance")
    stocks = st.text_input("Enter Ticker Symbols (comma separated):", "AAPL, RELIANCE.NS, HAL.NS")
    stock_list = [s.strip().upper() for s in stocks.split(",") if s.strip()]
    n = st.slider("Extrema Order (n)", 5, 20, 10)

    for symbol in stock_list:
        st.subheader(f"📊 {symbol}")
        df = yf.download(symbol, period='1y', interval='1d', auto_adjust=True)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.dropna(inplace=True)

        df['resistance'] = df.iloc[argrelextrema(df['High'].values, np.greater_equal, order=n)[0]]['High']
        df['support'] = df.iloc[argrelextrema(df['Low'].values, np.less_equal, order=n)[0]]['Low']
        df['resistance'] = df['resistance'].ffill()
        df['support'] = df['support'].ffill()

        def breakout_strategy(row):
            if row['Close'] > row['resistance']:
                return 1  # Buy
            elif row['Close'] < row['support']:
                return -1  # Sell
            else:
                return 0  # Hold

        df['Signal'] = df.apply(breakout_strategy, axis=1)

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df['Close'], label='Close', linewidth=1.5)
        ax.plot(df['resistance'], label='Resistance', linestyle='--', color='red', alpha=0.5)
        ax.plot(df['support'], label='Support', linestyle='--', color='green', alpha=0.5)
        ax.scatter(df.index[df['Signal'] == 1], df['Close'][df['Signal'] == 1], color='green', marker='^', label='Buy')
        ax.scatter(df.index[df['Signal'] == -1], df['Close'][df['Signal'] == -1], color='red', marker='v', label='Sell')
        ax.set_title(f"Support/Resistance Breakout: {symbol}")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

# ---------- Portfolio Forecasting Module ----------
elif MODE == "Portfolio Forecasting":
    st.header("💹 XGBoost Return Forecast + Portfolio Optimization")
    tickers = st.text_input("Enter Tickers (comma separated):", "AAPL, HAL.NS").upper().split(',')
    capital = st.number_input("Enter Capital to Invest:", min_value=1000.0, value=10000.0)

    adj_close_df = pd.DataFrame()
    for ticker in tickers:
        df = yf.download(ticker.strip(), start="2020-01-01", end="2025-06-30", auto_adjust=True)
        adj_close_df[ticker.strip()] = df['Close']
    adj_close_df.ffill(inplace=True)

    def calculate_rsi(data, periods=14):
        delta = data.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=periods).mean()
        avg_loss = loss.rolling(window=periods).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    predicted_returns = {}
    for ticker in tickers:
        df = adj_close_df[[ticker]].copy()
        df['LogReturn'] = np.log(df[ticker] / df[ticker].shift(1))
        df['Lag1'] = df['LogReturn'].shift(1)
        df['Lag5'] = df['LogReturn'].shift(5)
        df['MA5'] = df[ticker].rolling(5).mean()
        df['MA10'] = df[ticker].rolling(10).mean()
        df['Vol10'] = df['LogReturn'].rolling(10).std()
        df['RSI'] = calculate_rsi(df[ticker], 14)
        df.dropna(inplace=True)
        X = df[['Lag1', 'Lag5', 'MA5', 'MA10', 'Vol10', 'RSI']]
        y = df['LogReturn']
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)
        model = XGBRegressor(n_estimators=100)
        model.fit(X_train, y_train)
        pred_return = model.predict(X.tail(1))[0] * 252
        predicted_returns[ticker] = pred_return

    mu = pd.Series(predicted_returns)
    S = risk_models.sample_cov(adj_close_df)
    ef = EfficientFrontier(mu, S)
    ef.max_sharpe()
    weights = ef.clean_weights()
    st.write("### Optimal Portfolio Weights", weights)
    st.write("Expected Return, Volatility, Sharpe Ratio:", ef.portfolio_performance(verbose=True))

    latest_prices = get_latest_prices(adj_close_df)
    da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=capital)
    alloc, leftover = da.greedy_portfolio()
    st.write("### Discrete Allocation", alloc)
    st.write("Funds Remaining:", leftover)

# ---------- Intraday Prediction Module ----------
elif MODE == "Intraday Prediction":
    st.header("🕒 Intraday Prediction with Random Forest")
    ticker = st.text_input("Enter Ticker:", "HAL.NS")
    df = yf.download(ticker, period='10d', interval='30m', auto_adjust=True)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.dropna(inplace=True)
    df['Target'] = df['Close'].shift(-1)
    df.dropna(inplace=True)

    X = df[['Open', 'High', 'Low', 'Volume']]
    y = df['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.write("### R² Score:", r2_score(y_test, y_pred))
    st.write("### MSE:", mean_squared_error(y_test, y_pred))

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(y_test.values, label='Actual', color='blue')
    ax.plot(y_pred, label='Predicted', color='red')
    ax.set_title(f"Intraday Price Prediction for {ticker}")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
