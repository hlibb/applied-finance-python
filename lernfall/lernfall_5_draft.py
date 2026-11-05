import streamlit as st
import yfinance as yf
from datetime import date
import numpy as np
 
TRADING_DAYS = 252         # used to annualise daily volatility

st.title("Portfolio Analyzer")
 
# --- Input -------------------------------------------------------------
# TODO: allow more than one stock (ticker, purchase date, amount)
ticker = st.sidebar.text_input("Ticker", "ALV.DE")
buy_date = st.sidebar.date_input("Purchase date", date(2021, 9, 30))
benchmark = st.sidebar.selectbox("Benchmark", ["^GDAXI", "URTH", "^STOXX50E"])

# NEW VARIABLES
end_date = date(2026, 11, 12)
 
 
# --- Data --------------------------------------------------------------
@st.cache_data
def load_prices(symbol, start):
    """Download adjusted closing prices (dividends included) from Yahoo Finance."""
    return yf.download(symbol, start=start, auto_adjust=True)["Close"].squeeze()


def cagr(prices):
    # Compound annual growth rate, as in Case Studies 1 and 2.
    years = (prices.index[-1] - prices.index[0]).days / 365.25
    return (prices.iloc[-1] / prices.iloc[0]) ** (1 / years) - 1


def volatility(prices, trading_days=TRADING_DAYS):
    # Annualised volatility: standard deviation of daily returns, scaled to one year.
    daily_returns = prices.pct_change().dropna()
    return daily_returns.std() * np.sqrt(trading_days)


def max_drawdown(prices):
    # Largest fall from a previous peak (Case Study 3).
    return (prices / prices.cummax() - 1).min() 
 
stock = load_prices(ticker, buy_date)
 
# --- Metrics -----------------------------------------------------------
# TODO: add the remaining metrics from case studies 2-4

total_return = stock.iloc[-1] / stock.iloc[0] - 1
st.metric("Total return", f"{total_return:.1%}")

# CAGR 
cagr = cagr(stock)
st.metric("CAGR", f"{cagr:.1%}")

# Volatility
volatility = volatility(stock)
st.metric("Volatility", f"{volatility:.1%}")

# Max Drawdown
max_drawdown = max_drawdown(stock)
st.metric("Max Drawdown", f"{max_drawdown:.1%}")

# --- Charts ------------------------------------------------------------
st.line_chart({ticker: stock / stock.iloc[0] * 100})
 
# TODO: export results to Excel

