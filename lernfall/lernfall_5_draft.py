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
 
symbols = [symbol.strip() for symbol in ticker.split(",") if symbol.strip()]
stocks = [(symbol, load_prices(symbol, buy_date)) for symbol in symbols]
 
# --- Metrics -----------------------------------------------------------

results = []

for symbol, prices in stocks:
    total_return = prices.iloc[-1] / prices.iloc[0] - 1
    cagr_value = cagr(prices)
    volatility_value = volatility(prices)
    drawdown_value = max_drawdown(prices)
    risk_value = cagr_value / volatility_value

    st.metric(f"{symbol} total return", f"{total_return:.1%}")
    st.metric(f"{symbol} CAGR", f"{cagr_value:.1%}")
    st.metric(f"{symbol} volatility", f"{volatility_value:.1%}")
    st.metric(f"{symbol} max drawdown", f"{drawdown_value:.1%}")

    results.append({
        "Ticker": symbol,
        "Total return": total_return,
        "CAGR": cagr_value,
        "Volatility": volatility_value,
        "Max drawdown": drawdown_value,
        "Return/risk": risk_value,
    })

st.line_chart({
    symbol: prices / prices.iloc[0] * 100
    for symbol, prices in stocks
})

# TODO: export results to Excel
