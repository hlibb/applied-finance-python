from datetime import date
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from matplotlib.ticker import PercentFormatter

TRADING_DAYS = 252         # used to annualise daily volatility

# Chart style, as in Case Study 4
plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
})

st.title("Portfolio Analyzer")


# --- Input -------------------------------------------------------------
# Default values = Ms. Keller's portfolio (Case Study 4)
keller_portfolio = [
    {"ticker": "ALV.DE",  "buy_date": date(2021, 9, 30)},
    {"ticker": "SAP.DE",  "buy_date": date(2022, 3, 15)},
    {"ticker": "IFX.DE",  "buy_date": date(2022, 6, 1)},
    {"ticker": "DTE.DE",  "buy_date": date(2023, 1, 3)},
    {"ticker": "MUV2.DE", "buy_date": date(2023, 9, 29)},
]

# The advisor chooses how many stocks the client has; one input block is shown per stock
number_of_stocks = st.sidebar.number_input("Number of stocks", min_value=1, max_value=10, value=len(keller_portfolio))

portfolio = []   # list with one dict per stock, same structure as in Case Study 4
for i in range(number_of_stocks):
    # Pre-fill the first 5 blocks with Ms. Keller's stocks, further blocks start empty
    if i < len(keller_portfolio):
        default = keller_portfolio[i]
    else:
        default = {"ticker": "", "buy_date": date(2021, 9, 30)}

    # One block of input fields per stock: ticker, purchase date, amount
    # (key=... gives every field a unique id, otherwise Streamlit cannot tell the blocks apart)
    st.sidebar.subheader(f"Stock {i + 1}")
    ticker = st.sidebar.text_input("Ticker", default["ticker"], key=f"ticker_{i}")
    buy_date = st.sidebar.date_input("Purchase date", default["buy_date"], key=f"buy_date_{i}")
    amount = st.sidebar.text_input("Amount (€)", "10000", key=f"amount_{i}")

    ticker = ticker.strip().upper()            # " alv.de " -> "ALV.DE"
    if ticker:                                 # empty fields are ignored
        portfolio.append({"ticker": ticker, "buy_date": str(buy_date), "amount": amount})

# Benchmark and evaluation date
st.sidebar.subheader("Comparison")
benchmark_names = {"^GDAXI": "DAX", "URTH": "MSCI World", "^STOXX50E": "Euro Stoxx 50"}   # Yahoo ticker -> name
benchmark_ticker = st.sidebar.selectbox("Benchmark", ["^GDAXI", "URTH", "^STOXX50E"])
benchmark = {"name": benchmark_names[benchmark_ticker], "ticker": benchmark_ticker}   # as in Case Study 4
end_date = st.sidebar.date_input("Evaluation date", date(2026, 9, 23))   # default = evaluation date of Case Study 4

# Plausibility checks of the input; st.stop() ends the script with a message instead of crashing
if not portfolio:
    st.info("Please enter at least one ticker in the sidebar.")
    st.stop()

for stock in portfolio:
    # Dates as "YYYY-MM-DD" text can be compared directly
    if stock["buy_date"] >= str(end_date):
        st.error(f"{stock['ticker']}: the purchase date must be before the evaluation date.")
        st.stop()

    # Amount: whole euros only, greater than 0 (letters, "1.000" or "10,5" are rejected)
    try:
        stock["amount"] = int(stock["amount"])
    except ValueError:
        st.error(f"{stock['ticker']}: please enter the amount as a whole number without dots or commas, e.g. 10000.")
        st.stop()
    if stock["amount"] <= 0:
        st.error(f"{stock['ticker']}: the amount must be greater than 0.")
        st.stop()


# --- Data --------------------------------------------------------------
# @st.cache_data: data is downloaded only once and reused when the advisor changes an input
@st.cache_data
def load_prices(symbol, start, end):
    """Download adjusted closing prices (dividends included) from Yahoo Finance."""
    end_exclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")  # yfinance: end is exclusive
    df = yf.download(symbol, start=start, end=end_exclusive, auto_adjust=True, progress=False)
    if df.empty:                       # unknown ticker -> Yahoo returns no data (check as in Case Study 2)
        raise ValueError(f"No prices found for '{symbol}'. Please check the ticker (e.g. ALV.DE for Allianz).")
    return df["Close"].squeeze()


@st.cache_data
def load_name(symbol):
    """Company name from Yahoo Finance (max. 20 characters); if it is not available, the ticker is used."""
    try:
        name = yf.Ticker(symbol).info.get("longName", symbol)
    except Exception:                  # Yahoo sometimes refuses the request -> show the ticker instead
        return symbol
    if len(name) > 20:
        name = name[:20] + "…"      # very long names would not fit into tables and charts
    return name


# All stocks plus the benchmark, from the earliest purchase date onwards (as in Case Study 4)
tickers = [s["ticker"] for s in portfolio] + [benchmark["ticker"]]
start = min(s["buy_date"] for s in portfolio)

# Load prices ticker by ticker; an unknown ticker shows a clear message instead of crashing
adj_close = {}
for ticker in tickers:
    try:
        adj_close[ticker] = load_prices(ticker, start, end_date)
    except ValueError as error:
        st.error(str(error))
        st.stop()
adj_close = pd.DataFrame(adj_close)      # one column per ticker, like in Case Study 4

# Company names for tables and charts
for stock in portfolio:
    stock["name"] = load_name(stock["ticker"])


# --- Metrics (functions from Case Study 4) -----------------------------
def stock_and_benchmark(adj_close, stock, benchmark):
    # Stock and benchmark on common trading days from the purchase date onwards.
    pair = adj_close[[stock["ticker"], benchmark["ticker"]]].loc[stock["buy_date"]:].dropna()
    pair.columns = ["stock", "benchmark"]
    return pair


def rebase(prices, base=100.0):
    # 100 € invested on the first day would be worth this much on each later day.
    return base * prices / prices.iloc[0]


def total_return(prices):
    return prices.iloc[-1] / prices.iloc[0] - 1


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


def performance_table(adj_close, portfolio, benchmark):
    b = benchmark["name"]
    rows = []
    for stock in portfolio:
        pair = stock_and_benchmark(adj_close, stock, benchmark)
        rows.append({
            "Stock": stock["name"],
            "Ticker": stock["ticker"],
            "Purchase": pair.index[0].date(),
            "Years": (pair.index[-1] - pair.index[0]).days / 365.25,
            "Total return": total_return(pair["stock"]),
            f"Total return {b}": total_return(pair["benchmark"]),
            "CAGR": cagr(pair["stock"]),
            f"CAGR {b}": cagr(pair["benchmark"]),
        })
    df = pd.DataFrame(rows).set_index("Stock")
    df[f"CAGR vs. {b}"] = df["CAGR"] - df[f"CAGR {b}"]   # in percentage points per year
    return df


def risk_table(adj_close, portfolio, benchmark):
    b = benchmark["name"]
    rows = []
    for stock in portfolio:
        pair = stock_and_benchmark(adj_close, stock, benchmark)
        rows.append({
            "Stock": stock["name"],
            "Ticker": stock["ticker"],
            "Volatility": volatility(pair["stock"]),
            f"Volatility {b}": volatility(pair["benchmark"]),
            "Max. drawdown": max_drawdown(pair["stock"]),
            f"Max. drawdown {b}": max_drawdown(pair["benchmark"]),
        })
    return pd.DataFrame(rows).set_index("Stock")


# Calculate both tables: one row per stock, stock vs. benchmark over the same period
perf = performance_table(adj_close, portfolio, benchmark)
risk = risk_table(adj_close, portfolio, benchmark)

# Summary table with the most important metrics per stock
summary = perf[["Ticker", "Total return", "CAGR"]].join(risk[["Volatility", "Max. drawdown"]])
# The one number for return AND risk: CAGR / volatility (Case Study 4, Task 4)
summary["Return/risk"] = summary["CAGR"] / summary["Volatility"]
# What the invested amount is worth today
for stock in portfolio:
    summary.loc[stock["name"], "Current value (€)"] = stock["amount"] * (1 + summary.loc[stock["name"], "Total return"])
# Best stock first
summary = summary.sort_values("Return/risk", ascending=False)

# Show the results (.style.format only changes the display, e.g. 1.706 -> +170.6%)
st.subheader("Results per stock")
st.success(f"Best stock (most return per unit of risk): {summary.index[0]}")
st.caption("Return/risk = CAGR divided by volatility. The higher, the more return for the risk taken.")
st.dataframe(summary.style.format({
    "Total return": "{:+.1%}", "CAGR": "{:+.1%}", "Volatility": "{:.1%}",
    "Max. drawdown": "{:.1%}", "Return/risk": "{:.2f}", "Current value (€)": "{:,.0f}",
}))

# Detailed tables from Case Study 4: each stock next to the benchmark
st.subheader(f"Comparison with the {benchmark['name']}")
st.dataframe(perf.style.format({"Years": "{:.1f}", **{c: "{:+.1%}" for c in perf.columns if c not in ("Ticker", "Purchase", "Years")}}))
st.dataframe(risk.style.format({c: "{:.1%}" for c in risk.columns if c != "Ticker"}))


# --- Charts ------------------------------------------------------------
# One colour per stock, as in Case Study 4
PALETTE = ["#1f4e79", "#e8a202", "#2e8b57", "#c0392b", "#7d3c98", "#5d6d7e", "#16a085", "#d35400"]
COLORS = {s["ticker"]: PALETTE[i % len(PALETTE)] for i, s in enumerate(portfolio)}


def label_points(ax, points, min_gap=14):
    # Label points to their right; push labels apart vertically if they would overlap (gap in pixels).
    fig = ax.figure
    fig.canvas.draw()
    to_px = ax.transData.transform
    placed = []   # (x_px, y_px) of labels already placed
    for x, y, text, color in sorted(points, key=lambda p: -p[1]):
        px, py = to_px((x, y))
        label_y = py
        for qx, qy in placed:
            if abs(px - qx) < 160 and abs(label_y - qy) < min_gap:
                label_y = qy - min_gap
        placed.append((px, label_y))
        dy_points = (label_y - py) * 72 / fig.dpi   # pixels -> points, independent of savefig dpi
        ax.annotate(text, xy=(x, y), xytext=(9, dy_points), textcoords="offset points",
                    va="center", fontsize=9, fontweight="bold", color=color)


def plot_risk_return(perf, risk, portfolio, benchmark, end_date):
    b = benchmark["name"]
    fig, ax = plt.subplots(figsize=(10, 6.5))

    for stock in portfolio:
        name, color = stock["name"], COLORS[stock["ticker"]]
        x_s, y_s = risk.loc[name, "Volatility"], perf.loc[name, "CAGR"]
        x_b, y_b = risk.loc[name, f"Volatility {b}"], perf.loc[name, f"CAGR {b}"]
        ax.plot([x_b, x_s], [y_b, y_s], color=color, linestyle=":", linewidth=1.2, zorder=1)
        ax.scatter(x_b, y_b, s=70, facecolors="white", edgecolors=color, linewidths=1.6, zorder=2)
        ax.scatter(x_s, y_s, s=110, color=color, zorder=3)

    # legend entries for the marker types
    ax.scatter([], [], s=90, color="black", label="Stock")
    ax.scatter([], [], s=60, facecolors="white", edgecolors="black", label=f"{b}, same period as the stock")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Risk: annualised volatility  →  more ups and downs")
    ax.set_ylabel("Return: CAGR per year  →  more growth")
    ax.set_xlim(0, max(risk["Volatility"].max(), risk[f"Volatility {b}"].max()) * 1.35)  # room for labels
    ax.margins(y=0.15)
    label_points(ax, [(risk.loc[s["name"], "Volatility"], perf.loc[s["name"], "CAGR"], s["name"],
                       COLORS[s["ticker"]]) for s in portfolio])
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.set_title("Return vs. risk: was the extra risk worth it?", loc="left", fontweight="bold")
    fig.text(0.01, 0.005, f"Source: Yahoo Finance, purchase date until {end_date}", fontsize=8, color="grey")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    return fig


# Performance since purchase (Case Study 3): every stock starts at 100 on its purchase date
st.subheader("Performance since purchase (purchase date = 100)")
performance = {}
for stock in portfolio:
    series = adj_close[stock["ticker"]].loc[stock["buy_date"]:].dropna()
    performance[stock["name"]] = rebase(series)
st.line_chart(pd.DataFrame(performance))

# Return vs. risk (Case Study 4): up = more return, right = more risk
st.subheader("Return vs. risk")
st.pyplot(plot_risk_return(perf, risk, portfolio, benchmark, end_date))


# --- Excel export ------------------------------------------------------
# Same number formats as in the tables above, written in Excel notation
PERCENT_SIGN = "+0.0%;-0.0%"     # e.g. +170.6%
PERCENT = "0.0%"                 # e.g. 19.3%


def write_sheet(writer, df, sheet_name, formats):
    # Write one table to Excel and give each column its number format.
    df.to_excel(writer, sheet_name=sheet_name)
    sheet = writer.sheets[sheet_name]
    for col_number, column in enumerate(df.columns, start=2):    # column A holds the stock names
        if column in formats:
            for row_number in range(2, len(df) + 2):             # row 1 holds the headers
                sheet.cell(row=row_number, column=col_number).number_format = formats[column]
    sheet.column_dimensions["A"].width = 20                      # room for the company names


# The Excel file is built in memory (BytesIO) and handed to the download button, no file is saved
buffer = BytesIO()
with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
    write_sheet(writer, summary, "Summary", {
        "Total return": PERCENT_SIGN, "CAGR": PERCENT_SIGN, "Volatility": PERCENT,
        "Max. drawdown": PERCENT, "Return/risk": "0.00", "Current value (€)": "#,##0",
    })
    write_sheet(writer, perf, "Performance",
                {"Years": "0.0", **{c: PERCENT_SIGN for c in perf.columns if c not in ("Ticker", "Purchase", "Years")}})
    write_sheet(writer, risk, "Risk", {c: PERCENT for c in risk.columns if c != "Ticker"})

st.download_button(
    "Download results as Excel",
    data=buffer.getvalue(),
    file_name="portfolio_analysis.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
