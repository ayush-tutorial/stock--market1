import os
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

sns.set_style("whitegrid")


def normalize_yfinance(df):
    """Normalize yfinance output across versions."""
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    df.columns = [str(col).lower().replace(" ", "_") for col in df.columns]
    return df


# Load Reliance dataset
stock = "RELIANCE.NS"
rel_data = yf.download(stock, start="2022-02-01", end="2023-04-23", auto_adjust=False, progress=False)
rel_data = normalize_yfinance(rel_data)
rel_data = rel_data.reset_index().rename(columns={"index": "Date"})
rel_data["Date"] = pd.to_datetime(rel_data["Date"])

print(rel_data.head())
print(rel_data.tail())
print(rel_data.shape)
print(rel_data.describe())
print(rel_data.dtypes)
print(rel_data.info())
print(rel_data.isnull().sum())
print(rel_data.corr())

rel_data["Returns"] = rel_data["adj_close"].pct_change()
print(rel_data.head())

plt.figure(figsize=(10, 5))
sns.lineplot(data=rel_data, x="Date", y="adj_close")
plt.title(f"{stock} Closing Price")
plt.show()

plt.figure(figsize=(10, 5))
sns.lineplot(data=rel_data, x="Date", y="Returns")
plt.title("Returns Distribution")
plt.xlabel("Returns")
plt.show()

rel_data["volume"].plot(legend=True, figsize=(10, 4))
plt.show()

# Load tech stocks using yfinance because IEX API is no longer available.
end = datetime.now()
start = datetime(end.year - 1, end.month, end.day)
tech_stocks = ["AAPL", "GOOG", "MSFT", "AMZN"]

for ticker in tech_stocks:
    df = yf.download(ticker, start=start, end=end + timedelta(days=1), auto_adjust=False, progress=False)
    df = normalize_yfinance(df)
    df = df.reset_index().rename(columns={"index": "Date"})
    df["Date"] = pd.to_datetime(df["Date"])
    globals()[ticker] = df

# 1. For Amazon
print(AMZN.head())
print(AMZN.shape)
print(AMZN.describe())

# 2. For Apple
print(AAPL.head())
print(AAPL.describe())

# 3. For Microsoft
print(MSFT.head())
print(MSFT.describe())

# 4. For Google
print(GOOG.head())
print(GOOG.shape)
print(GOOG.describe())
print(AAPL.columns)

# Historical closing price
AAPL["close"].plot(legend=True, figsize=(12, 6))
AMZN["close"].plot(legend=True, figsize=(12, 6))
GOOG["close"].plot(legend=True, figsize=(12, 6))
MSFT["close"].plot(legend=True, figsize=(12, 6))
plt.show()

AAPL["volume"].plot(legend=True, figsize=(12, 6))
plt.show()
GOOG["volume"].plot(legend=True, figsize=(12, 6))
plt.show()
AMZN["volume"].plot(legend=True, figsize=(12, 6))
plt.show()
MSFT["volume"].plot(legend=True, figsize=(12, 6))
plt.show()

# Moving average
ma_day = [10, 20, 50]
for ma in ma_day:
    col_name = f"MA for {ma} days"
    AAPL[col_name] = AAPL["close"].rolling(window=ma).mean()
    GOOG[col_name] = GOOG["close"].rolling(window=ma).mean()
    MSFT[col_name] = MSFT["close"].rolling(window=ma).mean()
    AMZN[col_name] = AMZN["close"].rolling(window=ma).mean()

print(AAPL.head(25))
AAPL[["close", "MA for 10 days", "MA for 20 days", "MA for 50 days"]].plot(subplots=False, figsize=(12, 6))
plt.show()

GOOG[["close", "MA for 10 days", "MA for 20 days", "MA for 50 days"]].plot(subplots=False, figsize=(12, 6))
plt.show()
MSFT[["close", "MA for 10 days", "MA for 20 days", "MA for 50 days"]].plot(subplots=False, figsize=(12, 6))
plt.show()
AMZN[["close", "MA for 10 days", "MA for 20 days", "MA for 50 days"]].plot(subplots=False, figsize=(12, 6))
plt.show()

# Daily return analysis
AAPL["Daily Return"] = AAPL["close"].pct_change()
print(AAPL.head(15))
AAPL["Daily Return"].plot(figsize=(12, 6), legend=True, linestyle="--", marker="o")
plt.show()

sns.histplot(AAPL["Daily Return"].dropna(), bins=100, color="blue", kde=True)
plt.show()

# Build closing price dataframe
closing_df1 = pd.DataFrame({"AAPL_close": AAPL["close"]})
closing_df2 = pd.DataFrame({"GOOG_close": GOOG["close"]})
closing_df3 = pd.DataFrame({"MSFT_close": MSFT["close"]})
closing_df4 = pd.DataFrame({"AMZN_close": AMZN["close"]})
closing_df = pd.concat([closing_df1, closing_df2, closing_df3, closing_df4], axis=1)
print(closing_df.head())

tech_returns = closing_df.pct_change()
print(tech_returns.head())

sns.jointplot(x="GOOG_close", y="GOOG_close", data=tech_returns, kind="scatter", color="seagreen")
plt.show()

sns.jointplot(x="GOOG_close", y="MSFT_close", data=tech_returns, kind="scatter", color="seagreen")
plt.show()

sns.pairplot(tech_returns.dropna())
plt.show()

corr = tech_returns.dropna().corr()
sns.heatmap(corr, annot=True)
plt.show()

print("Stock analysis completed successfully.")