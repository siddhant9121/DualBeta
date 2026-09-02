import yfinance as yf

data = yf.download("AAPL", start="2024-01-01", end="2024-02-01")
print(data)
returns = data["Close"].pct_change()
print(returns)
