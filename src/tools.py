"""This module defines the tools used by the agent.

Feel free to modify or add new tools to suit your specific needs.

To learn how to create a new tool, see:
- https://python.langchain.com/docs/concepts/tools/
- https://python.langchain.com/docs/how_to/#tools
"""

from __future__ import annotations

import yfinance as yf
import requests
import os
import pandas as pd
import numpy as np
import openai
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Dict, Union

ALPHAVANTAGE_KEY = os.getenv("ALPHAVANTAGE_KEY")

### FUNDAMENTAL ANALYSIS ###
class FinancialsInput(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol")

@tool("get_financials", args_schema=FinancialsInput, return_direct=True)
def get_financials(ticker: str) -> Union[Dict, str]:
    """Fetch fundamental financial data and ratios from Yahoo Finance."""
    try:
        stock = yf.Ticker(ticker)
        financials = stock.financials


        income_statement = {
            "Revenue": financials.loc["Total Revenue"].iloc[0] if "Total Revenue" in financials.index else None,  # ✅ Fix: Use `.iloc[0]`
            "Net Income": financials.loc["Net Income"].iloc[0] if "Net Income" in financials.index else None,  # ✅ Fix
            "EBITDA": financials.loc["EBITDA"].iloc[0] if "EBITDA" in financials.index else None,  # ✅ Fix
            "EPS": stock.info.get("trailingEps")
        }

        ratios = {
            "P/E Ratio": stock.info.get("trailingPE"),
            "P/B Ratio": stock.info.get("priceToBook"),
            "Debt-to-Equity": stock.info.get("debtToEquity"),
            "Return on Equity (ROE)": stock.info.get("returnOnEquity")
        }

        return {"income_statement": income_statement, "ratios": ratios}
    except Exception as e:
        return {"error": str(e)}

# SENTIMENT ANALYSIS 
@tool("get_sentiment_analysis", return_direct=True)
def get_stock_sentiment(ticker: str) -> Dict:
    """Fetch sentiment analysis from Alpha Vantage."""
    url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={ALPHAVANTAGE_KEY}'
    
    try:
        response = requests.get(url)
        data = response.json()
        if "feed" not in data:
            return {"error": "No sentiment data available"}

        sentiments = [item["ticker_sentiment"][0]["ticker_sentiment_label"] for item in data["feed"] if item["ticker_sentiment"]]
        sentiment_counts = {s: sentiments.count(s) for s in set(sentiments)}
        filtered_sentiment_counts = {key: value for key, value in sentiment_counts.items() if key != 'Neutral'}
        highest_sentiment = max(filtered_sentiment_counts, key=sentiment_counts.get)

        return {"highest_sentiment": highest_sentiment,"news_data": data['feed'][:5]}
    except Exception as e:
        return {"error": str(e)}

### TECHNICAL ANALYSIS ###
class TechnicalAnalysisInput(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol")

def Supertrend(df, atr_period: int, multiplier: float):
    """Calculates the Supertrend indicator."""
    high, low, close = df['High'], df['Low'], df['Close']
    hl2 = (high + low) / 2
    atr = df['High'].rolling(atr_period).std()
    
    final_upperband = hl2 + (multiplier * atr)
    final_lowerband = hl2 - (multiplier * atr)

    return pd.DataFrame({'Supertrend': close > final_upperband, 'Final Upperband': final_upperband, 'Final Lowerband': final_lowerband})

@tool("perform_technical_analysis", args_schema=TechnicalAnalysisInput, return_direct=True)
def perform_technical_analysis(ticker: str) -> Union[Dict, str]:
    """Performs advanced technical analysis using multiple indicators."""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1y", interval="1d")

        if data.empty:
            return {"error": "No data available for technical analysis"}

        technicals = {}
        close_prices = data['Close'].round(2)

        # Simple Moving Averages (SMA)
        technicals['SMA_50'] = close_prices.rolling(window=50).mean().iloc[-1]
        technicals['SMA_200'] = close_prices.rolling(window=200).mean().iloc[-1]

        # Relative Strength Index (RSI)
        delta = close_prices.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        technicals['RSI'] = rsi.iloc[-1] if not rsi.empty else None

        # MACD
        ema_12 = close_prices.ewm(span=12, adjust=False).mean()
        ema_26 = close_prices.ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9, adjust=False).mean()
        technicals['MACD'] = macd.iloc[-1]
        technicals['MACD_Signal'] = signal.iloc[-1]

        # Bollinger Bands
        rolling_mean = close_prices.rolling(window=20).mean()
        rolling_std = close_prices.rolling(window=20).std()
        technicals['Bollinger_Upper'] = (rolling_mean + 2 * rolling_std).iloc[-1]
        technicals['Bollinger_Lower'] = (rolling_mean - 2 * rolling_std).iloc[-1]

        # Supertrend Indicator
        atr_period = 7
        atr_multiplier = 3.0
        supertrend = Supertrend(data, atr_period, atr_multiplier)
        technicals['Supertrend'] = supertrend['Supertrend'].iloc[-1]

        return technicals
    except Exception as e:
        return {"error": str(e)}
