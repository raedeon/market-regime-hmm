import yfinance as yf
import pandas as pd
import numpy as np

def fetch_data(tickers, start_date, end_date):
    """
    Fetches data from Yahoo Finance and ensures correct column formatting.
    """
    # auto_adjust=True accounts for dividends/splits
    raw_data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)
    
    # Handle yfinance multi-index vs single-index issues
    if 'Close' in raw_data.columns:
        data = raw_data['Close']
    else:
        data = raw_data
        
    # Ensure alphabetical sorting so we know which column is which
    data = data.sort_index(axis=1)
    return data

def calculate_returns(data):
    """Computes Log Returns for stationarity."""
    return np.log(data / data.shift(1)).dropna()