import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.data_loader import fetch_data, calculate_returns
from src.hmm_model import train_hmm
from src.optimizer import get_optimal_weights

# --- CONFIGURATION ---
TICKERS = ['SPY', 'TLT', 'GLD']
START_DATE = '2005-01-01'
END_DATE = '2024-01-01'
REBALANCE_FREQ = 20  # Days (Approx 1 month)
LOOKBACK_WINDOW = 50 # Days for Covariance estimation

def run_pipeline():
    # 1. Data Ingestion
    print(f"1. Fetching data for {TICKERS}...")
    data = fetch_data(TICKERS, START_DATE, END_DATE)
    returns = calculate_returns(data)
    
    # 2. AI Regime Detection
    print("2. Training HMM Regime Detector...")
    # Feature Engineering: Use SPY Returns & Volatility
    spy_col = 'SPY' # Assumes SPY is in the list
    if spy_col not in data.columns:
        # Fallback if column names got weird
        spy_col = data.columns[0]
    
    spy_ret = returns[spy_col].values.reshape(-1, 1)
    spy_vol = returns[spy_col].rolling(10).std().dropna().values.reshape(-1, 1)
    
    # Align data (drop NaN from rolling window)
    trunc_len = len(spy_vol)
    spy_ret = spy_ret[-trunc_len:]
    
    X = np.column_stack([spy_ret, spy_vol])
    
    # Train Model
    model, hidden_states = train_hmm(X, n_states=3)
    
    # Map States to Regimes based on Volatility
    analysis_df = pd.DataFrame(X, columns=['Ret', 'Vol'])
    analysis_df['State'] = hidden_states
    vol_by_state = analysis_df.groupby('State')['Vol'].mean()
    sorted_idx = vol_by_state.sort_values().index
    
    state_map = {
        sorted_idx[0]: 'BULL',   # Lowest Vol
        sorted_idx[1]: 'CHOP',   # Medium Vol
        sorted_idx[2]: 'BEAR'    # Highest Vol
    }
    print(f"   State Mapping: {state_map}")

    # 3. Backtesting Loop
    print("3. Running Backtest Simulation...")
    portfolio_values = [10000] # Start with $10k
    current_weights = np.ones(len(TICKERS)) / len(TICKERS)
    
    # Align backtest data with HMM states
    backtest_data = returns.iloc[-trunc_len:]
    dates = backtest_data.index
    
    for t in range(len(backtest_data)):
        # Calculate daily PnL
        day_ret = backtest_data.iloc[t].values
        port_ret = np.dot(current_weights, day_ret)
        portfolio_values.append(portfolio_values[-1] * (1 + port_ret))
        
        # Rebalance Logic
        if t % REBALANCE_FREQ == 0 and t > LOOKBACK_WINDOW:
            current_state = hidden_states[t]
            regime = state_map[current_state]
            
            # Estimate parameters (No lookahead: use t-LOOKBACK to t)
            window_data = backtest_data.iloc[t-LOOKBACK_WINDOW:t]
            mu = window_data.mean().values
            Sigma = window_data.cov().values
            
            current_weights = get_optimal_weights(mu, Sigma, regime)
            
    # 4. Visualization
    print("4. Plotting results...")
    plot_results(portfolio_values, dates, hidden_states, backtest_data[spy_col], sorted_idx)
    
    final_val = portfolio_values[-1]
    spy_val = (1 + backtest_data[spy_col]).cumprod().iloc[-1] * 10000
    print(f"Final Portfolio: ${final_val:,.2f}")
    print(f"Benchmark SPY:   ${spy_val:,.2f}")

def plot_results(portfolio_values, dates, hidden_states, benchmark_data, sorted_idx):
    """Generates the performance chart"""
    strategy_curve = pd.Series(portfolio_values[1:], index=dates)
    benchmark_curve = (1 + benchmark_data).cumprod() * 10000
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Top Plot: Equity Curves
    ax1.plot(strategy_curve, label='HMM + Convex Opt', color='blue')
    ax1.plot(benchmark_curve, label='S&P 500', color='gray', linestyle='--', alpha=0.6)
    ax1.set_title("Strategy Performance vs Benchmark")
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bottom Plot: Regimes
    colors = {sorted_idx[0]: 'green', sorted_idx[1]: 'yellow', sorted_idx[2]: 'red'}
    ax2.plot(dates, benchmark_data.cumsum(), color='black', alpha=0.5)
    
    for i in range(0, len(dates)-1, 10): 
        state = hidden_states[i]
        ax2.axvspan(dates[i], dates[i+10], color=colors[state], alpha=0.3, linewidth=0)
        
    ax2.set_title("Detected Market Regimes (Green=Bull, Red=Bear)")
    ax2.set_ylabel("Market Trend")
    
    plt.tight_layout()
    print("   Saving graph to results_graph.png...")
    plt.savefig('results_graph.png') 
    print("   Graph saved.")
    # plt.show() # freezing

if __name__ == "__main__":
    run_pipeline()