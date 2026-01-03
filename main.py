import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.data_loader import fetch_data, calculate_returns
from src.hmm_model import train_hmm
from src.optimizer import get_optimal_weights

# --- CONFIGURATION ---
TICKERS = ['SPY', 'TLT', 'GLD'] 
RISKY_ASSET = 'SPY' 
START_DATE = '2005-01-01'
END_DATE = '2024-01-01'
REBALANCE_FREQ = 20  
LOOKBACK_WINDOW = 50 

def run_pipeline():
    # 1. Data Ingestion
    print(f"1. Fetching data for {TICKERS}...")
    data = fetch_data(TICKERS, START_DATE, END_DATE)
    returns = calculate_returns(data)
    
    # Use a local variable to avoid Global scope conflicts
    risky_ticker = RISKY_ASSET
    
    # Fallback Check: If SPY isn't in the data, pick the first available asset
    if risky_ticker not in data.columns:
        risky_ticker = data.columns[0]

    # 2. AI Regime Detection
    print(f"2. Training HMM on {risky_ticker}...")
    
    asset_ret = returns[risky_ticker].values.reshape(-1, 1)
    asset_vol = returns[risky_ticker].rolling(10).std().dropna().values.reshape(-1, 1)
    
    trunc_len = len(asset_vol)
    asset_ret = asset_ret[-trunc_len:]
    X = np.column_stack([asset_ret, asset_vol])
    
    # Train Model
    model, hidden_states = train_hmm(X, n_states=3)
    
    # Map States to Regimes
    analysis_df = pd.DataFrame(X, columns=['Ret', 'Vol'])
    analysis_df['State'] = hidden_states
    vol_by_state = analysis_df.groupby('State')['Vol'].mean()
    sorted_idx = vol_by_state.sort_values().index
    
    state_map = {sorted_idx[0]: 'BULL', sorted_idx[1]: 'CHOP', sorted_idx[2]: 'BEAR'}
    print(f"   State Mapping: {state_map}")

    # 3. Backtesting Loop
    print("3. Running Backtest Simulation...")
    portfolio_values = [10000]
    current_weights = np.ones(len(TICKERS)) / len(TICKERS)
    
    backtest_data = returns.iloc[-trunc_len:]
    dates = backtest_data.index
    
    for t in range(len(backtest_data)):
        # Daily PnL
        day_ret = backtest_data.iloc[t].values
        port_ret = np.dot(current_weights, day_ret)
        portfolio_values.append(portfolio_values[-1] * (1 + port_ret))
        
        # Rebalance
        if t % REBALANCE_FREQ == 0 and t > LOOKBACK_WINDOW:
            current_state = hidden_states[t]
            regime = state_map[current_state]
            
            window_data = backtest_data.iloc[t-LOOKBACK_WINDOW:t]
            mu = window_data.mean().values
            Sigma = window_data.cov().values
            
            # Use default risk_aversion=1.0
            current_weights = get_optimal_weights(mu, Sigma, regime)
            
    # 4. Visualization
    print("4. Plotting results...")
    plot_results(portfolio_values, dates, hidden_states, backtest_data[risky_ticker], sorted_idx, risky_ticker)
    
    final_val = portfolio_values[-1]
    benchmark_val = (1 + backtest_data[risky_ticker]).cumprod().iloc[-1] * 10000
    print(f"Final Portfolio: ${final_val:,.2f}")
    print(f"Benchmark ({risky_ticker}):   ${benchmark_val:,.2f}")

def plot_results(portfolio_values, dates, hidden_states, benchmark_data, sorted_idx, risky_ticker):
    strategy_curve = pd.Series(portfolio_values[1:], index=dates)
    benchmark_curve = (1 + benchmark_data).cumprod() * 10000
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    ax1.plot(strategy_curve, label='HMM Strategy', color='blue')
    ax1.plot(benchmark_curve, label=f'Benchmark ({risky_ticker})', color='gray', linestyle='--', alpha=0.6)
    ax1.set_title("Strategy Performance vs Benchmark")
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    colors = {sorted_idx[0]: 'green', sorted_idx[1]: 'yellow', sorted_idx[2]: 'red'}
    ax2.plot(dates, benchmark_data.cumsum(), color='black', alpha=0.5)
    
    # Safe Plotting Loop
    for i in range(0, len(dates)-1, 10): 
        state = hidden_states[i]
        start_date = dates[i]
        end_idx = min(i + 10, len(dates) - 1)
        end_date = dates[end_idx]
        ax2.axvspan(start_date, end_date, color=colors[state], alpha=0.3, linewidth=0)
        
    ax2.set_title("Detected Market Regimes (Green=Bull, Red=Bear)")
    ax2.set_ylabel("Market Trend")
    
    plt.tight_layout()
    print("   Saving graph to results_graph.png...")
    plt.savefig('results_graph.png') 
    print("   Graph saved.")

if __name__ == "__main__":
    run_pipeline()