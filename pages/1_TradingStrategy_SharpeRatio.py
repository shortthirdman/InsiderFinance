import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import optuna
from optuna.samplers import TPESampler

st.title("Trading Strategy - Sharpe Ratio Optimization")

def get_data(ticker="F", start="2015-01-01", end="2024-12-31") -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end)
    df.columns = df.columns.get_level_values(0)
    df = df[['Close']]
    return df

# User inputs for stock data
symbol = st.text_input("Ticker Symbol", "^GSPC")
start_date = st.date_input("Start Date", pd.to_datetime("1990-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("2024-12-31"))
train_cutoff_date = st.date_input("Training Cutoff Date", pd.to_datetime("2019-12-31"))

data = get_data(symbol, start_date, end_date)

st.write(f"Data for {symbol} from {start_date} to {end_date}")
st.table(data.head())


st.write("Closing Price Plot")

plt.figure(figsize=(14, 7))
plt.plot(data.index, data['Close'], color='blue', linewidth=1)
plt.title(f'Closing Price of {symbol} (Ticker: {symbol})')
plt.xlabel('Date')
plt.ylabel('Close Price ($)')
plt.grid(True)
plt.tight_layout()
# plt.savefig('closing_price_plot.png', dpi=300)
plt.show()


# 80/20 train-test split
split_date = data.index[int(len(data) * 0.8)]
train_data = data[:split_date]
test_data = data[split_date:]
st.write(f"Train: {train_data.index[0].date()} to {train_data.index[-1].date()}")
st.write(f"Test:  {test_data.index[0].date()} to {test_data.index[-1].date()}")

def apply_dmac(df, short_window, long_window):
    if short_window >= long_window:
        return pd.DataFrame()

    data = df.copy()
    data['short_ma'] = data['Close'].rolling(window=short_window).mean()
    data['long_ma'] = data['Close'].rolling(window=long_window).mean()

    # Long/flat signals: 1 = long, 0 = flat (cash)
    data['signal'] = 0
    data.loc[data['short_ma'] > data['long_ma'], 'signal'] = 1

    data['position'] = data['signal'].shift(1).fillna(0)

    # Buy and sell flags (optional for plotting)
    data['buy'] = (data['position'] == 1) & (data['position'].shift(1) == 0)
    data['sell'] = (data['position'] == 0) & (data['position'].shift(1) == 1)

    # Calculate returns
    data['returns'] = data['Close'].pct_change()
    data['strategy_returns'] = data['position'] * data['returns']

    # Zero returns before first trade
    first_pos_idx = data['position'].first_valid_index()
    if first_pos_idx is not None:
        data.loc[:first_pos_idx, 'strategy_returns'] = 0

    data.dropna(inplace=True)
    return data

def calculate_sharpe(data, risk_free_rate=0.01):
    excess_returns = data['strategy_returns'] - risk_free_rate / 252
    sharpe = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    return sharpe

def objective(trial):
    short_window = trial.suggest_int("short_window", 5, 50)
    long_window = trial.suggest_int("long_window", short_window + 5, 200)

    df = apply_dmac(train_data, short_window, long_window)
    if df.empty or df['strategy_returns'].std() == 0:
        return -np.inf

    return calculate_sharpe(df)


sampler = TPESampler(seed=42)
study = optuna.create_study(direction="maximize", sampler=sampler)
study.optimize(objective, n_trials=50)

print("Best Sharpe Ratio:", round(study.best_value, 4))
print("Best Parameters:", study.best_params)


best_short = study.best_params['short_window']
best_long = study.best_params['long_window']
result_df = apply_dmac(test_data, best_short, best_long)

plt.figure(figsize=(14, 7))
plt.plot(result_df['Close'], label='Price', color='blue', linewidth=1)
plt.plot(result_df['short_ma'], label=f'Short MA ({best_short})', color='green')
plt.plot(result_df['long_ma'], label=f'Long MA ({best_long})', color='orange')

# Buy ▲ and Sell ▼ markers
plt.plot(result_df[result_df['buy']].index, result_df[result_df['buy']]['Close'],
         '^', markersize=14, color='lime', label='Buy Signal')
plt.plot(result_df[result_df['sell']].index, result_df[result_df['sell']]['Close'],
         'v', markersize=14, color='red', label='Sell Signal')

plt.title(f'DMAC Buy/Sell Signals on Test Set (Short={best_short}, Long={best_long})')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
# plt.savefig('dmac_signals_plot.png', dpi=300)
plt.show()


# Calculate cumulative returns for DMAC strategy (on test_data with best params)
cumulative_strategy = (1 + result_df['strategy_returns']).cumprod()

# Calculate buy & hold cumulative returns on full test_data (fixed, independent of params)
cumulative_bh = (1 + test_data['Close'].pct_change()).cumprod()

plt.figure(figsize=(12, 6))
plt.plot(result_df.index, cumulative_strategy, label='DMAC Strategy', color='orange')
plt.plot(test_data.index, cumulative_bh, label='Buy & Hold', color='green')
plt.title(f'Cumulative Returns on Test Set (Sharpe: {study.best_value:.2f})')
plt.xlabel('Date')
plt.ylabel('Growth of $1')
plt.legend()
plt.grid(True)
# plt.savefig('cumulative_returns_plot.png', dpi=300)
plt.show()


start_capital = 1  # dollars

final_strategy_value = start_capital * cumulative_strategy.iloc[-1]
final_bh_value = start_capital * cumulative_bh.iloc[-1]

strategy_return_pct = (final_strategy_value - start_capital) / start_capital * 100
bh_return_pct = (final_bh_value - start_capital) / start_capital * 100

num_trades = int(result_df['buy'].sum() + result_df['sell'].sum())

# Prepare table data
table = [
    ["Strategy", "Final Value ($)", "Return (%)", "Total Trades"],
    ["DMAC Strategy", f"{final_strategy_value:.2f}", f"{strategy_return_pct:.2f}%", num_trades],
    ["Buy & Hold", f"{final_bh_value:.2f}", f"{bh_return_pct:.2f}%", "—"],
]

print(tabulate(table, headers="firstrow", tablefmt="rounded_grid"))