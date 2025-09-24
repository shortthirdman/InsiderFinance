import streamlit as st

import sys

import yfinance as yf
import matplotlib.pyplot as plt

from ta.momentum import ROCIndicator
from bayes_opt import BayesianOptimization
from tabulate import tabulate

plt.style.use('dark_background')

st.markdown("## Loading the Data")

# Load Bitcoin stock data
symbol = 'BTC-USD'
initial_cash = 1  # Initial cash for backtesting
data = yf.download(symbol, start='2020-01-01', end='2024-12-31')
data.columns = data.columns.get_level_values(0)
data = data[['Close']]
data.dropna(inplace=True)
data.head()

st.markdown("## Visualize the closing price of the stock")

plt.figure(figsize=(14,6))
plt.plot(data.index, data['Close'], label=f'{symbol} Closing Price', color='blue')
plt.title(f'{symbol} Closing Price (2020–2024)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.grid(True)
plt.legend()
# plt.savefig('closing_price.png', dpi=300, bbox_inches='tight')
plt.show()

st.markdown("## Train-Test Split")

# Train-test split (70% train, 30% test)
train_size = int(len(data) * 0.7)
train_data = data.iloc[:train_size].copy()
test_data = data.iloc[train_size:].copy()

st.write(f"Training from {train_data.index[0]} to {train_data.index[-1]}")
st.write(f"Testing from {test_data.index[0]} to {test_data.index[-1]}")

st.markdown("## Defining the Backtesting Function")

def backtest_strategy(df, roc_window, buy_threshold, sell_threshold):
    df = df.copy()
    roc_window = int(roc_window)
    buy_threshold = float(buy_threshold)
    sell_threshold = float(sell_threshold)

    roc = ROCIndicator(close=df['Close'], window=roc_window)
    df['ROC'] = roc.roc()

    position = 0
    cash = initial_cash
    portfolio = []
    trades = 0
    buy_signals = []
    sell_signals = []

    for i in range(roc_window, len(df)):
        if df['ROC'].iloc[i] > buy_threshold and position == 0:
            position = cash / df['Close'].iloc[i]
            cash = 0
            buy_signals.append((df.index[i], df['Close'].iloc[i]))
            trades += 1
        elif df['ROC'].iloc[i] < sell_threshold and position > 0:
            cash = position * df['Close'].iloc[i]
            position = 0
            sell_signals.append((df.index[i], df['Close'].iloc[i]))
            trades += 1
        
        portfolio_value = cash + (position * df['Close'].iloc[i])
        portfolio.append(portfolio_value)

    final_value = cash + position * df['Close'].iloc[-1]
    return_percentage = (final_value - initial_cash) / initial_cash * 100

    return return_percentage, trades, buy_signals, sell_signals, portfolio

st.markdown("## Objective Function for Bayesian Optimization")

def objective(roc_window, buy_threshold, sell_threshold):
    returns, _, _, _, _ = backtest_strategy(train_data, roc_window, buy_threshold, sell_threshold)
    return returns


pbounds = {
    'roc_window': (3, 30),
    'buy_threshold': (0.1, 10),
    'sell_threshold': (-10, -0.1)
}

optimizer = BayesianOptimization(
    f=objective,
    pbounds=pbounds,
    random_state=42,
    verbose=2
)

optimizer.maximize(init_points=5, n_iter=20)
best_params = optimizer.max['params']

st.write("Best Parameters Found:")
st.write(best_params)

st.markdown("## Backtesting on Test Data Using Optimized Parameters")

best_roc = int(best_params['roc_window'])
best_buy = float(best_params['buy_threshold'])
best_sell = float(best_params['sell_threshold'])

returns, trades, buy_signals, sell_signals, portfolio = backtest_strategy(
    test_data, best_roc, best_buy, best_sell
)

st.write(f"Final Return on Test Set: {returns:.2f}%")
st.write(f"Number of Trades: {trades}")

st.markdown("## Comparing Strategy vs Buy-and-Hold")

# Visualize portfolio value vs. buy-and-hold on the test set
df = test_data.copy()
df = df.iloc[best_roc:].copy()  # Skip rows lost due to ROC calculation

# Strategy Portfolio Value
df['Strategy'] = portfolio

# Buy-and-hold simulation
initial_price = df['Close'].iloc[0]
df['BuyHold'] = (df['Close'] / initial_price) * initial_cash

# Plot both
plt.figure(figsize=(14,6))
plt.plot(df.index, df['Strategy'], label='ROC Strategy', color='orange')
plt.plot(df.index, df['BuyHold'], label='Buy & Hold', color='green')

plt.title('ROC Strategy vs. Buy & Hold (Test Set)')
plt.xlabel('Date')
plt.ylabel('Portfolio Value ($)')
plt.legend()
plt.grid(True)
## plt.savefig('portfolio_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

st.markdown("## Visualizing Buy and Sell Signals")

# Plot buy/sell signals on test set
plt.figure(figsize=(14,6))

# Plot the closing price first
plt.plot(test_data.index, test_data['Close'], label='Price', color='blue', zorder=1)

# Plot buy signals
for buy in buy_signals:
    plt.scatter(buy[0], buy[1], marker='^', color='lime', s=100, label='Buy Signal', zorder=2)

# Plot sell signals
for sell in sell_signals:
    plt.scatter(sell[0], sell[1], marker='v', color='red', s=100, label='Sell Signal', zorder=2)

# Only show each label once
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

plt.title('Buy/Sell Signals on Test Set')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid(True)
## plt.savefig('buy_sell_signals.png', dpi=300, bbox_inches='tight')
plt.show()

st.markdown("## Final Performance Summary")

# Strategy evaluation
total_trades = len(buy_signals) + len(sell_signals)

if len(buy_signals) == len(sell_signals):
    successful_trades = sum(
        sell[1] > buy[1] for buy, sell in zip(buy_signals, sell_signals)
    )
    win_rate = f"{(successful_trades / len(sell_signals) * 100):.2f}%"
else:
    win_rate = 'Inconsistent buy/sell pairs'

test_df = test_data.copy().iloc[best_roc:].copy()  # align with portfolio calculation start
initial_price = test_df['Close'].iloc[0]
final_price = test_df['Close'].iloc[-1]

final_strategy_value = portfolio[-1]
final_bh_value = (final_price / initial_price) * initial_cash
bh_return_pct = (final_bh_value - initial_cash) / initial_cash * 100
strategy_return_pct = (final_strategy_value - initial_cash) / initial_cash * 100

summary = [
    ["Initial Cash", f"${initial_cash}", f"${initial_cash}"],
    ["Final Portfolio Value", f"${final_strategy_value:.2f}", f"${final_bh_value:.2f}"],
    ["Total Return (%)", f"{strategy_return_pct:.2f}%", f"{bh_return_pct:.2f}%"],
    ["Total Trades Executed", total_trades, "N/A"],
    ["Buy Trades", len(buy_signals), "N/A"],
    ["Sell Trades", len(sell_signals), "N/A"],
    ["Win Rate", win_rate, "N/A"]
]

st.write(tabulate(summary, headers=["Metric", "ROC Strategy", "Buy & Hold"], tablefmt="rounded_grid"))