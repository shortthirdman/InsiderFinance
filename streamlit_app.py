import streamlit as st

# Set up the page configuration
st.set_page_config(
    page_title="Insider Finance",
    page_icon=":streamlit:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# if "visibility" not in st.session_state:
#     st.session_state.visibility = "visible"
#     st.session_state.disabled = False

# Set up the Streamlit app
st.title("Insider Finance: Stock Portfolio Optimization with Various Strategies")

st.write("Hello Streamlit-er 👋")

trading_strategy_sharpe = st.Page(page="pages/1_TradingStrategy_SharpeRatio.py", title="Trading Strategy - Sharpe Ratio")
stock_portfolio_deap = st.Page(page="pages/2_StockPortfolio_DEAP.py", title="Stock Portfolio Optimization with DEAP")
bitcoin_rate_of_change = st.Page(page="pages/3_Optimize_Bitcoin_RateofChange.py", title="Optimize Bitcoin Rate of Change")

## title="Insider Finance", icon=":streamlit:"
pg = st.navigation([trading_strategy_sharpe, stock_portfolio_deap, bitcoin_rate_of_change])

pg.run()
