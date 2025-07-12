import streamlit as st

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import calendar
import random
import time
from deap import base, creator, tools, algorithms

plt.style.use("dark_background")

if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

st.title("Hello Streamlit-er 👋")

symbol = st.text_input("Ticker Symbol", "^GSPC")
start_date = "1990-01-01"
end_date = "2024-12-31"
train_cutoff_date = "2019-12-31"

st.table(dataframe)