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

# Set up the Streamlit app
st.title("Stock Portfolio Optimization with DEAP")