import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime
from plot import plot_simulations, plot_histograms_and_normal_dist
from compute import multi_monte_carlo_sim, calculate_metrics
from test import plot_qq

st.set_page_config(layout="wide")

class StockAnalysisApp:
    def __init__(self):
        """Initialize the Stock Analysis App."""
        self.tickers = {
            "Asset Ticker": ["AAPL", "GOOG", "MTX.DE"],
            "Capital (in €)": [100, 400, 350],
        }
        self.start_date = "2010-01-01"
        self.end_date = "2025-01-01"
        self.stock_data = {}
        self.T = 1000  # Anzahl Simulationen
        self.N = 252  # Tage zu simulieren
        self.asset_metrics = {}
        self.simulation_started = False

    def sidebar(self):
        st.sidebar.title("Configure Simulation")
        time_options = ["1 Month", "1 Year", "5 Years", "10 Years"]
        st.sidebar.divider()
        selected_time = st.sidebar.select_slider("Number of years providing historical returns", options=time_options, value="10 Years")
        self.start_date = {"1 Month": "2024-01-12", "1 Year": "2024-01-01", "5 Years": "2020-01-01", "10 Years": "2010-01-01"}.get(selected_time, "2010-01-01")
        self.end_date = "2025-01-01"
        
        st.sidebar.divider()
        t_simulation_runs = ["10", "100", "500", "1,000", "10,000", "100,000"]
        selected_time = st.sidebar.select_slider("Number of Simulation Runs (Calculated Paths)", options=t_simulation_runs, value="1,000")
        self.T = int(selected_time.replace(",", ""))
        
        st.sidebar.divider()
        t_simulation_duration = ["1 Month", "6 Months", "1 Year", "2 Years"]
        selected_duration = st.sidebar.select_slider("Simulation Duration", options=t_simulation_duration, value="1 Year")
        self.N = {"1 Month": self.N // 12, "6 Months": self.N // 2, "1 Year": self.N, "2 Years": self.N * 2}.get(selected_duration, self.N)
        
        st.sidebar.divider()
        tickers_capital = pd.DataFrame(self.tickers)
        self.edited_tickers = st.sidebar.data_editor(tickers_capital, num_rows="dynamic")
        st.sidebar.divider()

    def get_user_input(self):
        return st.sidebar.button("Start Simulation")

    def fetch_stock_data(self):
        self.asset_metrics = {}
        if self.edited_tickers is None or self.edited_tickers.empty:
            st.warning("No tickers provided. Please add at least one stock.")
            return False
        
        for _, row in self.edited_tickers.iterrows():
            ticker = row["Asset Ticker"].strip()
            if not ticker:
                continue
            
            try:
                stock_data = yf.download(ticker, start=self.start_date, end=self.end_date)
                if stock_data.empty:
                    st.error(f"No data found for {ticker} in the specified date range.")
                    continue
                
                stock_data["Daily Returns"] = stock_data["Close"].pct_change()
                self.asset_metrics[ticker] = {
                    "mu": stock_data["Daily Returns"].mean(),
                    "sigma": stock_data["Daily Returns"].std(),
                    "initial_price": row["Capital (in €)"],
                    "returns": stock_data["Daily Returns"].dropna().values
                }
            except Exception as e:
                st.error(f"Error fetching data for {ticker}: {e}")
        
        return bool(self.asset_metrics)

    def plot_results(self):
        st.title("Portfolio Risk Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_histograms_and_normal_dist(self.asset_metrics))
            st.plotly_chart(plot_qq(self.asset_metrics))
        with col2:
            self.matrix = multi_monte_carlo_sim(self.T, self.N, self.asset_metrics)
            st.plotly_chart(plot_simulations(self.matrix, self.T, self.N))
            st.json(calculate_metrics(self.matrix))

    def show_readme(self):
        try:
            with open("README.md", "r") as file:
                readme_content = file.read()
                st.markdown(readme_content, unsafe_allow_html=True)
        except FileNotFoundError:
            st.warning("README.md not found. Please ensure the file is available.")

    def run(self):
        self.sidebar()
        if self.get_user_input():
            if self.fetch_stock_data():
                self.simulation_started = True
                self.plot_results()
        elif not self.simulation_started:
            self.show_readme()

if __name__ == "__main__":
    app = StockAnalysisApp()
    app.run()