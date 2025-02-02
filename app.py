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
        st.title("Portfolio Risk Analysis")
        self.tickers = {
    "Asset Ticker": ["AAPL", "GOOG", "MTX.DE"],
    "Capital (in €)": [100, 400, 350],
        }
        self.start_date = "2010-01-01"
        self.end_date = "2025-01-01"
        self.stock_data = {}
        self.T = 1000 #Anzahl Simulationen
        self.N = 252 #Tage zu simulieren
        self.initialPrice = None
        self.mu = None
        self.sigma = None
        self.asset_metrics = {}

    def sidebar(self):
        time_options = ["1 Month", "1 Year", "5 Years", "10 Years"]
        st.sidebar.divider()
        # Create the select slider
        selected_time = st.sidebar.select_slider(
            "Number of years providing historical returns",
            options=time_options,  # Discrete options
            value="10 Years"  # Default value
        )

        # Set start and end dates based on the selected option
        if selected_time == "1 Month":
            self.start_date = "2024-01-12"
            self.end_date = "2025-01-01"
        elif selected_time == "1 Year":
            self.start_date = "2024-01-01"
            self.end_date = "2025-01-01"
        elif selected_time == "5 Years":
            self.start_date = "2020-01-01"
            self.end_date = "2025-01-01"
        elif selected_time == "10 Years":
            self.start_date = "2010-01-01"
            self.end_date = "2025-01-01"

        st.sidebar.divider()

        t_simulation_runs = ["10", "100", "500", "1,000", "10,000", "100,000"]

        selected_time = st.sidebar.select_slider(
            "Number of Simulation Runs (Calculated Paths)",
            options=t_simulation_runs,  # Discrete options
            value="1,000"  # Default value
        )

        if selected_time == "10":
            self.T = 10
        elif selected_time == "100":
            self.T = 100
        elif selected_time == "1,000":
            self.T = 1000
        elif selected_time == "10,000":
            self.T = 10000
        elif selected_time == "100,000":
            self.T = 100000
        
        st.sidebar.divider()

        t_simulation_runs = ["1 Month", "6 Months", "1 Year", "2 Years"]

        selected_time = st.sidebar.select_slider(
            "Simulation Duration",
            options=t_simulation_runs,  # Discrete options
            value="1 Year"  # Default value
        )

        if selected_time == "1 Month":
            self.N = int(self.N/12)
        elif selected_time == "6 Months":
            self.N = int(self.N/2)
        elif selected_time == "1 Year":
            pass
        elif selected_time == "2 Years":
            self.N = self.N*2

        st.sidebar.divider()

        tickers_capital = pd.DataFrame(self.tickers)

        # Editable Table
        self.edited_tickers = st.sidebar.data_editor(tickers_capital, num_rows="dynamic")

        st.sidebar.divider()

    def get_user_input(self):
        analyze_button_clicked = st.sidebar.button("Start Simulation")
        return analyze_button_clicked




    def fetch_stock_data(self):
        """Fetch historical stock data for all assets and compute mu, sigma, and initial price."""
        self.asset_metrics = {}

        if self.edited_tickers is None or self.edited_tickers.empty:
            st.warning("No tickers provided. Please add at least one stock.")
            return False

        for index, row in self.edited_tickers.iterrows():
            ticker = row["Asset Ticker"].strip()  # Entferne unnötige Leerzeichen
            if not ticker:
                continue  # Falls ein leerer Ticker vorhanden ist

            try:
                stock_data = yf.download(ticker, start=self.start_date, end=self.end_date)

                if stock_data.empty:
                    st.error(f"No data found for {ticker} in the specified date range.")
                    continue

                stock_data["Daily Returns"] = stock_data["Close"].pct_change()
                daily_returns = stock_data["Daily Returns"].dropna()

                mu = daily_returns.mean()
                sigma = daily_returns.std()
                initial_price = initial_price = row["Capital (in €)"]

                # Speichern der berechneten Werte für das jeweilige Asset
                self.asset_metrics[ticker] = {
                    "mu": mu,
                    "sigma": sigma,
                    "initial_price": initial_price,
                    "returns": daily_returns.values
                }

            except Exception as e:
                st.error(f"Error fetching data for {ticker}: {e}")

        if not self.asset_metrics:
            return False  # Falls keine Daten erfolgreich geladen wurden
        return True

    def plotHistograms(self):
        fig = plot_histograms_and_normal_dist(self.asset_metrics)

        st.plotly_chart(fig)
        
    def plotMC(self):

        self.matrix = multi_monte_carlo_sim(self.T, self.N, self.asset_metrics)

        fig = plot_simulations(self.matrix, self.T, self.N)

        st.plotly_chart(fig)
    
    def plot_qq(self):
        qq = plot_qq(self.asset_metrics)

        st.plotly_chart(qq)
    
    def plotMetrics(self):
        metrics = calculate_metrics(self.matrix)

        st.json(metrics)


    def run(self):
        """Run the Streamlit app."""
        st.sidebar.title("Configure Simulation")
        self.sidebar()
        analyze_button_clicked = self.get_user_input()

        if analyze_button_clicked:
            if self.fetch_stock_data():
                col1, col2 = st.columns(2)
                with col1:
                    self.plotHistograms()
                    self.plot_qq()
                with col2:
                    self.plotMC()
                    self.plotMetrics()


if __name__ == "__main__":
    app = StockAnalysisApp()
    app.run()
