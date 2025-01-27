import streamlit as st
import yfinance as yf
import numpy as np
from datetime import datetime
from plot import plot_simulations, plot_histogram_with_normal
from compute import monte_carlo_simulation

st.set_page_config(layout="wide")

class StockAnalysisApp:
    def __init__(self):
        """Initialize the Stock Analysis App."""
        self.ticker = None
        self.start_date = "2010-01-01"
        self.end_date = "2025-01-01"
        self.stock_data = None
        self.T = 1000 #Anzahl Simulationen
        self.N = 252 #Tage zu simulieren
        self.initialPrice = None
        self.mu = None
        self.sigma = None

    def get_user_input(self):
        """Create input widgets for the user to specify stock ticker and date range."""
        st.title("Portfolio Risk Analysis")

        # Create a 3-column layout for ticker input, date range, and the Analyze button
        col1, col2, col3 = st.columns(3)

        # Ticker input in the first column
        with col1:
            self.ticker = st.text_input("Enter the stock ticker (e.g., AAPL, TSLA):", "AAPL")

        # Dropdown and button in the second column
        with col2:
            col21, col22, col23, col24, col25 = st.columns(5)
            with col21:
                col21.markdown("<div style='width: 1px; height: 28px'></div>", unsafe_allow_html=True)
                # Add the "Analyze Stock" button
                analyze_button_clicked = st.button("Go")

            return analyze_button_clicked
    def sidebar(self):
        time_options = ["1 Month", "1 Year", "5 Years", "10 Years"]
        st.sidebar.divider()
        # Create the select slider
        selected_time = st.sidebar.select_slider(
            "Number of years providing historical returns So,  10 Years means from 2015 to 2025",
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

        t_simulation_runs = ["10", "100", "1,000", "10,000", "100,000"]

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


    def fetch_stock_data(self):
        """Fetch historical stock data from Yahoo Finance."""
        if self.ticker:
            try:
                self.stock_data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
                if self.stock_data.empty:
                    st.error("No data found for the specified ticker and date range.")
                    return False
                return True
            except Exception as e:
                st.error(f"An error occurred while fetching stock data: {e}")
                return False
        else:
            st.warning("Please enter a valid stock ticker.")
            return False

    def plot_hist_norm(self):
        """Analyze and visualize the daily returns of the stock."""
        if self.stock_data is not None:
            self.stock_data['Daily Returns'] = self.stock_data['Close'].pct_change()
            daily_returns = self.stock_data['Daily Returns'].dropna()

            # Perform analysis
            self.mu = daily_returns.mean()
            self.sigma = daily_returns.std()
            self.initialPrice = self.stock_data['Close'].iloc[-1]

            # Plot histogram with normal distribution
            fig = plot_histogram_with_normal(daily_returns)
            st.plotly_chart(fig)

            # Display key statistics
            st.write(f"Mean (\u03bc): {self.mu:.6f}")
            st.write(f"Standard Deviation (\u03c3): {self.sigma:.6f}")
        
    def plotMC(self):

        matrix = monte_carlo_simulation(self.N, self.T, self.initialPrice, self.mu, self.sigma)

        fig = plot_simulations(matrix, self.N, self.T)

        st.plotly_chart(fig)

    def run(self):
        """Run the Streamlit app."""
        st.sidebar.title("Configure Simulation")
        self.sidebar()
        analyze_button_clicked = self.get_user_input()

        if analyze_button_clicked:
            if self.fetch_stock_data():
                col1, col2 = st.columns(2)
                with col1:
                    self.plot_hist_norm()
                with col2:
                    self.plotMC()


if __name__ == "__main__":
    app = StockAnalysisApp()
    app.run()
