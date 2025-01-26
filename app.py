import streamlit as st
import yfinance as yf
import numpy as np
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
        self.T = 10000 #Anzahl Simulationen
        self.N = 252 #Tage zu simulieren
        self.initialPrice = None
        self.mu = None
        self.sigma = None

    def get_user_input(self):
        """Create input widgets for the user to specify stock ticker and date range."""
        st.title("Stock Analysis and Visualization App")

        # Create a 3-column layout for ticker input, date range, and the Analyze button
        col1, col2, col3 = st.columns(3)

        # Ticker input in the first column
        with col1:
            self.ticker = st.text_input("Enter the stock ticker (e.g., AAPL, TSLA):", "AAPL")

        # Dropdown and button in the second column
        with col2:
            col21, col22, col23, col24, col25 = st.columns(5)
            # Create a dropdown with options
            with col21:
                time_range = st.selectbox("Hist. Pricing:", ["1 Y", "5 Y", "10 Y"])

                # Set the start and end date based on the selected option
                if time_range == "1 Y":
                    self.start_date = "2024-01-01"
                    self.end_date = "2025-01-01"
                elif time_range == "5 Y":
                    self.start_date = "2020-01-01"
                    self.end_date = "2025-01-01"
                elif time_range == "10 Y":
                    self.start_date = "2010-01-01"
                    self.end_date = "2025-01-01"
            
            with col22:
                col22.markdown("<div style='width: 1px; height: 28px'></div>", unsafe_allow_html=True)
                # Add the "Analyze Stock" button
                analyze_button_clicked = st.button("Go")

            return analyze_button_clicked



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

        matrix = monte_carlo_simulation(self.N, self.T, 100, self.mu, self.sigma)

        fig = plot_simulations(matrix, self.N, self.T)

        st.plotly_chart(fig)

    def run(self):
        """Run the Streamlit app."""
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
