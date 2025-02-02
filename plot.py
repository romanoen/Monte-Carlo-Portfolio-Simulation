import plotly.graph_objects as go
import numpy as np
from scipy.stats import norm
import scipy.stats as stats
import plotly.colors as pc

# T = Anzahl Simulationen
# N = Tage

def plot_simulations(simulations, T, N):
    """
    Visualize the Monte Carlo simulations using Plotly.

    Parameters:
        simulations (ndarray): A 2D NumPy array of simulated prices with T rows and N columns.
        T (int): Number of simulations.
        N (int): Number of trading days.
    """
    fig = go.Figure()

    # Add simulated prices as lines to the Plotly figure
    for i in range(T):  # N steht für die Anzahl der Simulationen (Spalten)
        fig.add_trace(go.Scatter(
            x=np.arange(N+1),  # X-Achse: Handelstage (Zeilen)
            y=simulations[i, :],  # Zugriff auf die i-te Spalte (Simulation)
            mode='lines',
            line=dict(color='rgba(0, 0, 255, 0.1)', width=1),  # rgba mit Transparenz
        ))

    # Add titles and axis labels
    fig.update_layout(
        title=f"Monte Carlo Simulation ({T} Simulations)",
        xaxis_title="Trading Days",
        yaxis_title="Share Price",
        showlegend=False
    )

    return fig

def plot_histograms_and_normal_dist(asset_metrics):
    """
    Visualize the historical returns and the theoretical normal distribution for each asset using Plotly.

    Parameters:
        asset_metrics (dict): Dictionary containing asset-specific parameters.
    
    Example structure of `asset_metrics`:
    {
        "AAPL": {
            "mu": 0.0005,            # Average daily return for AAPL
            "sigma": 0.02,           # Standard deviation of daily returns for AAPL
            "returns": [0.003, 0.005, 0.002, ...]  # Historical returns for AAPL
        },
        "GOOG": {
            "mu": 0.0007,            
            "sigma": 0.015,          
            "returns": [0.003, 0.005, 0.002, ...] 
        }
    }
    """
    fig = go.Figure()

    # Colors for each asset (to distinguish them in the plot)
    colors = pc.qualitative.Set1

    # Loop over each asset and plot its histogram and normal distribution
    for idx, (asset, metrics) in enumerate(asset_metrics.items()):
        mu = metrics["mu"]
        sigma = metrics["sigma"]
        returns = metrics["returns"]
        
        # Name for legend (will be the same for both histogram and normal distribution)
        legend_name = f'{asset}'

        # Plotting the histogram of the historical returns for the asset
        fig.add_trace(go.Histogram(
            x=returns,
            nbinsx=30,  # Number of bins in the histogram
            histnorm='probability density',  # Normalize the histogram to show probabilities
            name=legend_name,  # Use the same name for legend
            marker_color=colors[idx],  # Assign a color to each asset
            opacity=0.6,  # Set opacity for the histogram to make it slightly transparent
            bingroup=0,  # Grouping for layering the plots
            legendgroup=legend_name,  # Group the histogram with the normal distribution in the legend
            showlegend=True  # Hide histogram from the legend
        ))

        # Creating the theoretical normal distribution based on mu and sigma
        x_values = np.linspace(min(returns) - 0.01, max(returns) + 0.01, 1000)
        normal_dist = stats.norm.pdf(x_values, loc=mu, scale=sigma)
        
        # Plotting the normal distribution for the asset
        fig.add_trace(go.Scatter(
            x=x_values,
            y=normal_dist,
            mode='lines',
            name=legend_name,  # Use the same name for legend
            line=dict(color=colors[idx], dash='dash'),  # Same color for asset's normal distribution
            opacity=1,  # Fully opaque line for normal distribution
            showlegend=False,  # Show only the normal distribution in the legend
            legendgroup=legend_name  # Group the normal distribution with the histogram in the legend
        ))

    # Update layout to make it more informative
    fig.update_layout(
        title="Histogram of Historical Returns and Normal Distribution",
        xaxis_title="Returns",
        yaxis_title="Probability Density",
        barmode='overlay',  # Overlay the histogram and the normal distribution
        template='plotly_dark',  # Use a dark theme for better visual clarity
        showlegend=True  # Show the legend to differentiate the assets
    )

    return fig