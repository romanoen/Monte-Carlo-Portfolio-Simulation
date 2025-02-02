import plotly.graph_objects as go
import numpy as np
import scipy.stats as stats

def plot_qq(asset_metrics):
    """
    Generate a Q-Q plot for all assets in asset_metrics.

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

    colors = ['blue', 'red', 'green', 'purple', 'orange', 'pink']  # Color list

    for idx, (asset, metrics) in enumerate(asset_metrics.items()):
        returns = np.array(metrics["returns"])  # Convert to NumPy array
        mu = metrics["mu"]
        sigma = metrics["sigma"]

        # Sort the empirical returns
        empirical_quantiles = np.sort(returns)

        # Generate theoretical quantiles from a normal distribution
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(returns)), loc=mu, scale=sigma)

        # Scatter plot for Q-Q comparison
        fig.add_trace(go.Scatter(
            x=theoretical_quantiles,
            y=empirical_quantiles,
            mode='markers',
            name=f"{asset} Q-Q Plot",
            marker=dict(color=colors[idx % len(colors)], size=5),
        ))

    # Add reference diagonal line (y=x)
    min_val = min(empirical_quantiles.min(), theoretical_quantiles.min())
    max_val = max(empirical_quantiles.max(), theoretical_quantiles.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name="Perfect Normal Fit",
        line=dict(color='white', dash='solid')
    ))

    # Update layout
    fig.update_layout(
        title="Q-Q Plot: Empirical Returns vs. Normal Distribution",
        xaxis_title="Theoretical Quantiles (Normal Dist.)",
        yaxis_title="Empirical Quantiles",
        template='plotly_dark',
        showlegend=True
    )

    return fig
