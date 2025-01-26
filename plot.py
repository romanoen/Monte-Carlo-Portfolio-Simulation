import plotly.graph_objects as go
import numpy as np
from scipy.stats import norm

def plot_monte_carlo_results(simulated_data):
    """
    Visualizes Monte Carlo simulation results using Plotly.

    Parameters:
        simulated_data (numpy.ndarray): A 2D numpy array (N x T) where rows represent different simulations
                                         and columns represent time steps.

    Returns:
        plotly.graph_objects.Figure: The generated Plotly figure object.
    """
    if simulated_data.ndim != 2:
        raise ValueError("Input data must be a 2D numpy array with shape (N, T).")

    fig = go.Figure()

    # Add each simulation as a separate trace
    for i in range(simulated_data.shape[0]):
        fig.add_trace(
            go.Scatter(
                x=np.arange(simulated_data.shape[1]),
                y=simulated_data[i, :],
                mode='lines',
                line=dict(width=0.5),
                name=f'Simulation {i + 1}' if i < 10 else None,  # Show names only for the first 10
                showlegend=i < 10  # Limit legend entries to the first 10 simulations
            )
        )

    # Update layout for better visualization
    fig.update_layout(
        title="Monte Carlo Simulation Results",
        xaxis_title="Time Steps",
        yaxis_title="Price",
        template="plotly_white",
        showlegend=True,
        legend=dict(title="Simulations", itemsizing="trace")
    )

    return fig

def plot_histogram_with_normal(random_data):
    """
    Visualizes a histogram of random data with an overlaid normal distribution curve.

    Parameters:
        random_data (array-like): A 1D array of random numbers.

    Returns:
        plotly.graph_objects.Figure: The generated Plotly figure object.
    """
    if random_data.ndim != 1:
        raise ValueError("Input data must be a 1D array.")

    # Calculate mean and standard deviation
    mu = np.mean(random_data)
    sigma = np.std(random_data)

    # Generate the normal distribution curve
    x = np.linspace(min(random_data), max(random_data), 1000)
    pdf = norm.pdf(x, mu, sigma)

    fig = go.Figure()

    # Add histogram
    fig.add_trace(
        go.Histogram(
            x=random_data,
            histnorm='probability density',
            name='Histogram',
            opacity=0.4,
            marker=dict(color='blue')
        )
    )

    # Add normal distribution curve
    fig.add_trace(
        go.Scatter(
            x=x,
            y=pdf,
            mode='lines',
            line=dict(color='red', width=2),
            name='Normal Distribution'
        )
    )

    # Update layout
    fig.update_layout(
        title="Distribution of Daily Return",
        xaxis_title="Value",
        yaxis_title="Density",
        template="plotly_white",
        showlegend=True
    )

    return fig
