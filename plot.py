import plotly.graph_objects as go
import numpy as np
from scipy.stats import norm

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
    for i in range(N):  # N steht für die Anzahl der Simulationen (Spalten)
        fig.add_trace(go.Scatter(
            x=np.arange(T),  # X-Achse: Handelstage (Zeilen)
            y=simulations[:, i],  # Zugriff auf die i-te Spalte (Simulation)
            mode='lines',
            line=dict(color='rgba(0, 0, 255, 0.1)', width=1),  # rgba mit Transparenz
        ))

    # Add titles and axis labels
    fig.update_layout(
        title=f"Monte Carlo Simulation ({N} Simulations)",
        xaxis_title="Trading Days",
        yaxis_title="Share Price",
        showlegend=False
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