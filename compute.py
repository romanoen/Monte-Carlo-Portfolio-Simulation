import numpy as np

def multi_monte_carlo_sim(T: int, N: int, asset_metrics: dict):
    """
    Runs a Monte Carlo simulation for portfolio risk management using 
    Geometric Brownian Motion (GBM).

    Parameters:
    - T (int): Number of simulation runs (paths).
    - N (int): Number of simulation days.
    - asset_metrics (dict): Dictionary containing asset-specific parameters.

    Returns:
    - portfolio_paths (np.ndarray): A (T x (N+1)) matrix containing portfolio values 
                                    for each simulation run and time step.
    """

    # Extract asset names and count
    assets = list(asset_metrics.keys())  
    num_assets = len(assets)  

    # Convert asset parameters into numpy arrays for vectorized operations
    initial_prices = np.array([asset_metrics[asset]["initial_price"] for asset in assets])
    mu = np.array([asset_metrics[asset]["mu"] for asset in assets])
    sigma = np.array([asset_metrics[asset]["sigma"] for asset in assets])

    # Preallocate portfolio paths (T x (N+1))
    portfolio_paths = np.zeros((T, N + 1))  
    portfolio_paths[:, 0] = np.sum(initial_prices)  # Initial portfolio value

    # Generate all random normal values for efficiency (T x N x num_assets)
    rand_normals = np.random.normal(size=(T, N, num_assets))  

    # Compute drift and diffusion terms ahead of time
    drift = (mu - 0.5 * sigma ** 2)  # (num_assets,)
    diffusion = sigma * rand_normals  # (T, N, num_assets)

    # Compute cumulative log returns using np.cumsum()
    log_returns = drift * 1 + diffusion  # Daily returns
    cumulative_log_returns = np.cumsum(log_returns, axis=1)  # Cumulative sum over days

    # Compute asset prices using vectorized exponentiation
    asset_prices = initial_prices * np.exp(cumulative_log_returns)  # Element-wise exponentiation

    # Compute portfolio values efficiently using np.einsum (sum across assets)
    portfolio_paths[:, 1:] = np.einsum('ijk->ij', asset_prices)  

    return portfolio_paths
