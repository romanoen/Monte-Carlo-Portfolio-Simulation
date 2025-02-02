import numpy as np
import pandas as pd

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

    print(portfolio_paths)

    return portfolio_paths


import numpy as np
import pandas as pd

import numpy as np

def calculate_metrics(simulation_paths, confidence_level=0.95):
    """
    Calculates simulation-wide metrics from a list of simulation paths.
    Each inner list (or row in a 2D array) represents a single simulation run.

    Calculated Metrics:
      - Mean Return: Average of the final values across all simulation runs.
      - Standard Deviation: Standard deviation of the final values.
      - Confidence Interval: e.g. 95% confidence interval of the final values.
      - VaR (5%): Value at Risk at the 5th percentile.
      - CVaR (5%): Conditional Value at Risk at the 5th percentile.
      - Max Value: Maximum final value across all simulation runs.
      - Min Value: Minimum final value across all simulation runs.
      - Average Max Drawdown: Average of the maximum drawdown of each simulation run.
      - Worst-Case Max Drawdown: Highest maximum drawdown among all simulation runs.

    Parameters:
        simulation_paths (list of lists or 2D array): Simulated paths.
            Example:
            [
              [334.0, 324.72, 335.46, ..., 542.05, 562.25, 538.09],
              [334.0, 344.33, 343.33, ..., 726.75, 752.41, 751.60],
              [334.0, 324.15, 314.39, ..., 782.15, 798.56, 806.87],
              ...
            ]
        confidence_level (float): Confidence level for the confidence interval (default: 0.95).

    Returns:
        dict: A dictionary containing all the calculated metrics.
    """
    # Convert the list of lists to a NumPy array.
    # Each row corresponds to a simulation path.
    paths_array = np.array(simulation_paths)
    
    # Extract the final values from each simulation run.
    # This assumes that the final value is the last element in each simulation path.
    final_values = paths_array[:, -1]
    
    # Calculate the mean return (average of the final values).
    mean_return = np.mean(final_values)
    
    # Calculate the standard deviation of the final values.
    std_dev = np.std(final_values)
    
    # Calculate the confidence interval for the final values.
    # For a 95% confidence interval, this finds the 2.5th and 97.5th percentiles.
    ci_low = np.percentile(final_values, (1 - confidence_level) / 2 * 100)
    ci_high = np.percentile(final_values, (1 + confidence_level) / 2 * 100)
    
    # Calculate Value at Risk (VaR) at the 5th percentile.
    var_5 = np.percentile(final_values, 5)
    
    # Calculate Conditional Value at Risk (CVaR) at the 5th percentile,
    # which is the average of the final values that are below the VaR threshold.
    cvar_5 = np.mean(final_values[final_values <= var_5])
    
    # Determine the maximum and minimum final values across all simulation runs.
    max_value = np.max(final_values)
    min_value = np.min(final_values)
    
    # Initialize a list to hold the maximum drawdown for each simulation path.
    max_drawdowns = []
    
    # Loop through each simulation path to calculate its maximum drawdown.
    for path in paths_array:
        # Compute the cumulative maximum along the path.
        cumulative_max = np.maximum.accumulate(path)
        # Calculate the drawdown at each point: the difference between the cumulative max and the current value.
        drawdowns = cumulative_max - path
        # Append the maximum drawdown from this simulation run to the list.
        max_drawdowns.append(np.max(drawdowns))
    
    # Calculate the average maximum drawdown over all simulation runs.
    avg_max_drawdown = np.mean(max_drawdowns)
    
    # Determine the worst-case maximum drawdown (i.e., the maximum drawdown among all simulation runs).
    worst_case_drawdown = np.max(max_drawdowns)
    
    # Combine all the calculated metrics into a dictionary.
    metrics = {
        "Mean Return": mean_return,
        "Standard Deviation": std_dev,
        f"{int(confidence_level*100)}% Confidence Interval": f"[{ci_low:.2f}, {ci_high:.2f}]",
        "VaR (5%)": var_5,
        "CVaR (5%)": cvar_5,
        "Max Value": max_value,
        "Min Value": min_value,
        "Average Max Drawdown": avg_max_drawdown,
        "Worst-Case Max Drawdown": worst_case_drawdown
    }
    
    # Return the dictionary containing all the metrics.
    return metrics