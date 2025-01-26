import numpy as np
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
from plot import plot_monte_carlo_results

def simulate_single_batch(T, initial_price, mu, sigma, n_sim):
    Z = np.random.normal(0, 1, (T-1, n_sim))
    drift = (mu - 0.5 * sigma**2)
    diffusion = sigma * Z
    log_prices = np.cumsum(drift + diffusion, axis=0)
    prices = initial_price * np.exp(log_prices)
    return np.vstack((np.full((1, n_sim), initial_price), prices))

def compute_parallel(N, T, initial_price, mu, sigma):

    num_cores = 1  # Limit to a reasonable number of threads for Streamlit
    n_sim_per_core = N // num_cores

    args = [(T, initial_price, mu, sigma, n_sim_per_core) for _ in range(num_cores)]

    with ThreadPoolExecutor(max_workers=num_cores) as executor:
        results = list(executor.map(lambda a: simulate_single_batch(*a), args))

    return np.hstack(results)


def plot_monte_carlo_results(prices):
    T = prices.shape[0]  # Anzahl der Zeitpunkte (Zeilen)
    n_sim = prices.shape[1]  # Anzahl der Simulationen (Spalten)

    plt.figure(figsize=(10, 6))

    # Plot der einzelnen Simulationen
    for i in range(n_sim):
        plt.plot(prices[:, i], color='blue', alpha=0.1)  # Jede Simulation in Blau mit geringer Opazität

    # Plot der mittleren Preisentwicklung (Durchschnitt der Simulationen)
    avg_prices = np.mean(prices, axis=1)
    plt.plot(avg_prices, color='red', label='Durchschnitt', linewidth=2)

    plt.title('Monte Carlo Simulation von Asset-Preisen')
    plt.xlabel('Zeit')
    plt.ylabel('Preis')
    plt.legend()
    plt.grid(True)
    plt.show()

# Beispiel-Aufruf:
N = 1000  # Anzahl der Simulationen
T = 252   # Anzahl der Zeitpunkte (z.B. 252 Handelstage in einem Jahr)
initial_price = 100  # Initialer Preis
mu = 0.05  # Erwartete Rendite (5%)
sigma = 0.2  # Volatilität (20%)

# Berechnungen durchführen
prices = compute_parallel(N, T, initial_price, mu, sigma)

# Simulationsergebnisse plotten
plot_monte_carlo_results(prices)


