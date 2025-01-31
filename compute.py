import numpy as np


def multi_monte_carlo_sim(T: int, N: int, asset_metrics: dict):
    """
    asset_metrics structure:
    {
    "AAPL": {
        "mu": 0.0005,       # Durchschnittliche tägliche Rendite für AAPL
        "sigma": 0.02,      # Standardabweichung der täglichen Rendite für AAPL
        "initial_price": 150.25  # Aktuelles Kapital in dieser Aktie
        "returns": [0.003, 0.005, 0.002 ..] # Liste von historischen Renditen
    },
    "GOOG": {
        "mu": 0.0007,       # Durchschnittliche tägliche Rendite für GOOG
        "sigma": 0.015,     # Standardabweichung der täglichen Rendite für GOOG
        "initial_price": 2800.50  # Aktuelles Kapital in dieser Aktie
        "returns": [0.003, 0.005, 0.002 ..] # Liste von historischen Renditen
    }
}
    """
    compute_cholesky_from_returns(asset_metrics)

def monte_carlo_simulation(T: int, N: int, initial_price, mu, sigma):
    """
    Führt eine Monte-Carlo-Simulation für Aktienkurse durch.

    Parameters:
        T (int): Anzahl der Tage pro Simulation.
        N (int): Anzahl der Simulationsläufe.
        initial_price (float): Startpreis der Aktie.
        mu (float): Erwartete Rendite (Drift).
        sigma (float): Volatilität.

    Returns:
        np.ndarray: Ein 2D-Array der simulierten Preise (T x N).
    """
    simulated_prices = np.zeros((T, N))  # Jede Spalte ist eine Simulation

    for i in range(N):
        # Initialwert für den Preis (Startwert)
        prices = np.zeros(T)
        prices[0] = initial_price

        # Generiere die Kursentwicklung für jeden Tag
        for t in range(1, T):
            # Ziehe eine Zufallszahl Z aus einer Normalverteilung N(0, 1)
            Z = np.random.normal(0, 1)
            # Berechne den nächsten Preis
            prices[t] = prices[t-1] * np.exp((mu - 0.5 * sigma**2) * 1 + sigma * Z)

        simulated_prices[:, i] = prices  # Speichere die Simulation (spaltenweise)

    return simulated_prices

def compute_cholesky_from_returns(asset_metrics: dict):
    """
    Berechnet die Korrelationsmatrix aus den historischen Renditen der Assets
    und führt die Cholesky-Zerlegung durch.

    Parameters:
        asset_metrics (dict): Dictionary mit Asset-Daten inklusive "returns".

    Returns:
        np.ndarray: Cholesky-Dekomposition der Korrelationsmatrix.
        np.ndarray: Korrelationsmatrix.
    """
    # Sammle alle Renditen als NumPy-Arrays
    returns_list = [np.array(metrics["returns"]) for metrics in asset_metrics.values()]

    # Bestimme die minimale Länge unter allen Renditen
    min_length = min(len(returns) for returns in returns_list)

    # Trimme alle Renditen auf die gleiche Länge
    trimmed_returns = [returns[-min_length:] for returns in returns_list]  # Letzte min_length Werte

    # Erstelle eine Matrix mit den Renditen als Zeilen
    returns_matrix = np.vstack(trimmed_returns)  # (Anzahl Assets, Anzahl Tage)

    # Berechne die Korrelationsmatrix
    correlation_matrix = np.corrcoef(returns_matrix)

    # Cholesky-Zerlegung der Korrelationsmatrix
    L = np.linalg.cholesky(correlation_matrix)

    print(correlation_matrix)
    print(L)
    return L, correlation_matrix

