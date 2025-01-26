import numpy as np

def monte_carlo_simulation(T, N, initial_price, mu, sigma):
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

