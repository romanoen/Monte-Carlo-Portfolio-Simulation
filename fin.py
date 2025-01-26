import yfinance as yf
import numpy as np
import plotly.graph_objects as go

ticker = "AAPL"

data = yf.download(ticker, start="2010-01-01", end="2025-01-01")

data['Daily Returns'] = data[('Close', ticker)].pct_change()

mu = data['Daily Returns'].mean()
sigma = data['Daily Returns'].std()

initial_price = data['Close'].iloc[-1]  # Der letzte Kurs in den historischen Daten
T = 252  # Anzahl der Handelstage in einem Jahr (252 Handelstage)
N = 100000  # Anzahl der Simulationen (Pfad-Wiederholungen)
r_f = 0.02 # Risikofreier Zins

simulated_prices = np.zeros((T, N)) 
initial_price

for i in range(N):
    # Initialwert für den Preis (Startwert)
    prices = np.zeros(T)
    prices[0] = float(initial_price.iloc[0])

    # Generiere die Kursentwicklung für jeden Tag
    for t in range(1, T):
        # Ziehe eine Zufallszahl Z aus einer Normalverteilung N(0, 1)
        Z = np.random.normal(0, 1)
        # Berechne den nächsten Preis
        prices[t] = prices[t-1] * np.exp((mu - 0.5 * sigma**2) * 1 + sigma * Z)

    simulated_prices[:, i] = prices  # Speichere den simulierten Pfad

fig = go.Figure()

# Füge die simulierten Kurse als Linie zu Plotly hinzu
for i in range(N):
    fig.add_trace(go.Scatter(
        x=np.arange(T), 
        y=simulated_prices[:, i],
        mode='lines',
        line=dict(color='rgba(0, 0, 255, 0.1)', width=1),  # rgba mit Transparenz
    ))

# Hinzufügen der Titel und Achsenbeschriftungen
fig.update_layout(
    title=f"Monte-Carlo-Simulation für {ticker} ({N} Simulationen)",
    xaxis_title="Handelstage",
    yaxis_title="Aktienkurs",
    showlegend=False
)

# Anzeige der interaktiven Grafik
fig.show()

# Endwerte der Simulation (Kurse am letzten Handelstag)
final_prices = simulated_prices[-1, :]

# Mittelwert und Standardabweichung der Endpreise
mean_price = np.mean(final_prices)
std_price = np.std(final_prices)

print(f"Simulierte durchschnittliche Endkurs (nach {T} Tagen): {mean_price:.2f}")
print(f"Simulierte Standardabweichung des Endkurses: {std_price:.2f}")

# 95% Konfidenzintervall (zwei Standardabweichungen)
conf_interval = [mean_price - 2*std_price, mean_price + 2*std_price]
print(f"95% Konfidenzintervall: {conf_interval}")
