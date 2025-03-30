# **Monte Carlo Simulation for Portfolio Risk Analysis**

Monte Carlo simulations estimate the future distribution of portfolio returns by randomly generating thousands of potential price paths based on historical data and statistical assumptions. This technique is particularly useful in risk management, as it allows for the estimation of probabilities of extreme losses.

## **1. Mathematical Foundation of Monte Carlo Simulations**
Monte Carlo simulations rely on **stochastic processes**, particularly **Geometric Brownian Motion (GBM)**, which models stock prices as:

\[
S_{t+\Delta t} = S_t e^{\left( \mu - \frac{1}{2} \sigma^2 \right) \Delta t + \sigma \sqrt{\Delta t} Z}
\]

where:
- \( S_t \) is the stock price at time \( t \).
- \( \mu \) is the expected return (drift).
- \( \sigma \) is the standard deviation (volatility).
- \( \Delta t \) is the time increment.
- \( Z \sim N(0,1) \) is a standard normal random variable.

Each simulation run generates a possible price path based on this stochastic model.

## **2. Understanding the Plots**
Monte Carlo simulations often include several key visualizations:

### **Histogram of Historical Returns and Normal Distribution**
- Displays the empirical distribution of past returns.
- Often compared to a normal distribution \( N(\mu, \sigma^2) \).
- A strong deviation from normality indicates potential non-Gaussian risks (e.g., fat tails).

### **Monte Carlo Simulation Paths**
- Simulated stock prices following the stochastic differential equation (SDE) above.
- Thousands of potential paths \( S_t \) are generated.
- The **mean trajectory** provides an expected price evolution, while **dispersion** indicates risk.

### **Q-Q Plot: Empirical vs. Theoretical Quantiles**
- A quantile-quantile (Q-Q) plot tests the normality assumption.
- If returns are normally distributed, points align with the diagonal.
- Deviations indicate skewness and kurtosis in asset returns.

### **Risk Metrics from Simulation**
From the generated paths, key risk metrics are computed:

#### **1. Value at Risk (VaR)**
VaR at confidence level \( \alpha \) (e.g., 5%) is defined as:

\[
\text{VaR}_{\alpha} = \Phi^{-1}(\alpha) \cdot \sigma - \mu
\]

where \( \Phi^{-1} \) is the inverse cumulative distribution function (CDF) of the normal distribution.

#### **2. Conditional Value at Risk (CVaR)**
CVaR provides the expected loss beyond VaR:

\[
\text{CVaR}_{\alpha} = \frac{1}{1-\alpha} \int_{\alpha}^{1} \text{VaR}_p \, dp
\]

#### **3. Maximum Drawdown**
The maximum drawdown (MDD) measures the largest peak-to-trough loss:

\[
\text{MDD} = \max \left( \frac{S_{\max} - S_t}{S_{\max}} \right)
\]

where \( S_{\max} \) is the highest price observed before a decline.

## **3. Setting Up a Monte Carlo Simulation**
To implement a Monte Carlo simulation for portfolio risk analysis:

### **1. Define Model Parameters**
- **Historical data**: Use past returns to estimate \( \mu \) and \( \sigma \).
- **Number of simulations**: Typically 1,000 – 100,000 paths.
- **Time steps**: Daily (\(\Delta t = 1/252\)) or monthly (\(\Delta t = 1/12\)).

### **2. Run Simulations**
For each path:
1. Generate random shocks \( Z_t \sim N(0,1) \).
2. Compute stock price using GBM equation.
3. Repeat for all assets in the portfolio.

### **3. Analyze Results**
- Compute expected return \( E[R] \).
- Calculate risk measures (VaR, CVaR, MDD).
- Optimize portfolio allocation based on risk-adjusted returns.

Monte Carlo simulations provide a probabilistic view of portfolio performance, helping investors assess uncertainty and make data-driven decisions.
