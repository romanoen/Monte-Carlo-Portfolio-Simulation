# **Monte Carlo Simulation for Portfolio Risk Analysis**

Monte Carlo simulations estimate the future distribution of portfolio returns by randomly generating thousands of potential price paths based on historical data and statistical assumptions. This technique is particularly useful in risk management, as it allows for the estimation of probabilities of extreme losses.

## **1. How to Start the Monte Carlo Simulation**

On the left of this application, you can see a sidebar with parameters. To try it out, you do not have to set up anything—just click on **Start Simulation** to see how the simulation works! 

### **1.1. Define Model Parameters**
- **Number of years**: Use past returns of the last $n$ years to estimate $\mu$ and $\sigma$.
- **Number of simulations**: Typically 1,000 – 100,000 paths. The more, the more accurate. 
- **Simulation duration**: Define how many months/years you want to simulate into the future.
- **Note**: Timesteps are set to one day and are not changeable.

### **1.2. What Happens Behind the Curtains**

For each path:
1. Generate random shocks $Z_t \sim N(0,1)$.
2. Compute stock price using the GBM equation.
3. Repeat for all assets in the portfolio.

### **1.3. Your Job: Analyze Results**
- Compute expected return $E[R]$.
- Calculate risk measures ($\text{VaR}$, $\text{CVaR}$, $\text{MDD}$).
- Optimize portfolio allocation based on risk-adjusted returns.

Monte Carlo simulations provide a probabilistic view of portfolio performance, helping investors assess uncertainty and make data-driven decisions. 

**Note**: This tool does not provide stock simulation!

---

## **2. Mathematical Foundation of Monte Carlo Simulations**

Monte Carlo simulations rely on **stochastic processes**, particularly **Geometric Brownian Motion (GBM)**, which models stock prices as:

$$
S_{t+\Delta t} = S_t e^{\left( \mu - \frac{1}{2} \sigma^2 \right) \Delta t + \sigma \sqrt{\Delta t} Z}
$$

Where:
- $S_t$ is the stock price at time $t$.
- $\mu$ is the expected return (drift).
- $\sigma$ is the standard deviation (volatility).
- $\Delta t$ is the time increment.
- $Z \sim N(0,1)$ is a standard normal random variable.

Each simulation run generates a possible price path based on this stochastic model.

---

## **3. Understanding the Plots**

Monte Carlo simulations often include several key visualizations:

### **3.1. Histogram of Historical Returns and Normal Distribution**
- Displays the empirical distribution of past returns.
- Often compared to a normal distribution $N(\mu, \sigma^2)$.
- A strong deviation from normality indicates potential non-Gaussian risks (e.g., fat tails).

### **3.2. Monte Carlo Simulation Paths**
- Simulated stock prices follow the stochastic differential equation (SDE) above.
- Thousands of potential paths $S_t$ are generated.
- The **mean trajectory** provides an expected price evolution, while **dispersion** indicates risk.

### **3.3. Q-Q Plot: Empirical vs. Theoretical Quantiles**
- A quantile-quantile (Q-Q) plot tests the normality assumption.
- If returns are normally distributed, points align with the diagonal.
- Deviations indicate skewness and kurtosis in asset returns.

---

## **4. Risk Metrics from Simulation**

When analyzing the potential risks associated with a portfolio or investment strategy, several key metrics are derived from simulations of asset paths. These metrics provide valuable insights into the potential losses an investor might face under adverse conditions. Below are the key risk metrics computed from the generated paths:

### **4.1. Value at Risk (VaR)**

Value at Risk (VaR) is one of the most commonly used risk metrics. It estimates the potential loss in value of a portfolio over a given time horizon at a specific confidence level. 

Mathematically, for a confidence level $\alpha$ (e.g., 5%), VaR is defined as:

$$
\text{VaR}_{\alpha} = \Phi^{-1}(\alpha) \cdot \sigma - \mu
$$

Where:
- $\Phi^{-1}(\alpha)$ is the inverse of the cumulative distribution function (CDF) of the normal distribution at the confidence level $\alpha$. 
- $\sigma$ is the standard deviation of the asset returns (representing volatility).
- $\mu$ is the expected return (mean).

**Interpretation:** VaR provides a threshold value below which the portfolio's loss is expected to occur with probability $\alpha$. For example, a 5% VaR of $1,000 means that there is a 5% chance that the portfolio will lose more than $1,000 in value over the given period.

**Example Usage:** A portfolio with a 1-day VaR at the 5% confidence level of $1,000 implies that, with 95% confidence, the portfolio will not lose more than $1,000 in a single day.

---

### **4.2. Conditional Value at Risk (CVaR)**

Conditional Value at Risk (CVaR), also known as Expected Shortfall (ES), measures the average loss assuming that the loss is beyond the VaR threshold. It gives an indication of the severity of losses in the tail of the distribution.

Mathematically, for a confidence level $\alpha$, CVaR is defined as:

$$
\text{CVaR}_{\alpha} = \frac{1}{1-\alpha} \int_{\alpha}^{1} \text{VaR}_p \, dp
$$

Where:
- $\text{VaR}_p$ is the VaR at a different probability level $p$.

**Interpretation:** CVaR provides the expected loss assuming that losses have exceeded the VaR level. It takes into account the severity of tail events, thus offering more insight than VaR alone. CVaR is particularly useful for understanding risk during extreme market events.

**Example Usage:** If the 5% VaR of a portfolio is $1,000, the 5% CVaR might suggest that, in the worst 5% of cases, the average loss exceeds $1,500.

---

### **4.3. Maximum Drawdown (MDD)**

The Maximum Drawdown (MDD) measures the largest peak-to-trough loss observed over a specified period. It quantifies the most significant potential loss an investor could have faced if they had invested at the highest point and then sold at the lowest point in the drawdown period.

Mathematically, MDD is defined as:

$$
\text{MDD} = \max \left( \frac{S_{\max} - S_t}{S_{\max}} \right)
$$

Where:
- $S_{\max}$ is the highest value of the portfolio or asset price observed before a decline (the peak).
- $S_t$ is the value of the portfolio or asset price at time $t$ (the trough).

**Interpretation:** MDD provides a measure of risk that captures the worst drawdown an investor could experience, which is critical for understanding the potential for loss over time. It helps in assessing the historical volatility and the magnitude of losses during periods of market downturns.

**Example Usage:** If an investor experienced a portfolio peak of $10,000 and a subsequent trough of $6,000, the MDD would be:

$$
\text{MDD} = \frac{10,000 - 6,000}{10,000} = 0.4 \quad \text{or} \quad 40\%
$$

This means the portfolio lost 40% of its value from its peak to its lowest point.


