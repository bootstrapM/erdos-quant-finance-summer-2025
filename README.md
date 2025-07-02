# Quantitative Finance Projects at Erdos Institute, Summer 2025 

## Overview

This submission features four rigorous quantitative finance projects that applied mathematical and computational tools to address problems in financial modeling and risk management. The first project focused on constructing high- and low-risk portfolios across diversified and tech-heavy stock baskets using quadratic optimization. Results demonstrated strong in-sample performance (Sharpe ratio up to 2.42), with expected degradation during backtesting—highlighting the importance of robust validation. The second project investigated the assumption of normality in the log returns of financial data. Through rolling Shapiro-Wilk p-value tests on log returns (2015–2025), transient periods of normality were found. We demonstrated how these periods can be exploited to build portfolios with some statistical evidence of normality. The third project analyzed the sensitivity of Black-Scholes price of option on time to expiration and spot price. Finally, the fourth project simulated delta hedging in stochastic volatility environments (via a custom and GARCH(1,1) models), showing that while hedging reduces variance and tail risk, it cannot eliminate risk exposure entirely unlike the idealized Black-Scholes setting.

## Repository Structure

erdos-quant-finance-summer-2025/
├── README.md
├── portfolio_analysis_functions.py
├── Function-Mini-Project-4.py
├── Project 1-Portfolio-Analysis-Final.ipynb
├── Project 2-Final.ipynb
├── Project 3-Final.ipynb
└── Project-4-Final.ipynb



## Summary of Project 1 (Portfolio Construction)

In this mini project we created various portfolios with low an high risk using stock data. Last 3.5 years of data was used to create (first three years) and backtest (last six months) the portfolios. We considered two universes / basket of stocks. The first basket is diversified by sector (Tech, Finance, Healthcare, Energy, Consumer Staples). The considered stocks were
```
diversified_tickers = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',  # Tech    
    'JPM', 'BAC', 'GS', 'MS', 'WFC',          # Finance    
    'JNJ', 'PFE', 'MRK', 'UNH', 'ABBV',       # Healthcare    
    'XOM', 'CVX', 'COP', 'SLB', 'EOG',        # Energy    
    'PG', 'KO', 'PEP', 'WMT', 'COST'          # Consumer Staples]
```
The second basket is built out of tech stocks:
```
tech_tickers = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',  # Big Tech
    'ADBE', 'CRM', 'ORCL', 'INTC', 'AMD', 'AVGO', 'QCOM', 
    'CSCO', 'IBM', 'TXN', 'AMAT', 'NOW', 'SHOP', 'UBER', 
    'SNOW', 'ZM', 'DOCU', 'PYPL', 'NFLX'
]
```

For portfolio construction we analyze the return series $r_t^i \equiv \frac{P_t^i}{P_{t-1}^i}$ where $t$ is the time index and $i$ labels the stock. Denote $w_i$ as the weight of the stock in the portfolio the return series of the portfolio is given by $R_t = r_t^i w_i$. We constrain $w_i\in[-1,1]$. Negative values corresponds to short position in the $i$'th stock. Further we impose that the portfolio is fully invested in all the stocks $\sum_i w_i = 1$. The expected return of the portfolio is then 

$$
\bar{R} = \mathbb{E}[r^i_t] w_i
$$

and its variance is given by

$$
\mathbb{V}[R] = w_i C^{ij}w_j~,\qquad \qquad C^{ij} = \mathbb{E}[(r^i-\bar{r^i})(r^j-\bar{r^j})]~,\qquad \text{(covaraince matrix of returns)}
$$

We solve the following optimization problem

$$
\text { Minimize: }-\mathbb{E}[R]+\lambda|w|^2~,\qquad \text{subjected to the risk tolerance level}~,\mathbb{V}[R]<\sigma^2
$$

where $\lambda$ is the L2 regularization strength (l2_reg=0.1 set by default). The associated $|w|^2$ term penalizes concentrated positions and encourages diversification. We construct a total of 8 portfolios. Following four types in each basket:

- Low-risk, long-only
- Low-risk, long-short
- High-risk, long-only
- High-risk, long-short

where the risk levels are defined as

$$
\sqrt{\mathbb{V}[R]} < 0.08 ~,\qquad \qquad \text{Low Risk}~,
$$

$$
\sqrt{\mathbb{V}[R]} > 0.25 ~,\qquad \qquad \text{High Risk}
$$

The negative weights $w_i<0$ are bounded by below to avoid large short positions. After optimizing the weights we get the following portfolios

### Basket 1 Long-only

![Screenshot 2025-06-28 at 20 34 28](https://github.com/user-attachments/assets/5ac31675-1191-42cf-8c5b-5a612a95b314)

![Screenshot 2025-06-28 at 20 34 48](https://github.com/user-attachments/assets/4584adad-5626-42f5-a6e1-ce0f6c540038)

### Basket 1 Long-short

![Screenshot 2025-06-28 at 20 36 09](https://github.com/user-attachments/assets/6627101e-9aaf-4f9a-b756-6ff401577a16)

![Screenshot 2025-06-28 at 20 36 26](https://github.com/user-attachments/assets/979ca274-eb50-47e5-8f65-54e912a8d0da)

### Basket 2 Long-only

![Screenshot 2025-06-28 at 20 37 29](https://github.com/user-attachments/assets/ef970fc5-2a8b-46e8-ad02-7a543f186e98)

![Screenshot 2025-06-28 at 20 37 57](https://github.com/user-attachments/assets/5fffc044-96ac-4c28-b069-4c3157083722)


### Basket 2 Long-short

![Screenshot 2025-06-28 at 20 38 28](https://github.com/user-attachments/assets/9a8d79f5-4422-4206-80d2-93e94940a1e3)

![Screenshot 2025-06-28 at 20 39 07](https://github.com/user-attachments/assets/c7a43e19-5b16-4d6e-a9d4-c0d5fa9f9efb)

## In-sample cumulative returns 

### Basket 1
![Screenshot 2025-06-28 at 20 41 02](https://github.com/user-attachments/assets/b36e523c-0004-4d7d-b7ff-b837e55be518)

### Basket 2

![Screenshot 2025-06-28 at 20 41 20](https://github.com/user-attachments/assets/4aa459c7-ab96-44fd-b088-17d6a30c6fca)

## Backtesting (Analyzing portfolio performance during last six months)

### Basket 1

<img width="1206" alt="image" src="https://github.com/user-attachments/assets/17bfae4c-7a1a-416a-84a0-99192422b9f9" />

The Sharpe ratio decreased from 1.70 in the construction period to 0.91 during the backtesting period

### Basket 2

<img width="1211" alt="image" src="https://github.com/user-attachments/assets/de24cc11-6ca4-442e-aa6e-140ce3695c7d" />

The Sharpe ratio decreased from 2.42 in the construction period to 1.2 during the backtesting period

## Forecasting

<img width="1244" alt="image" src="https://github.com/user-attachments/assets/80effdae-f3eb-4cdd-b181-6c1a27ceb475" />


## Summary of Project 2 (Normality Testing)

In this mini project we address to what extent is the financial data normally distributed. Historical log return of a stock/index data shows strong evidence against normality as is evident from the plots below. 

![Screenshot 2025-06-28 at 20 51 01](https://github.com/user-attachments/assets/4c41e79c-3cb0-4c4f-914b-54383e2c77a9)

Here we show that the log return distribution and QQ plots for AAPL, MSFT, GOOGL and NVDA from 2015-01-01 to 2025-01-01. However there can be periods of time where the log returns data shows evidence of being normally distributed. We consider the daily log-return data of stocks: ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'JPM', 'NVDA', 'SPY', ‘QQQ'] from 2015-01-01 to 2025-01-01 and perform of a Rolling Shapiro-Wilk p-values with window of 252 days

![Screenshot 2025-06-28 at 20 55 55](https://github.com/user-attachments/assets/3a1e3508-4e2a-4640-a242-7f2b95a20728)

We observe that there are one-year periods where Shapiro-Wilk p-value p>0.05. Therefore cannot reject that the data is normal. However need to do more tests in order to gather evidence for normality. From the Rolling Shapiro-Wilk p-value test get the following periods for various stocks (ranked by period length) for which Anderson-Darling, D'Agostino-Pearson and Kolmogorov-Smirnov passes. 

- GOOGL: 2022-12-12 to 2023-04-25 (~90+ days)
  ```
  Shapiro-Wilk: stat=0.9800, p=0.1687
  Anderson-Darling: stat=0.3585
  D'Agostino-Pearson: stat=5.3203, p=0.0699
  Kolmogorov-Smirnov: stat=0.0605, p=0.8684
  ```
  
- AAPL: 2022-12-30 to 2023-03-15 (~50+ days)
  ```
  Shapiro-Wilk: stat=0.9863, p=0.8177
  Anderson-Darling: stat=0.2424
  D'Agostino-Pearson: stat=0.2361, p=0.8887
  Kolmogorov-Smirnov: stat=0.0872, p=0.8011
  ```
  
- TSLA: 2022-11-07 to 2023-01-03 (~40 days)
  ```
  Shapiro-Wilk: stat=0.9732, p=0.4691
  Anderson-Darling: stat=0.3494
  D'Agostino-Pearson: stat=0.2776, p=0.8704
  Kolmogorov-Smirnov: stat=0.0803, p=0.9456
  ```
  
- MSFT: 2021-11-11 to 2022-01-03 (~35 days)
  ```
  Shapiro-Wilk: stat=0.9752, p=0.5842
  Anderson-Darling: stat=0.3011
  D'Agostino-Pearson: stat=0.1215, p=0.9411
  Kolmogorov-Smirnov: stat=0.0879, p=0.9207
  ```

Below we show the QQ plots for these time windows

<img width="1004" alt="image" src="https://github.com/user-attachments/assets/b2919d6b-1ed2-462d-b59f-d079b1b1337b" />

In these periods there were no significant extreme returns suggesting that removing extremal returns creates a distribution with evidence of being normal. We test this below

![Screenshot 2025-06-28 at 21 05 40](https://github.com/user-attachments/assets/d0b00808-093c-4366-b4ba-d24a5a129f5a)

We also made an effort to construct a portfolio of stocks with historical log return data that is normally distributed. Based on the Shapiro-Wilk p-value test, we searched for the longest window containing at least 5 stocks that passes the test. We then construct an equal weight portfolio using these stocks:

```
Largest window passing the rolling Shapiro-Wilk p-value test: 2023-01-04 to 2023-02-28 (56 days)
Stocks: JPM, SPY, GOOGL, NVDA, AAPL, MSFT, QQQ
```

The QQ plot of the portfolio containing these stock in equal weigh for the period 2023-01-04 to 2023-02-28 is shown below

<img width="611" alt="image" src="https://github.com/user-attachments/assets/22636cf6-fbbc-41a2-bbf5-d19beae98220" />

The result of the normality test are as follows:

```
    Normality tests for equal-weighted portfolio from 2023-01-04 to 2023-02-28:
    Shapiro-Wilk: stat=0.9720, p=0.4473
    Anderson-Darling: stat=0.2662
    D'Agostino-Pearson: stat=3.6089, p=0.1646
    Kolmogorov-Smirnov: stat=0.0904, p=0.8877
```

Running a rolling Shapiro-Wilk p-value on the portfolio shows more periods of normality 

<img width="997" alt="image" src="https://github.com/user-attachments/assets/81797278-48e8-4b74-935b-315f58354f74" />

## Summary of Project 3

In this project we analyzing the sensitivity of Black-Scholes call and put prices on time to expiry and spot price of the underlying. Our observations are

- *Call Option – Time Sensitivity:*  We observe that the rate of change of the BS call with respect to time is negative and more pronounced near maturity for ATM calls but decreases towards zero for ITM and OTM calls
  
- *Call Option – Spot Price Sensitivity:*  We observe that the rate of change of the BS call with respect to option price is positive and monotonic increasing function regardless the value of the strike price. Approaches 0 for deep out-of-the-money calls and 1 for deep in-the-money calls. Most sensitive (steepest slope) when at-the-money.
  
- *Put Option – Time Sensitivity:*  We observe that the rate of change of the BS put with respect to time is mostly negative and more pronounced near maturity for ATM puts but decreases towards zero for ITM and OTM puts
  
- *Put Option – Spot Price Sensitivity:*  We observe that the rate of change of the BS put with respect to option price is negative and monotonic increasing function regardless the value of the strike price. Approaches -1 for deep in-the-money puts and 0 for deep out-the-money calls. Most sensitive (steepest slope) when at-the-money.


## Summary of Project 4

In this project we investigated the process of hedging in a stochastic volatility environment. We explored how to delta hedge sold call options built on stock whose volatility is non-constant. In particular we analyzed the following two models of stocks whose log returns follows the process 

$$
\Delta \ln S_{j}=\left(\mu+r-\frac{1}{2} \sigma_{j}^2\right) \Delta t+\sigma_{j} \sqrt{\Delta t} \cdot Z_{j}
$$

but the volatilites in the two models are drawn from the following distributions:
- $\sigma_j$ is drawn from the set $\{.2,.3,.45\}$ with probabilities $\{.5, .3, .2\}$ respectively
- $\sigma_j$ is given by the GARCH(1,1) process

$$
r_t=\sigma_t \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0,1)
$$

$$
\sigma_t^2=\omega+\alpha r_{t-1}^2+\beta \sigma_{t-1}^2
$$

We start the simulation by setting $\sigma_0 = \omega/(1-\alpha-\beta)$ the unconditional variance if $\omega \neq 0$ and $\alpha+\beta <1$ else use $\sigma_0 = 0.01$.


The stock price process in the first model is shown below (for five paths)

<img width="736" alt="image" src="https://github.com/user-attachments/assets/75e8c25a-bfdc-4e6f-b189-df6bec25497d" />

We find the value of the European call $V = e^{-rT}\mathbb{E}_\mathbb{Q}[\text{max}(S_T-K)]$ via monte carlo simulations. The result for K=100, and 100,000 simulations without $\Delta$￼hedging is: $13.12 \pm 0.06$. Below we implement the $\Delta$ hedging procedure for which much less simulations will be needed

### $\Delta$ Hedging

To protect from huge losses, the seller maintains a portfolio of options and underlying stock that need to be rebalanced regularly. In an n-step hedging process 

$$
P = \text{premium}-e^{-rT}C_T + \sum_{i=i}^n e^{-rt_i}\{S_{t_i} - e^{r(t_i-t_{i-1})}S_{t_{i-1}}\}\Delta_{t_{i-1}}
$$

with 

$$
\Delta_{t_{i}} = \frac{\partial C}{\partial S}\bigg|_{t=t_{i}}
$$

In models where the analytical depedence of the option value on the spot price in not known we compute $\Delta$ numerically using the finite difference:

$$
\Delta_{C_0}\approx \frac{C_0(S_0+\varepsilon) - C_0(S_0-\varepsilon)}{2\varepsilon}
$$

### Call option on stock with volatility drawn from the custom distribution
Below we show the result of the 1000 MC simulation of the value of call option as we increase the number of hedges

<img width="1309" alt="image" src="https://github.com/user-attachments/assets/9044b5e3-d208-4e71-9529-550cf6ab899c" />

We observe that the hedging procedure decreases the variance in the value of the price. Note that unlike in Black-Scholes, $\Delta$ heding will not eliminate the risk completely. 

We also consider the profit and loss distribution of the seller of the call option where she sell 100 contracts at a premium given by

```
   black_scholes_call(S0, K, t, r, bs_sigma=0.5)= 20.96
```

P&L distribution without $\Delta$ hedging is

<img width="872" alt="image" src="https://github.com/user-attachments/assets/6aa06c25-f90b-43c2-86fb-826aa5e619cb" />

P&L distribution with 252-fold $\Delta$ hedging (in 1 year) is 

<img width="789" alt="image" src="https://github.com/user-attachments/assets/e6f6ca0f-1778-43d3-b752-1731ada81962" />

### Call option on stock with volatility following GARCH(1,1) dynamics

We repeated this analysis for the option on the stock following GARCH(1,1) dynamics. The model parameters were

```
  omega = 0, alpha = 0.1, beta = 0.88
  S_0 = 100, K=100, T=1, r=0.05, mu=0
```

The MC result with 10,000 simulations without $\Delta$ hedging: 5.16 $\pm$ 0.0395. This is close to the Black-Scholes value of 5.07 with volatility = average GARCH volatility 0.04 (for the chosen GARCH parameters). With $\delta$ hedging we get the following results. 

<img width="1435" alt="image" src="https://github.com/user-attachments/assets/86f4cb2d-49c0-49af-a81a-9f9762405a05" />


Again we analyzed the profit and loss distribution of the seller of the call option where she sell 100 contracts at a premium given by

```
black_scholes_price(S0, K, T, r, avg_volatility+0.01)= 5.28
```

where the `avg_volatility=0.04` as mentioned above. The P&L distribution without $\Delta$ hedging is

<img width="937" alt="image" src="https://github.com/user-attachments/assets/a06ec1fc-722b-411b-ae26-626b1e302388" />



P&L distribution with 252-fold $\Delta$ hedging (in 1 year) is 

<img width="836" alt="image" src="https://github.com/user-attachments/assets/12d42401-57aa-4c4c-a246-8d3a594b596a" />


We observe that the losses are significantly reduced with $\Delta$ hedging 

## Some questions for the future

- Implement Delta + Sigma + Gamma hedging in stochastic volatility models like GARCH(1,1) or Heston where delta hedging is not enough to eliminate risk completely. 
- In practice delta or delta-vega hedging are inadequate due to real-world complexities (for instance transaction costs involved in the hedging procedure, stochastic volatility or jumps in the assest price). Deep hedging makes use of deep learning methods to construct optimal hedging strategies in financial markets, especially to deal with such complexities.
- Explore the use of reinforcement learning for portfolio optimization and option pricing.











