import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from datetime import datetime, timedelta
import scipy.stats as stats
import warnings

def get_stock_data(tickers, start_date, end_date):
    """
    Fetch historical stock data for multiple tickers.
    Returns a DataFrame with closing prices.
    """
    data = pd.DataFrame()
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            data[ticker] = df['Close']
        except Exception as e:
            print(f"Error fetching data for {ticker}: {str(e)}")
    return data

def calculate_returns(prices):
    """
    Calculate daily returns from price data.
    We use simple returns r_t = P_t/P_{t-1}-1 for portfolio analysis
    """
    return prices.pct_change().dropna()

def calculate_portfolio_metrics(returns, weights):
    """
    Calculate portfolio metrics including expected return, volatility, and Sharpe ratio.

    Input: Pandas dataframe consisting of daily returns of all the stocks in the portfolio.
    Input: Weights of the portfolio.

    Output: Dictionary of portfolio metrics: expected_return, volatility, sharpe_ratio
    """
    # Expected annual return
    expected_return = np.sum(returns.mean() * weights) * 252
    
    # Portfolio volatility (annualized)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    
    # average 3-month US Treasury bill rate from December 2020 to June 2024 
    # (note that there was huge variation in the risk-free interest rate from nrear 0% in 2020–2021 due to pandemic-era monetary policy to 4–5% in 2023–2024)
    # source: https://ycharts.com/indicators/3_month_t_bill
    risk_free_rate = 0.025

    # Sharpe ratio (assuming risk-free rate of 0.025)
    sharpe_ratio = (expected_return - risk_free_rate) / portfolio_volatility
    
    return {
        'expected_return': expected_return,
        'volatility': portfolio_volatility,
        'sharpe_ratio': sharpe_ratio
    }

def optimize_portfolio(returns, max_volatility=0.15, allow_short=False, l2_reg=0.1):
    """
    Maximize expected return subject to portfolio volatility being below max_volatility,
    with L2 regularization to encourage diversification.
    Returns the optimal weights.
    l2_reg: regularization strength (higher = more diversification)
    """
    n_assets = len(returns.columns)
    
    def objective(weights):
        # Negative expected return + L2 penalty
        return -np.sum(returns.mean() * weights) * 252 + l2_reg * np.sum(weights**2)
    
    def volatility_constraint(weights):
        # Annualized portfolio volatility
        cov_matrix = returns.cov() * 252
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return max_volatility - port_vol
    
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # weights sum to 1
        {'type': 'ineq', 'fun': volatility_constraint}   # volatility <= max_volatility
    ]
    if allow_short:
        bounds = tuple((-0.2, 0.2) for _ in range(n_assets))
    else:
        bounds = tuple((0.01, None) for _ in range(n_assets))
    initial_weights = np.array([1/n_assets] * n_assets)
    result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

def create_portfolios(tickers, start_date, end_date, l2_reg=0.1):
    """
    Create and analyze four portfolios:
    (low risk, long only), (low risk, long-short), (high risk, long only), (high risk, long-short)
    Returns a dictionary of portfolios and their metrics.
    """
    prices = get_stock_data(tickers, start_date, end_date)
    returns = calculate_returns(prices)

    # Define risk levels (annualized volatility)
    low_risk_vol = 0.08
    high_risk_vol = 0.25

    # Optimize portfolios
    weights_low_long = optimize_portfolio(returns, max_volatility=low_risk_vol, allow_short=False, l2_reg=l2_reg)
    weights_low_short = optimize_portfolio(returns, max_volatility=low_risk_vol, allow_short=True, l2_reg=l2_reg)
    weights_high_long = optimize_portfolio(returns, max_volatility=high_risk_vol, allow_short=False, l2_reg=l2_reg)
    weights_high_short = optimize_portfolio(returns, max_volatility=high_risk_vol, allow_short=True, l2_reg=l2_reg)

    # Create portfolio DataFrames
    portfolios = pd.DataFrame({
        'Low Risk Long Only': weights_low_long,
        'Low Risk Long-Short': weights_low_short,
        'High Risk Long Only': weights_high_long,
        'High Risk Long-Short': weights_high_short
    }, index=tickers)

    # Calculate metrics for each portfolio
    metrics = {}
    for name, weights in portfolios.items():
        metrics[name] = calculate_portfolio_metrics(returns, weights)

    # Calculate individual stock metrics
    stock_metrics = pd.DataFrame({
        'Expected Return': returns.mean() * 252,
        'Volatility': returns.std() * np.sqrt(252),
        'Sharpe Ratio': (returns.mean() * 252 - 0.02) / (returns.std() * np.sqrt(252))
    })

    return portfolios, metrics, stock_metrics

def calculate_annual_returns(returns, weights, initial_investment=10000):
    """
    Calculate annual returns for a given investment amount.
    Uses log-normal confidence interval for more accurate bounds.
    """
    # Calculate portfolio returns
    portfolio_returns = returns.dot(weights)
    
    # Calculate annual metrics
    annual_return = portfolio_returns.mean() * 252
    annual_volatility = portfolio_returns.std() * np.sqrt(252)
    
    # Calculate expected value after one year
    expected_value = initial_investment * (1 + annual_return)
    
    # Calculate 95% confidence interval using log-normal formula
    lower_bound = initial_investment * np.exp(annual_return - 1.96 * annual_volatility)
    upper_bound = initial_investment * np.exp(annual_return + 1.96 * annual_volatility)
    
    return {
        'annual_return': annual_return,
        'annual_volatility': annual_volatility,
        'expected_value': expected_value,
        'confidence_interval': (lower_bound, upper_bound)
    }

def forecast_portfolio_returns(returns, weights, n_simulations=1000, forecast_days=252):
    """
    Forecast portfolio returns using Monte Carlo simulation.
    """
    # Calculate portfolio returns
    portfolio_returns = returns.dot(weights)
    
    # Calculate mean and standard deviation of returns
    mean_return = portfolio_returns.mean()
    std_return = portfolio_returns.std()
    
    # Generate simulations
    simulations = np.random.normal(
        mean_return,
        std_return,
        (n_simulations, forecast_days)
    )
    
    # Calculate cumulative returns
    cumulative_returns = (1 + simulations).cumprod(axis=1)
    
    # Calculate percentiles
    percentiles = np.percentile(cumulative_returns, [5, 25, 50, 75, 95], axis=0)
    
    return {
        'mean_return': mean_return,
        'std_return': std_return,
        'simulations': simulations,
        'cumulative_returns': cumulative_returns,
        'percentiles': percentiles
    }

def plot_forecast(forecast_results, actual_returns, title):
    """
    Plot forecast results with confidence intervals and overlay the actual realized path.
    """
    plt.figure(figsize=(12, 6))
    days = range(len(forecast_results['percentiles'][0]))
    # Plot forecast percentiles
    plt.fill_between(days, forecast_results['percentiles'][0], 
                     forecast_results['percentiles'][4], 
                     alpha=0.2, label='90% Confidence Interval')
    plt.fill_between(days, forecast_results['percentiles'][1], 
                     forecast_results['percentiles'][3], 
                     alpha=0.2, label='50% Confidence Interval')
    plt.plot(days, forecast_results['percentiles'][2], 
             label='Median Forecast', linewidth=2)
    # Plot actual realized path
    actual_cumulative = (1 + actual_returns).cumprod().values
    plt.plot(days[:len(actual_cumulative)], actual_cumulative, label='Actual Path', color='black', linewidth=2, linestyle='--')
    plt.title(f'6-Month Forecast vs Actual for {title}')
    plt.xlabel('Trading Days')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_portfolio_weights(portfolios):
    """
    Plot a pie chart of weights for each portfolio in the DataFrame if all weights are non-negative.
    Otherwise, use a bar chart for long-short portfolios.
    """
    for name in portfolios.columns:
        weights = portfolios[name]
        if (weights < 0).any():
            # Bar chart for long-short portfolios
            plt.figure(figsize=(12, 6))
            weights.plot(kind='bar', color=['#4ECDC4' if w >= 0 else '#FF6B6B' for w in weights], edgecolor='black')
            plt.title(f'{name} - Portfolio Allocation', fontsize=16, fontweight='bold')
            plt.ylabel('Weight')
            plt.xlabel('Stock')
            plt.axhline(0, color='black', linewidth=1)
            plt.tight_layout()
            plt.show()
        else:
            # Pie chart for long-only portfolios
            significant_weights = weights[weights.abs() > 0.005]
            other_weight = weights[weights.abs() <= 0.005].sum()
            labels = list(significant_weights.index)
            values = list(significant_weights.values)
            if abs(other_weight) > 0:
                labels.append('Others (<0.5%)')
                values.append(other_weight)
            plt.figure(figsize=(10, 7))
            colors = plt.cm.tab20.colors
            while len(colors) < len(labels):
                colors = colors + colors
            wedges, texts, autotexts = plt.pie(values, labels=labels, autopct='%1.1f%%',
                                               colors=colors[:len(labels)], startangle=90)
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(10)
            plt.title(f'{name} - Portfolio Allocation', fontsize=16, fontweight='bold')
            plt.axis('equal')
            plt.legend(wedges, labels, title="Stocks", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
            plt.tight_layout()
            plt.show()

def plot_portfolio_sector(portfolios):
    """
    Plot a pie chart of portfolio weights by sector for each portfolio in the DataFrame if all sector weights are non-negative.
    Otherwise, use a bar chart for long-short portfolios.
    """
    # Define sector mapping based on tickers
    sector_map = {
        'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology', 'AMZN': 'Technology', 'META': 'Technology',
        'NVDA': 'Technology', 'TSLA': 'Technology', 'ADBE': 'Technology', 'CRM': 'Technology', 'ORCL': 'Technology',
        'INTC': 'Technology', 'AMD': 'Technology', 'AVGO': 'Technology', 'QCOM': 'Technology', 'CSCO': 'Technology',
        'IBM': 'Technology', 'TXN': 'Technology', 'AMAT': 'Technology', 'NOW': 'Technology', 'SHOP': 'Technology',
        'UBER': 'Technology', 'SNOW': 'Technology', 'ZM': 'Technology', 'DOCU': 'Technology', 'SQ': 'Technology',
        'PYPL': 'Technology', 'NFLX': 'Technology',
        'JPM': 'Finance', 'BAC': 'Finance', 'GS': 'Finance', 'MS': 'Finance', 'WFC': 'Finance',
        'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'MRK': 'Healthcare', 'UNH': 'Healthcare', 'ABBV': 'Healthcare',
        'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy', 'SLB': 'Energy', 'EOG': 'Energy',
        'PG': 'Consumer Staples', 'KO': 'Consumer Staples', 'PEP': 'Consumer Staples', 'WMT': 'Consumer Staples', 'COST': 'Consumer Staples'
    }
    for name in portfolios.columns:
        weights = portfolios[name]
        # Aggregate weights by sector
        sector_weights = {}
        for ticker, weight in weights.items():
            sector = sector_map.get(ticker, 'Other')
            sector_weights[sector] = sector_weights.get(sector, 0) + weight
        sector_weights_series = pd.Series(sector_weights)
        if (sector_weights_series < 0).any():
            # Bar chart for long-short portfolios
            plt.figure(figsize=(10, 6))
            sector_weights_series.plot(kind='bar', color=['#4ECDC4' if w >= 0 else '#FF6B6B' for w in sector_weights_series], edgecolor='black')
            plt.title(f'{name} - Sector Allocation', fontsize=16, fontweight='bold')
            plt.ylabel('Weight')
            plt.xlabel('Sector')
            plt.axhline(0, color='black', linewidth=1)
            plt.tight_layout()
            plt.show()
        else:
            # Pie chart for long-only portfolios
            labels = list(sector_weights.keys())
            values = list(sector_weights.values())
            plt.figure(figsize=(10, 7))
            colors = plt.cm.Set3.colors
            while len(colors) < len(labels):
                colors = colors + colors
            wedges, texts, autotexts = plt.pie(values, labels=labels, autopct='%1.1f%%',
                                               colors=colors[:len(labels)], startangle=90)
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(10)
            plt.title(f'{name} - Sector Allocation', fontsize=16, fontweight='bold')
            plt.axis('equal')
            plt.legend(wedges, labels, title="Sectors", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
            plt.tight_layout()
            plt.show()
