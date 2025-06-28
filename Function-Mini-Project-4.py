import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
from scipy.stats import norm

def stock_path_custom_sigma(S0, t, r, mu, n_paths, n_steps):
    '''
    Generation of custom stock paths following Geometeric Brownian motion,
    but log-returns do not have constant volatility.
    
    Each step of the log-returns, there is a 
    1) 50% the volatility is .2
    2) 30% chance the volatility is .3
    3) 20% chance the volatility is .45
    
    Inputs:
    S0 (float): initial stock value
    t (float): time interval of stock path movements in years
    r (float): risk-free interest rate
    mu (float): drift of log-returns
    n_paths (int): number of stock paths
    n_steps (float): number of steps in each stock path
    
    Returns:
    
    Simuatled stock paths
    '''
    
    #Noise in volatility
    noise = np.random.normal(0,1,size = (n_paths, n_steps))
    
    #Custom sigma that is not constant
    sigma = np.random.choice([.2,.3,.45], p = [.5, .3, .2], size = (n_paths, n_steps))
    
    #Time increment between each step
    dt = t/n_steps
    
    #log-returns between each step
    increments = (mu + r - .5*sigma**2)*dt + sigma*np.sqrt(dt)*noise
    
    #Cumulative log-returns at each step
    log_returns = np.cumsum(increments, axis = 1)
    
    
    #paths
    paths = S0*np.exp(log_returns)
    
    
    #Adjoint initial value S0 at start of each simulated path
    paths = np.insert(paths, 0, S0, axis = 1)
    
    
    return paths

def MC_call_delta_custom(S0, K, sigma,sigma_probs, t, r, delta_sims = int(250)):
    """Description: 
    Monte-Carlo Simulation of Custom Stock Call Delta
    
    Parameters:
    S0 (float): spot price
    K (float): strike price
    sigma (array): array of volatilities
    sigma_probs (array): probabilities of sigmas
    t (float): time to expiration
    r (float): risk-free interest rate
    delta_sims (int): Number of simulations
    
    Return
    float: simulated delta of call option
    
    """
    bump = .01*S0

    noise = np.random.normal(0,1,delta_sims)
    
    sampled_sigma = np.random.choice(sigma,p=sigma_probs,size=delta_sims)

    log_returns = (r - .5*sampled_sigma**2)*t + sampled_sigma*np.sqrt(t)*noise

    paths_up = (S0+bump)*np.exp(log_returns)
    paths_down = (S0-bump)*np.exp(log_returns)

    call_up = np.maximum(paths_up - K, 0)*np.exp(-r*t)
    call_down = np.maximum(paths_down - K, 0)*np.exp(-r*t)

    simulated_deltas = (call_up-call_down)/(2*bump)

    return np.mean(simulated_deltas)
    
def MC_call_delta_custom_array(S, K, sigma, sigma_probs, t, r, delta_sims=250):
    """
    Monte Carlo estimation of Custom Stock call deltas for an array of spot prices
    
    Parameters:
    S (np.array): array of spot prices
    K (float): strike price
    sigma (array): array of volatilities
    sigma_probs (array): probabilities of sigmas
    t (float): time to expiration
    r (float): risk-free interest rate
    delta_sims (int): Number of simulations
    
    Return
    float: simulated delta of call option
    """
    bump = 0.01 * S
    noise = np.random.normal(0, 1, (delta_sims, len(S)))

    sampled_sigma = np.random.choice(sigma, p=sigma_probs, size=(delta_sims, len(S)))
    log_returns = (r - 0.5 * sampled_sigma**2) * t + sampled_sigma * np.sqrt(t) * noise

    paths_up = (S + bump) * np.exp(log_returns)
    paths_down = (S - bump) * np.exp(log_returns)

    call_up = np.maximum(paths_up - K, 0) * np.exp(-r * t)
    call_down = np.maximum(paths_down - K, 0) * np.exp(-r * t)

    deltas = (call_up - call_down) / (2 * bump)
    return np.mean(deltas, axis=0)

def MC_call_custom_sigma(S0, K, sigma, sigma_probs, t, r, mu = 0, n_sims = 2500, n_hedges = 50, delta_sims = 250):
    
    """Description
    Monte-Carlo simulation of the value of a call option based on custom stock with Delta based control variants
    
    
    Parameters:
    S0 (float): spot price
    K (float): strike price
    sigma (array): array of volatilities
    sigma_probs (array): probabilities of sigmas
    r (float): risk-free interest rate
    t (float): time to expiration
    mu (float): Drift of log-returns
    n_sims (int): Number of simulations
    n_hedges (int): number of delta control variants at evenly spaced increments
    
    
    Return:
    np.array of simulated values of Black-Scholes value of call option
    """
    
    #Create random noise for n_sims number of paths with n_hedges steps in simulated stock movements
    noise = np.random.normal(0,1, (n_sims,n_hedges))

    dt = t/n_hedges #time interval between each step in simulated path
    sampled_sigma = np.random.choice(sigma, p = sigma_probs,size = (n_sims,n_hedges))


    exponent = (mu + r - .5*sampled_sigma**2)*dt + sampled_sigma*np.sqrt(dt)*noise

    log_returns = np.cumsum(exponent, axis = 1)

    paths = S0*np.exp(log_returns)


    #Simulate call payoffs discounted to time 0

    path_ends = paths[:,-1] 

    call_payoffs = np.maximum(path_ends - K, 0)*np.exp(-r*t)


    #Simulate stock profits at each interval

    ## profit from start to first step discounted to time 0

    paths_first_step = paths[:,0]

    delta_start = MC_call_delta_custom(S0,K,sigma,sigma_probs,t,r,delta_sims)

    stock_profits_start = (paths_first_step - np.exp(r*dt)*S0)*delta_start*np.exp(-r*dt)

    total_stock_profits = []

    total_stock_profits.append(stock_profits_start)

    ## stock profits in intermediate steps
    for i in range(1,n_hedges):
        path_starts = paths[:,i-1]
        path_ends = paths[:,i]
    #time to expiration from starting point 
    #needed to find delta of option and how much stock should be held to be delta neutral until next step
        tte = t - i*dt 
        deltas = MC_call_delta_custom_array(path_starts, K, sigma,sigma_probs, tte, r,delta_sims)
        stock_profit = (path_ends - path_starts*np.exp(r*dt))*deltas*np.exp(-r*(i+1)*dt)
        total_stock_profits.append(stock_profit)

    stock_profits = np.sum(total_stock_profits, axis = 0)
    
    profits_hedged = call_payoffs - stock_profits
    
    return profits_hedged

# Function for BS call price
def black_scholes_call(S, K, T, r, sigma):
    from scipy.stats import norm
    if T < 1e-6:
        return max(0, S - K)
    sigma = max(sigma, 1e-6)
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def simulate_garch_path(S0, T, r, omega, alpha, beta):
    """Simulate a single GARCH(1,1) path"""
    n_steps = int(T * 252)  # Assuming 252 trading days per year
    dt = T / n_steps
    
    # Initialize arrays
    returns = np.zeros(n_steps + 1)
    variance = np.zeros(n_steps + 1)
    prices = np.zeros(n_steps + 1)
    prices[0] = S0
    
    # Initial variance (unconditional variance)
    if (alpha + beta) < 1 and omega > 0:
        variance[0] = omega / (1 - alpha - beta)
    else:
        variance[0] = 0.01  # fallback initial variance
    
    # Simulate the path
    for t in range(1, n_steps + 1):
        z = np.random.normal(0, 1)
        variance[t] = omega + alpha * returns[t-1]**2 + beta * variance[t-1]
        returns[t] = np.sqrt(variance[t]) * z
        prices[t] = prices[t-1] * np.exp((r - 0.5 * variance[t]) * dt + np.sqrt(variance[t]) * np.sqrt(dt) * z)
    
    return prices, variance

def simulate_garch_paths(S0, T, r, omega, alpha, beta, n_paths):
    """
    Vectorized simulation of multiple GARCH(1,1) paths
    Returns arrays of prices and variances for all paths
    """
    n_steps = int(T * 252)
    dt = T / n_steps
    
    # Initialize arrays for all paths
    prices = np.zeros((n_paths, n_steps + 1))
    variances = np.zeros((n_paths, n_steps + 1))
    returns = np.zeros((n_paths, n_steps + 1))
    
    # Set initial values
    prices[:, 0] = S0
    if (alpha + beta) < 1 and omega > 0:
        variances[:, 0] = omega / (1 - alpha - beta)
    else:
        variances[:, 0] = 0.01
    
    # Generate all random numbers at once
    z = np.random.normal(0, 1, (n_paths, n_steps+1))
    
    # Simulate all paths
    for t in range(1, n_steps + 1):        
        # Update variance using GARCH(1,1)
        variances[:, t] = omega + alpha * returns[:, t-1]**2 + beta * variances[:, t-1]

        # Calculate returns
        returns[:, t] = np.sqrt(variances[:, t]) * z[:, t]
        
        # Update price
        prices[:, t] = prices[:, t-1] * np.exp((r - 0.5 * variances[:, t]) * dt + np.sqrt(variances[:, t]) * np.sqrt(dt) * z[:, t-1])
    
    return prices, variances

def plot_simulation_paths(S0, T, r, omega, alpha, beta, n_paths=5):
    """Plot sample simulation paths"""
    plt.figure(figsize=(10, 6))
    for _ in range(n_paths):
        path, _ = simulate_garch_path(S0, T, r, omega, alpha, beta)
        plt.plot(np.linspace(0, T, len(path)), path)
    
    plt.title('Sample GARCH(1,1) Price Paths')
    plt.xlabel('Time (years)')
    plt.ylabel('Stock Price')
    plt.grid(True)
    plt.show()

def get_empirical_volatility_distribution(S0, T, r, omega, alpha, beta, n_paths):
    """
    Get empirical distribution of volatility from GARCH(1,1) simulations
    
    Parameters:
    -----------
    S0, T, r, omega, alpha, beta : GARCH parameters
    n_paths : int
        Number of paths to simulate
        
    Returns:
    --------
    numpy.ndarray : Array of volatilities for random sampling
    """
    # Simulate paths
    prices, variances = simulate_garch_paths(S0, T, r, omega, alpha, beta, n_paths)
    
    # Extract all volatilities (sqrt of variances)
    volatilities = np.sqrt(variances.flatten())
    
    return volatilities

def price_call_option(S0, K, T, r, omega, alpha, beta, n_simulations):
    """Price a European call option using vectorized Monte Carlo simulation"""
    prices, _ = simulate_garch_paths(S0, T, r, omega, alpha, beta, n_simulations)
    final_prices = prices[:, -1]
    payoffs = np.maximum(0, final_prices - K)
    option_price = np.exp(-r * T) * np.mean(payoffs)
    
    option_price_std = np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_simulations)
    
    return option_price, option_price_std, payoffs

def black_scholes_price(S, K, T, r, sigma):
    """
    Compute the theoretical Black-Scholes price for a European call option.
    """
    if T <= 0 or sigma <= 0:
        return max(0, S - K)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def MC_call_delta_GARCH(S0, K, t, r, omega, alpha, beta, delta_sims = int(250)):
    """Description: 
    Monte-Carlo Simulation of GARCH-based Call option Delta
    
    Parameters:
    S0 (float): spot price
    K (float): strike price
    sigma (array): array of volatilities
    sigma_probs (array): probabilities of sigmas
    t (float): time to expiration
    r (float): risk-free interest rate
    delta_sims (int): Number of simulations
    
    Return
    float: simulated delta of call option
    
    """
    bump = .01*S0

    noise = np.random.normal(0,1,delta_sims)
    
    # Get the empirical volatility distribution from the model
    volatilities = get_empirical_volatility_distribution(S0, t, r, omega, alpha, beta, n_paths=10000)
    
    # # Sample multiple volatilities
    # sample_size = 1000
    # sampled_volatilities = np.random.choice(volatilities, size=sample_size)
    
    # Sample with replacement (default)
    sampled_sigma = np.random.choice(volatilities, size=delta_sims)
    
    # sampled_sigma = np.random.choice(sigma,p=sigma_probs,size=delta_sims)

    log_returns = (r - .5*sampled_sigma**2)*t + sampled_sigma*np.sqrt(t)*noise

    paths_up = (S0+bump)*np.exp(log_returns)
    paths_down = (S0-bump)*np.exp(log_returns)

    call_up = np.maximum(paths_up - K, 0)*np.exp(-r*t)
    call_down = np.maximum(paths_down - K, 0)*np.exp(-r*t)

    simulated_deltas = (call_up-call_down)/(2*bump)

    return np.mean(simulated_deltas)
    
def MC_call_delta_GARCH_array(S, K, t, r, omega, alpha, beta, delta_sims=250):
    """
    Monte Carlo estimation of GARCH-based Call Option deltas for an array of spot prices
    
    Parameters:
    S (np.array): array of spot prices
    K (float): strike price
    sigma (array): array of volatilities
    sigma_probs (array): probabilities of sigmas
    t (float): time to expiration
    r (float): risk-free interest rate
    delta_sims (int): Number of simulations
    
    Return
    float: simulated delta of call option
    """
    bump = 0.01 * S
    noise = np.random.normal(0, 1, (delta_sims, len(S)))

    # Get the empirical volatility distribution from the model
    volatilities = get_empirical_volatility_distribution(0.5, t, r, omega, alpha, beta, n_paths=10000)
    
    # # Sample multiple volatilities
    # sample_size = 1000
    # sampled_volatilities = np.random.choice(volatilities, size=sample_size)
    
    # Sample with replacement (default)
    sampled_sigma = np.random.choice(volatilities, size=(delta_sims, len(S)))

    # sampled_sigma = np.random.choice(sigma, p=sigma_probs, size=(delta_sims, len(S)))
    log_returns = (r - 0.5 * sampled_sigma**2) * t + sampled_sigma * np.sqrt(t) * noise

    paths_up = (S + bump) * np.exp(log_returns)
    paths_down = (S - bump) * np.exp(log_returns)

    call_up = np.maximum(paths_up - K, 0) * np.exp(-r * t)
    call_down = np.maximum(paths_down - K, 0) * np.exp(-r * t)

    deltas = (call_up - call_down) / (2 * bump)
    return np.mean(deltas, axis=0)

def MC_call_GARCH(S0, K, t, r, omega, alpha, beta, mu = 0, n_sims = 2500, n_hedges = 50, delta_sims = 250):
    
    """Description
    Monte-Carlo simulation of the value of a call option on a stock following GARCH(1,1) dynamics with Delta based control variants
    
    
    Parameters:
    S0 (float): spot price
    K (float): strike price
    sigma (array): array of volatilities
    sigma_probs (array): probabilities of sigmas
    r (float): risk-free interest rate
    t (float): time to expiration
    mu (float): Drift of log-returns
    n_sims (int): Number of simulations
    n_hedges (int): number of delta control variants at evenly spaced increments
    
    
    Return:
    np.array of simulated values of Black-Scholes value of call option
    """
    
    #Create random noise for n_sims number of paths with n_hedges steps in simulated stock movements
    noise = np.random.normal(0,1, (n_sims,n_hedges))

    dt = t/n_hedges #time interval between each step in simulated path

    volatilities = get_empirical_volatility_distribution(S0, t, r, omega, alpha, beta, n_paths=10000)
    sampled_sigma = np.random.choice(volatilities, size = (n_sims,n_hedges))


    exponent = (mu + r - .5*sampled_sigma**2)*dt + sampled_sigma*np.sqrt(dt)*noise

    log_returns = np.cumsum(exponent, axis = 1)

    paths = S0*np.exp(log_returns)


    #Simulate call payoffs discounted to time 0

    path_ends = paths[:,-1] 

    call_payoffs = np.maximum(path_ends - K, 0)*np.exp(-r*t)


    #Simulate stock profits at each interval

    ## profit from start to first step discounted to time 0

    paths_first_step = paths[:,0]

    delta_start = MC_call_delta_GARCH(S0, K, t, r, omega, alpha, beta, delta_sims)

    stock_profits_start = (paths_first_step - np.exp(r*dt)*S0)*delta_start*np.exp(-r*dt)

    total_stock_profits = []

    total_stock_profits.append(stock_profits_start)

    ## stock profits in intermediate steps
    for i in range(1,n_hedges):
        path_starts = paths[:,i-1]
        path_ends = paths[:,i]
    #time to expiration from starting point 
    #needed to find delta of option and how much stock should be held to be delta neutral until next step
        tte = t - i*dt 
        deltas = MC_call_delta_GARCH_array(path_starts, K, tte, r, omega, alpha, beta, delta_sims)
        stock_profit = (path_ends - path_starts*np.exp(r*dt))*deltas*np.exp(-r*(i+1)*dt)
        total_stock_profits.append(stock_profit)

    stock_profits = np.sum(total_stock_profits, axis = 0)
    
    profits_hedged = call_payoffs - stock_profits
    
    return profits_hedged