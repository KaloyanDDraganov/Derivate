import numpy as np

def valuateOption(S0, K, T, r, sigma, n, option_type="call"):
    
    # Calculate the time step per period
    dt = T / n
    
    # Calculate the up and down factors
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    
    # Calculate the risk-neutral probability
    q = ((1+r * dt) - d) / (u - d)
    
    # Initialize the stock prices at maturity
    stock_prices = np.zeros(n + 1)
    option_values = np.zeros(n + 1)
    
    # Calculate the stock prices at the end of the n-th period
    for i in range(n + 1):
        stock_prices[i] = S0 * (u ** (n - i)) * (d ** i)
    
    # Calculate the option values at the end of the n-th period
    if option_type == "call":
        option_values = np.maximum(0, stock_prices - K)
    elif option_type == "put":
        option_values = np.maximum(0, K - stock_prices)
    
    # Backward recursion to calculate the option values at each previous node
    for j in range(n - 1, -1, -1):
        for i in range(j + 1):
            option_values[i] =  (q * option_values[i] + (1 - q) * option_values[i + 1])/(1+r * dt)
    
    return option_values[0]

S0 = 100 
K = 90  
T = 2    
r = 0.1
sigma = 0.1216
n = 1000 

call_price = valuateOption(S0, K, T, r, sigma, n, option_type="call")
put_price = valuateOption(S0, K, T, r, sigma, n, option_type="put")

print(f"Call: {call_price:.4f}")
print(f"Put:  {put_price:.4f}")
