import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def valuateOption(S0, X, T, rf, sigma, n, option_type="call"):
    
    # Calculate the time step per period
    dt = T / n
    
    # Calculate the up and down factors
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    
    # Calculate the risk-neutral probability
    q = ((1+rf * dt) - d) / (u - d)
    
    # Initialize the stock prices at maturity
    stock_prices = np.zeros(n + 1)
    option_values = np.zeros(n + 1)
    
    # Calculate the stock prices at the end of the n-th period
    for i in range(n + 1):
        stock_prices[i] = S0 * (u ** (n - i)) * (d ** i)
    
    # Calculate the option values at the end of the n-th period
    if option_type == "call":
        option_values = np.maximum(0, stock_prices - X)
    elif option_type == "put":
        option_values = np.maximum(0, X - stock_prices)
    
    # Backward recursion to calculate the option values at each previous node
    for j in range(n - 1, -1, -1):
        for i in range(j + 1):
            option_values[i] =  (q * option_values[i] + (1 - q) * option_values[i + 1])/np.exp(rf * dt)
    
    return option_values[0]

def blackScholes(S0, X, T, rf, sigma, t):
    d1 = (np.log(S0/X)+(rf + (sigma*sigma)/2)*(T-t)) / (sigma * np.sqrt(T-t))

    C = S0 * norm.cdf(d1) - X* np.exp(-rf *(T-t)) * norm.cdf(d1-sigma*np.sqrt(T-t))
    return C

S0 = 18131.97 
X = 19200
T = 0.5    
rf = 0.03671
sigma = 0.1216

call_price = valuateOption(S0, X, T, rf, sigma, 100, option_type="call")
black_scholes = blackScholes(S0=18131.97, X=19200, T=0.5, rf=0.03671, sigma=0.1216, t=0)

print(f"Call: {call_price:.4f}")
print(f"Black Scholes:  {black_scholes:.4f}")

#put_price = valuateOption(S0, X, T, rf, sigma, 100, option_type="put")
#print(f"Put:  {put_price:.4f}")

n_values = list(range(1, 1000))
prices = list(map(lambda n: valuateOption(n=n, T=0.5, S0=18131.97, X=19200, sigma=0.1216, rf=0.03671), n_values))

plt.plot(n_values, prices, marker='o', linestyle='-', color='b', label='Option price against n')
plt.axhline(y=black_scholes, linestyle='-', color='r', label='Black Scholes')
plt.xscale('log')
plt.title("Calculating a call price with the binomial method")
plt.xlabel("Log-Scaled n values")
plt.ylabel("Option Price in points")
plt.legend()
plt.show()
