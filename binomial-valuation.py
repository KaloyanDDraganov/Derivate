import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

cache = dict()

def valuateOption(n: int, maturity: np.complex128, underlying: np.complex128, strike: np.complex128, sigma: np.complex128, rf: np.complex128) -> np.complex128:

    if (n <= 0):
        return max(0, underlying - strike)


    dT = maturity / n
    u = math.pow(math.e, math.sqrt(dT))
    d = 1 / u
    Ph = underlying * u
    Pl = underlying * d

    q = (pow(1+rf, dT) - d) / (u - d)

    if  Ph in cache:
        Ch = cache.get(Ph)
    else:
        Ch = valuateOption(n-1, maturity, Ph, strike, sigma, rf)
        cache.update({Ph:Ch})

    if  Pl in cache:
        Cl = cache.get(Pl)
    else:
        Cl = valuateOption(n-1, maturity, Pl, strike, sigma, rf)
        cache.update({Pl:Cl})

    c = (q*Ch + (1- q) * Cl) / (pow(1+rf, dT))
    
    return c

def blackScholes(S0, t, T, X, s, r):
    d1 = (np.log(S0/X)+(r + (s*s)/2)*(T-t)) / (s * math.sqrt(T-t))

    C = S0 * norm.cdf(d1) - X* np.exp(-r *(T-t)) * norm.cdf(d1-s*math.sqrt(T-t))
    return C

if __name__ == "__main__":
    #print(valuateOption(n=30,maturity=1,underlying=100,strike=90,sigma=10,rf=0.1))
    # EURIBOR 6 Monate = 3,671%
    #print(valuateOption(n=100, maturity=0.5, underlying=18131.97, strike=19200, sigma=0.1216, rf=0.03671))

    #n_values = list(range(1, 25))
    #prices = list(map(lambda n: valuateOption(n, maturity=0.5, underlying=18131.97, strike=19200, sigma=0.1216, rf=0.03671), n_values))

    #plt.plot(n_values, prices, marker='o', linestyle='-', color='b', label='Option price against n')
    #plt.xscale('log')
    #plt.title("Calculating a call price with the binomial method")
    #plt.xlabel("Log-Scaled n values")
    #plt.ylabel("Option Price in points")
    #plt.legend()

    #TODO: Plot Black-Scholes
    #TODO: Plot market value
    #plt.show()
    print(blackScholes(50, 0, 5/12, 50, 0.4, 0.1))
    
#def pointsToEur(points: int) -> np.complex128:
    #return points / 1000