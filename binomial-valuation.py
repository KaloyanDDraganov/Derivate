import math
import numpy as np

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


if __name__ == "__main__":
    print(valuateOption(n=30,maturity=1,underlying=100,strike=90,sigma=10,rf=0.1))
    # EURIBOR 6 Monate = 3,671%
    #print(valuateOption(n=100, maturity=0.5, underlying=18131.97, strike=19200, sigma=0.1216, rf=0.03671))
    
#def pointsToEur(points: int) -> np.complex128:
    #return points / 1000