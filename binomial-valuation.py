import math

def valuateOption(n: int, maturity: float, underlying: float, strike: float, sigma: float, rf: float) -> float:

    if (n <= 0):
        return max(0, underlying - strike)


    dT = maturity / n
    u = math.pow(math.e, math.sqrt(dT))
    d = 1 / u
    Ph = underlying * u
    Pl = underlying * d

    q = (pow(1+rf, dT) - d) / (u - d)
    Ch = valuateOption(n-1, maturity, Ph, strike, sigma, rf)
    Cl = valuateOption(n-1, maturity, Pl, strike, sigma, rf)


    c = (q*Ch + (1-q)*Cl) / (pow(1+rf, dT))
    
    return c


if __name__ == "__main__":
   print(valuateOption(n=2,maturity=1,underlying=100,strike=90,sigma=10,rf=0.1))
    
def pointsToEur(points: int) -> float:
    return 