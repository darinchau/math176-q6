from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod as virtual
import matplotlib.pyplot as plt

from scipy.special import ndtr as Phi
from math import log as ln
from math import sqrt
from math import exp

from typing import Generator, Callable

##############################################################################################################
############################################### Option #######################################################
##############################################################################################################

pi = 3.141592653589793

def phi(x):
    return exp(-x*x/2)/sqrt(2 * pi)

class Options(ABC):
    def __init__(self, exercisePrice: int):
        self.E = exercisePrice
        self._ub: int = exercisePrice + 15

    @virtual
    def __repr__(self):
        raise NotImplementedError

    @property
    @virtual
    def profit(self):
        raise NotImplementedError

    def plot(self):
        x = []
        y = []
        for i in range(self._ub * 5):
            x.append(i/5)
            y.append(self.profit(i/5))

        plt.figure()
        plt.plot(x, y)
        plt.show()

    @virtual
    def cost(self, current_stock_price: float, risk_free_interest: float, sd_continuous_annual_return: float, time_to_maturity: float):
        raise NotImplementedError

    def __add__(self, other: Options) -> Options:
        return MixedPortfolio(self, other)

    def __neg__(self) -> Options:
        return ShortOption(self)

    def __sub__(self, other: Options) -> Options:
        return self + (-other)

    def __rmul__(self, lhs: int) -> Options:
        ub = self._ub
        if lhs == 0:
            return EmptyOption(ub)
        if lhs < 0:
            s = -self
            for i in range(1, -lhs):
                s = s - self
            return s
        else:
            s = self
            for i in range(1, lhs):
                s = s + self
        return s

    @virtual
    def delta(self, current_stock_price: float, risk_free_interest: float, sd_continuous_annual_return: float, time_to_maturity: float):
        raise NotImplementedError

    @virtual
    def vega(self, current_stock_price: float, risk_free_interest: float, sd_continuous_annual_return: float, time_to_maturity: float):
        raise NotImplementedError

    @virtual
    def rho(self, current_stock_price: float, risk_free_interest: float, sd_continuous_annual_return: float, time_to_maturity: float):
        raise NotImplementedError

    @virtual
    def theta(self, current_stock_price: float, risk_free_interest: float, sd_continuous_annual_return: float, time_to_maturity: float):
        raise NotImplementedError

    @virtual
    def gamma(self, current_stock_price: float, risk_free_interest: float, sd_continuous_annual_return: float, time_to_maturity: float):
        raise NotImplementedError

class MixedPortfolio(Options):
    def __init__(self, o1: Options, o2: Options) -> None:
        self.a, self.b = o1, o2
        self._ub = max(o1._ub, o2._ub)

    @property
    def profit(self):
        return lambda x: self.a.profit(x) + self.b.profit(x)

    def __repr__(self):
        s = "Mixed portfolio"
        return s

    def cost(self, current_stock_price: float, risk_free_interest: float, sd_continuous_annual_return: float, time_to_maturity: float):
        st = current_stock_price
        r = risk_free_interest
        sd = sd_continuous_annual_return
        t = time_to_maturity
        return self.a.cost(st, r, sd, t) + self.b.cost(st, r, sd, t)

    def delta(self, current_stock_price: float, risk_free_interest: float, sd_continuous_annual_return: float, time_to_maturity: float):
        st = current_stock_price
        r = risk_free_interest
        sd = sd_continuous_annual_return
        t = time_to_maturity
        return self.a.delta(st, r, sd, t) + self.b.delta(st, r, sd, t)

    def vega(self, current_stock_price: float, risk_free_interest: float, sd_continuous_annual_return: float, time_to_maturity: float):
        st = current_stock_price
        r = risk_free_interest
        sd = sd_continuous_annual_return
        t = time_to_maturity
        return self.a.vega(st, r, sd, t) + self.b.vega(st, r, sd, t)

    def theta(self, current_stock_price: float, risk_free_interest: float, sd_continuous_annual_return: float, time_to_maturity: float):
        st = current_stock_price
        r = risk_free_interest
        sd = sd_continuous_annual_return
        t = time_to_maturity
        return self.a.theta(st, r, sd, t) + self.b.theta(st, r, sd, t)

    def rho(self, current_stock_price: float, risk_free_interest: float, sd_continuous_annual_return: float, time_to_maturity: float):
        st = current_stock_price
        r = risk_free_interest
        sd = sd_continuous_annual_return
        t = time_to_maturity
        return self.a.rho(st, r, sd, t) + self.b.rho(st, r, sd, t)

    def gamma(self, current_stock_price: float, risk_free_interest: float, sd_continuous_annual_return: float, time_to_maturity: float):
        st = current_stock_price
        r = risk_free_interest
        sd = sd_continuous_annual_return
        t = time_to_maturity
        return self.a.gamma(st, r, sd, t) + self.b.gamma(st, r, sd, t)

class ShortOption(Options):
    def __init__(self, o: Options):
        self.o = o
        self._ub = o._ub

    @property
    def profit(self):
        return -self.o.profit

    def __repr__(self):
        return f"- {self.o}"

    def cost(self, current_stock_price: float, risk_free_interest: float, sd_continuous_annual_return: float, time_to_maturity: float):
        return -self.o.cost(current_stock_price, risk_free_interest, sd_continuous_annual_return, time_to_maturity)

    def delta(self, current_stock_price: float, risk_free_interest: float, sd_continuous_annual_return: float, time_to_maturity: float):
        return -self.o.delta(current_stock_price, risk_free_interest, sd_continuous_annual_return, time_to_maturity)

    def vega(self, current_stock_price: float, risk_free_interest: float, sd_continuous_annual_return: float, time_to_maturity: float):
        return -self.o.vega(current_stock_price, risk_free_interest, sd_continuous_annual_return, time_to_maturity)

    def rho(self, current_stock_price: float, risk_free_interest: float, sd_continuous_annual_return: float, time_to_maturity: float):
        return -self.o.rho(current_stock_price, risk_free_interest, sd_continuous_annual_return, time_to_maturity)

    def theta(self, current_stock_price: float, risk_free_interest: float, sd_continuous_annual_return: float, time_to_maturity: float):
        return -self.o.theta(current_stock_price, risk_free_interest, sd_continuous_annual_return, time_to_maturity)

    def gamma(self, current_stock_price: float, risk_free_interest: float, sd_continuous_annual_return: float, time_to_maturity: float):
        return -self.o.gamma(current_stock_price, risk_free_interest, sd_continuous_annual_return, time_to_maturity)

class EmptyOption(Options):
    def __init__(self, ub):
        self._ub = ub

    @property
    def profit(self):
        return lambda _: 0

    def __repr__(self):
        return ""

    def cost(self, current_stock_price: float, risk_free_interest: float, sd_continuous_annual_return: float, time_to_maturity: float):
        return 0

    def delta(self, current_stock_price: float, risk_free_interest: float, sd_continuous_annual_return: float, time_to_maturity: float):
        return 0

    def vega(self, current_stock_price: float, risk_free_interest: float, sd_continuous_annual_return: float, time_to_maturity: float):
        return 0

    def rho(self, current_stock_price: float, risk_free_interest: float, sd_continuous_annual_return: float, time_to_maturity: float):
        return 0

    def theta(self, current_stock_price: float, risk_free_interest: float, sd_continuous_annual_return: float, time_to_maturity: float):
        return 0

    def gamma(self, current_stock_price: float, risk_free_interest: float, sd_continuous_annual_return: float, time_to_maturity: float):
        return 0

class Call(Options):
    @property
    def profit(self):
        return lambda x: max(0, x - self.E)

    def __repr__(self):
        return f"C(E = {self.E})"

    def cost(self, current_stock_price: float, risk_free_interest: float, sd_continuous_annual_return: float, time_to_maturity: float):
        """The black scholes cost of a Call option, calculated using the formula in p31 lecture 12
        Enter risk free interest rate and standard deviation using percecntage"""
        S = current_stock_price
        r = risk_free_interest/100
        sigma = sd_continuous_annual_return/100
        T = time_to_maturity
        K = self.E

        d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        cost = S * Phi(d1) - K * np.exp(-r*T)* Phi(d2)
        return cost

    def delta(self, current_stock_price: float, risk_free_interest: float, sd_continuous_annual_return: float, time_to_maturity: float):
        S = current_stock_price
        r = risk_free_interest/100
        sigma = sd_continuous_annual_return/100
        T = time_to_maturity
        K = self.E

        d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
        delta = Phi(d1)
        return delta

    def vega(self, current_stock_price: float, risk_free_interest: float, sd_continuous_annual_return: float, time_to_maturity: float):
        S = current_stock_price
        r = risk_free_interest/100
        sigma = sd_continuous_annual_return/100
        T = time_to_maturity
        K = self.E

        d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))

        vega = S * phi(d1) * sqrt(T)
        return vega

    def rho(self, current_stock_price: float, risk_free_interest: float, sd_continuous_annual_return: float, time_to_maturity: float):
        S = current_stock_price
        r = risk_free_interest/100
        sigma = sd_continuous_annual_return/100
        T = time_to_maturity
        K = self.E

        d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        rho = K * T * exp(-r * T) * Phi(d2)
        return rho

    def theta(self, current_stock_price: float, risk_free_interest: float, sd_continuous_annual_return: float, time_to_maturity: float):
        S = current_stock_price
        r = risk_free_interest/100
        sigma = sd_continuous_annual_return/100
        T = time_to_maturity
        K = self.E

        d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        theta = - (S * phi(d1) * sigma) / (2 * sqrt(T)) - r * K * exp(-r * T) * Phi(d2)
        return theta

    def gamma(self, current_stock_price: float, risk_free_interest: float, sd_continuous_annual_return: float, time_to_maturity: float):
        """The black scholes cost of a Call option, calculated using the formula in p31 lecture 12
        Enter risk free interest rate and standard deviation using percecntage"""
        S = current_stock_price
        r = risk_free_interest/100
        sigma = sd_continuous_annual_return/100
        T = time_to_maturity
        K = self.E

        d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))

        gamma = phi(d1)/(S * sigma * sqrt(T))
        return gamma

class Put(Options):
    @property
    def profit(self):
        return lambda x: max(0, self.E - x)

    def __repr__(self):
        return f"P(E = {self.E})"

    def cost(self, current_stock_price: float, risk_free_interest: float, sd_continuous_annual_return: float, time_to_maturity: float):
        """The black scholes cost of a Put option, calculated using the formula in p31 lecture 12
        Enter risk free interest rate and standard deviation using percecntage"""
        S = current_stock_price
        r = risk_free_interest/100
        sigma = sd_continuous_annual_return/100
        T = time_to_maturity
        K = self.E

        d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        cost = K * exp(-r*T)* Phi(-d2) - S * Phi(-d1)
        return cost

    def delta(self, current_stock_price: float, risk_free_interest: float, sd_continuous_annual_return: float, time_to_maturity: float):
        """The black scholes cost of a Put option, calculated using the formula in p31 lecture 12
        Enter risk free interest rate and standard deviation using percecntage"""
        S = current_stock_price
        r = risk_free_interest/100
        sigma = sd_continuous_annual_return/100
        T = time_to_maturity
        K = self.E

        d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))

        delta = Phi(d1) - 1
        return delta

    def vega(self, current_stock_price: float, risk_free_interest: float, sd_continuous_annual_return: float, time_to_maturity: float):
        """The black scholes cost of a Put option, calculated using the formula in p31 lecture 12
        Enter risk free interest rate and standard deviation using percecntage"""
        S = current_stock_price
        r = risk_free_interest/100
        sigma = sd_continuous_annual_return/100
        T = time_to_maturity
        K = self.E

        d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))

        vega = S * phi(d1) * sqrt(T)
        return vega

    def rho(self, current_stock_price: float, risk_free_interest: float, sd_continuous_annual_return: float, time_to_maturity: float):
        """The black scholes cost of a Put option, calculated using the formula in p31 lecture 12
        Enter risk free interest rate and standard deviation using percecntage"""
        S = current_stock_price
        r = risk_free_interest/100
        sigma = sd_continuous_annual_return/100
        T = time_to_maturity
        K = self.E

        d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        rho = -K * T * exp(-r * T) * Phi(-d2)
        return rho

    def theta(self, current_stock_price: float, risk_free_interest: float, sd_continuous_annual_return: float, time_to_maturity: float):
        """The black scholes cost of a Put option, calculated using the formula in p31 lecture 12
        Enter risk free interest rate and standard deviation using percecntage"""
        S = current_stock_price
        r = risk_free_interest/100
        sigma = sd_continuous_annual_return/100
        T = time_to_maturity
        K = self.E

        d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        theta = - (S * phi(d1) * sigma) / (2 * sqrt(T)) + r * K * exp(-r * T) * Phi(-d2)
        return theta

    def gamma(self, current_stock_price: float, risk_free_interest: float, sd_continuous_annual_return: float, time_to_maturity: float):
        """The black scholes cost of a Put option, calculated using the formula in p31 lecture 12
        Enter risk free interest rate and standard deviation using percecntage"""
        S = current_stock_price
        r = risk_free_interest/100
        sigma = sd_continuous_annual_return/100
        T = time_to_maturity
        K = self.E

        d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))

        gamma = phi(d1)/(S * sigma * sqrt(T))
        return gamma

def fst1(st, mode, mu, sd, t):
    mu1 = ln(st) + t * (mu - 1/2 * sd * sd)
    s1 = sqrt(sd * sd * t)
    r = 1/mode / s1 / sqrt(2 * pi) * exp(-1/2 * ((ln(mode) - mu1)/s1) ** 2)
    return r

def fst2(st, mu, sd, t):
    return 1/(np.sqrt(2*pi) * st * sd * np.sqrt(t)) * np.exp((sd * sd - mu) * t)

def mode_of_stock_price(stock_price, mean_return, sd_return, time_to_maturity):
    S = stock_price
    mu = mean_return/100
    sd = sd_return/100
    t = time_to_maturity

    mode = S * exp((mu - 3/2*sd*sd) * t)
    print(f"Mode: {mode}")

    rr = fst1(S, mode, mu, sd, t)
    print(f"f(S(t)) = {rr}")

    rr2 = fst2(S, mu, sd, t)
    print(f"f(S(t)) = {rr2}")

    return mode

## Use binary search to reverse engineer answer
class FindAnswer:
    """Given a function f and a target value y, solve for x in y = f(x) for start < x < end"""
    __slots__ = ("start", "end", "target", "f")

    def __init__(self, start: float, end: float, targetValue: float, f: Callable[[float], float]):
        if start > end:
            start, end = end, start
        self.start = start
        self.end = end
        self.target = targetValue
        self.f = f

    @property
    def diff(self) -> float:
        return abs(self.end - self.start)

    @property
    def mid(self) -> float:
        return (self.start + self.end) / 2

    def isBetween(self):
        """Returns true if target is between value 1 and value 2"""
        return self.f(self.start) <= self.target <= self.f(self.end) or self.f(self.end) <= self.target <= self.f(self.start)


    def __call__(self, accuracy: float = 0.0000001, partitions: int = 10) -> Generator[float, None, None]:
        ## Use the marching line approach to narrow it down first
        ## Plus recursion to narrow it down
        ## Only works if the function is not too crazy

        # Base case
        if self.diff < accuracy and self.isBetween():
            yield self.mid
            return

        ## Recursive
        increment = self.diff / partitions

        # Calculates the value
        values = [self.start + increment * i for i in range(partitions + 1)]

        # Recursively yield the values
        for i in range(partitions):
            ## By the intermediate value theorem
            subrange = FindAnswer(values[i], values[i+1], targetValue = self.target, f = self.f)
            if subrange.isBetween():
                for v in subrange(accuracy, partitions):
                    yield v

        return

    @property
    def answer(self) -> float:
        return next(self())
