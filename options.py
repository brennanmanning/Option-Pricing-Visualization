import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

class GeometricBrownianMotion:

    def __init__(self, drift, volatility, initial_val):
        self.mu = drift
        self.sigma = volatility
        self.initial_val = initial_val
        self.true_path = np.array([])

    def get_drift(self):
        return self.mu

    def get_volatility(self):
        return self.sigma

    def get_initial_value(self):
        return self.initial_val

    def generate_path(self, steps):
        dt = 1 / steps
        S = np.empty([steps + 1])
        S[0] = self.initial_val

        for i in range(steps):
            S[i+1] = S[i] * np.exp((self.mu - self.sigma**2 / 2) * dt +
                                   self.sigma * np.random.normal(0, np.sqrt(dt)))

        return S

    def set_path(self, path):
        self.true_path = path

    def plot_sample_paths(self, samples, steps):
        S = np.empty([steps + 1, samples])

        for j in range(samples):
            S[:, j] = self.generate_path(steps)

        t = np.linspace(0, 1, num = steps + 1)
        fig = plt.figure()
        plt.plot(t, S, color= "#000000", alpha = 0.01, figure = fig)
        return fig
        
# Need to figure out what the best way to declare value functions is, especially as we consider path dependent functions

class Option:

    def __init__(self, gbm, strike, interest, value_fn):
       self.gbm = gbm
       self.strike = strike
       self.interest = interest
       self.mu =gbm.get_drift()
       self.sigma = gbm.get_volatility()
       self.initial_value = gbm.get_initial_value()
       self.value_fn = value_fn

    def get_monte_carlo_price(self, steps, samples):
        S = np.empty([steps +1, samples])
        C = np.empty([samples])

        for j in range(samples):
            S[:, j] = self.gbm.generate_path(steps)
            C[j] = self.value_fn(S[:,j], self.strike)

        return np.exp(-self.interest) * np.mean(C)


class EuropeanPutOption(Option):

    def __init__(self, gbm, strike, interest):

        # Right now we pass in a full path for the underlying function for our value function
        # So, we could pass in gbm.true_path into PutFunction and get the value 
        def PutFunction(S, K):
            return np.maximum(K - S[-1], 0)
        Option.__init__(self, gbm, strike, interest, PutFunction)

    def get_value(self, time, current_value):
        d1 = (np.log(current_value / self.strike) + (self.interest + self.sigma**2 / 2) * (1 - time))/(self.sigma * np.sqrt(1-time))
        d2 = d1 - self.sigma * np.sqrt(1 - time)

        return self.strike * np.exp(-self.interest * (1 - time)) * norm.cdf(-d2) - current_value * norm.cdf(-d1)

    def get_price(self):
        d1 = (np.log(self.initial_value / self.strike) + self.interest + self.sigma**2 / 2) / self.sigma
        d2 = d1 - self.sigma
        return self.strike * np.exp(-self.interest) * norm.cdf(-d2) - self.initial_value * norm.cdf(-d1)

class EuropeanCallOption(Option):

    def __init__(self, gbm, strike, interest):

        def CallFunction(S, K):
            return np.maximum(S[-1] - K, 0)

        Option.__init__(self, gbm, strike, interest, CallFunction)

    def get_value(self, time, current_value):
        d1 = (np.log(current_value / self.strike) + (self.interest + self.sigma**2/2)* (1 - time))/(self.sigma* np.sqrt(1-time))
        d2 = d1 - self.sigma * np.sqrt(1-time)

        return current_value * norm.cdf(d1) - np.exp(-self.interest * (1- time)) * self.strike * norm.cdf(d2)

    def get_price(self):
        d1 = (np.log(self.initial_value / self.strike) + self.interest + self.sigma**2 / 2) / self.sigma
        d2 = d1 - self.sigma 
        return self.initial_value * norm.cdf(d1)  - np.exp(-self.interest) * self.strike * norm.cdf(d2)


# Arithmetic Asian Call Option
# This could be extended to take in a parameter called "mean" to signify whether you want to use
# an arithmetic or a geometric mean
class AsianCallOption(Option):

    def __init__(self, gbm, strike, interest):

        def AsianCallFunction(S, K):
            return np.maximum(np.mean(S) - K, 0)

        Option.__init__(self, gbm, strike, interest, AsianCallFunction)


class AsianPutOption(Option):

    def __init__(self, gbm, strike, interest):

        def AsianPutFunction(S, K):
            return np.maximum(K - np.mean(S), 0)

        Option.__init__(self, gbm, strike, interest, AsianPutFunction)
    
