import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

class GeometricBrownianMotion:

    def __init__(self, drift, volatility, initial_val):
        self.mu = drift
        self.sigma = volatility
        self.initial_val = initial_val

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

    def plot_sample_paths(self, samples, steps):
        S = np.empty([steps + 1, samples])

        for j in range(samples):
            S[:, j] = self.generate_path(steps)

        t = np.linspace(0, 1, num = steps + 1)
        fig = plt.figure()
        plt.plot(t, S, color= "#000000", alpha = 0.01, figure = fig)
        return fig
        

# Example Usage

# GBM with drift 0 and volatility 0.1 that starts at 1
gbm = GeometricBrownianMotion(0, 0.1, 1)

# Sample Paths can be generated as follows
path1 = gbm.generate_path(100)
path2 = gbm.generate_path(10)

gbm_plot = gbm.plot_sample_paths(1000, 100)

# Implement a method for plotting the pricing surface and make it so that this is a subclass of Options

class EuropeanCallOption:

    def __init__(self, gbm, strike, interest):
        self.strike = strike
        self.interest = interest
        self.sigma = gbm.get_volatility()
        self.initial_value = gbm.get_initial_value()

    def get_price(self, time, current_price):
        d1 = (np.log(current_price / self.strike) + (self.interest + self.sigma**2/2)* (1 - time))/(self.sigma* np.sqrt(1-time))
        d2 = d1 - self.sigma * np.sqrt(1-time)

        return current_price * norm.cdf(d1) - np.exp(-self.interest * (1- time)) * self.strike * norm.cdf(d2)


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

# See if making the value function dependent of gbm will helpful and how to implement it

class EuropeanPutOption(Option):

    def __init__(self, gbm, strike, interest):
        def PutFunction(S, K):
            return np.maximum(K - S, 0)
        Option.__init__(self, gbm, strike, interest, PutFunction)
