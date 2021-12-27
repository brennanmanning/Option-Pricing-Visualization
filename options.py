import numpy as np


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

    def generate_path(self, nr_steps):
        dt = 1 / nr_steps
        S = np.empty([nr_steps + 1, 1])
        S[0] = self.initial_val

        for i in range(nr_steps):
            S[i+1] = S[i] * np.exp((self.mu - self.sigma**2 / 2) * dt +
                                   self.sigma * np.random.normal(0, np.sqrt(dt)))

        return S
