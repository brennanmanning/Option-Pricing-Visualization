import numpy as np
import matplotlib.pyplot as plt

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
        fig, ax = plt.subplots()
        ax.plot(t, S, color= "#ffffff", alpha = 0.01)
        return ax
        

# Example Usage

# GBM with drift 0 and volatility 0.1 that starts at 1
gbm = GeometricBrownianMotion(0, 0.1, 1)

# Sample Paths can be generated as follows
path1 = gbm.generate_path(100)
path2 = gbm.generate_path(10)

gbm_plot = gbm.plot_sample_paths(1000, 100)
