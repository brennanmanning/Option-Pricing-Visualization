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

    def set_drift(self, new_drift):
        self.mu = new_drift

    def plot_sample_paths(self, samples, steps):
        S = np.empty([steps + 1, samples])

        for j in range(samples):
            S[:, j] = self.generate_path(steps)

        t = np.linspace(0, 1, num = steps + 1)
        fig = plt.figure()
        plt.plot(t, S, color= "#000000", alpha = 0.01, figure = fig)
        return fig
        
class Option:

    def __init__(self, gbm, strike, interest, call, value_fn):
       self.gbm = gbm
       self.strike = strike
       self.interest = interest
       self.mu = gbm.get_drift()
       self.sigma = gbm.get_volatility()
       self.initial_value = gbm.get_initial_value()
       self.call = call
       self.value_fn = value_fn

    def get_monte_carlo_price(self, steps, samples):
        S = np.empty([steps +1, samples])
        C = np.empty([samples])

        for j in range(samples):
            S[:, j] = self.gbm.generate_path(steps)
            C[j] = self.value_fn(S[:,j], self.strike, self.call)

        return np.exp(-self.interest) * np.mean(C)

    def get_monte_carlo_simulation(self, steps, samples):
        # Similar to get_monte_carlo_price but returns every sample path's resulting price
        # Potential to merge the two and add  a verbose parameter to see if user wants simply the price
        # or entire sample for debugging purposes

        S = np.empty([steps + 1, samples])
        C = np.empty([samples])

        for j in range(samples):
            S[:, j] = self.gbm.generate_path(steps)
            C[j] = self.value_fn(S[:,j], self.strike, call)

        return C

class EuropeanOption(Option):

    def __init__(self, gbm, strike, interest, call):
        def EuropeanValue(S, K, c):
            return np.maximum(S[-1] - K, 0) * c + np.maximum(K - S[-1], 0) * (1 - c)

        Option.__init__(self, gbm, strike, interest, call, EuropeanValue)
    
    def get_d1(self, time, current_value):
        return (np.log(current_value / self.strike) + (self.interest + self.sigma**2 / 2) * (1 - time))/(self.sigma * np.sqrt(1-time))

    def get_value(self, time, current_value):
        d1 = self.get_d1(time, current_value)
        d2 = d1 - self.sigma * np.sqrt(1 - time)
        if self.call:
            return current_value * norm.cdf(d1) - np.exp(-self.interest * (1 - time)) * self.strike * norm.cdf(d2)
        return self.strike * np.exp(-self.interest * (1 - time)) * norm.cdf(-d2) - current_value * norm.cdf(-d1)

    def get_price(self):
        d1 = self.get_d1(0, self.initial_value)
        d2 = d1 - self.sigma
        if self.call:
            return self.initial_value * norm.cdf(d1) - np.exp(-self.interest) * self.strike * norm.cdf(d2)

        return self.strike * np.exp(-self.interest) * norm.cdf(-d2) - self.initial_value * norm.cdf(-d1)

    def plot_option_surface(self, X, Y, Z, zlabel):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = "3d")
        surf = ax.plot_surface(X, Y, Z, cmap = "viridis")
        ax.set_xlabel("Time")
        ax.set_ylabel("Stock Price")
        ax.set_zlabel(zlabel)
        fig.colorbar(surf, shrink = 0.6)

        return fig
    def get_pricing_surface(self, low_price, high_price):
        t = np.linspace(0, 0.999, 999)
        x = np.linspace(low_price, high_price, 1000)
        T, X = np.meshgrid(t, x)
        c = self.get_value(T.ravel(), X.ravel()).reshape(X.shape)
        c1 = np.maximum(x - self.strike, 0) * self.call + np.maximum(self.strike - x, 0) * (1 - self.call)
        C = np.c_[c, c1]
        T = np.c_[T, np.ones(1000)]
        X = np.c_[X, x]

        return self.plot_option_surface(T, X, C, "Option Price")

    def get_Delta(self, time, current_value):
        if self.call:
            return norm.cdf(self.get_d1(time, current_value))
        return norm.cdf(self.get_d1(time, current_value)) - 1

    def get_Gamma(self, time, current_value):
        return norm.pdf(self.get_d1(time, current_value)) / (current_value * self.sigma * np.sqrt(1 - time))

    def get_Theta(self, time, current_value):
        d1 = self.get_d1(time, current_value)
        d2 = d1 - self.sigma * np.sqrt(1 - time) 
        if self.call:
            return - (current_value * norm.pdf(d1) * self.sigma) / (2 * np.sqrt(1 - time)) - self.interest * self.strike * np.exp(- self.interest * (1 -time)) * norm.cdf(d2)
        return - (current_value * norm.pdf(d1) * self.sigma) / (2 * np.sqrt(1 - time)) + self.interest * self.strike * np.exp(- self.interest * (1 - time)) * norm.cdf(-d2)

    def get_Vega(self, time, current_value):
        d2 = self.get_d1(time, current_value) - self.sigma * np.sqrt(1 - time)
        return np.sqrt(1 - time) * self.strike * np.exp(-self.interest * (1 - time)) * norm.pdf(d2)

    def get_Rho(self, time, current_value):
        d2 = self.get_d1(time, current_value) - self.sigma * np.sqrt(1 - time)
        if self.call:
            return self.strike * (1 - time) * np.exp(self.interest * (1 -time)) * norm.cdf(d2)
        return - self.strike * (1 -time) * np.exp(self.interest * (1 - time)) * norm.cdf(-d2)
        
    def get_Delta_surface(self, low_price, high_price):
        t = np.linspace(0, 0.999, 1000)
        x = np.linspace(low_price, high_price, 1000)
        T, X = np.meshgrid(t, x)
        Z = self.get_Delta(T.ravel(), X.ravel()).reshape(X.shape)

        return self.plot_option_surface(T, X, Z, r"$\Delta$")

    def get_Gamma_surface(self, low_price, high_price):
        t = np.linspace(0, 0.999, 1000)
        x = np.linspace(low_price, high_price, 1000)
        T, X = np.meshgrid(t, x)
        Z = self.get_Gamma(T.ravel(), X.ravel()).reshape(X.shape)

        return self.plot_option_surface(T, X, Z, r"$\Gamma$")

    def get_Theta_surface(self, low_price, high_price):
        t = np.linspace(0, 0.999, 1000)
        x = np.linspace(low_price, high_price, 1000)
        T, X = np.meshgrid(t, x)
        Z = self.get_Theta(T.ravel(), X.ravel()).reshape(X.shape)

        return self.plot_option_surface(T, X, Z, r"$\Theta$")

    def get_Vega_surface(self, low_price, high_price):
        t = np.linspace(0, 0.999, 1000)
        x = np.linspace(low_price, high_price, 1000)
        T, X = np.meshgrid(t, x)
        Z = self.get_Vega(T.ravel(), X.ravel()).reshape(X.shape)

        return self.plot_option_surface(T, X, Z, r"$\mathcal{V}$")

    def get_rho_surface(self, low_price, high_price):
        t = np.linspace(0, 0.999, 1000)
        x = np.linspace(low_price, high_price, 1000)
        T, X = np.meshgrid(t, x)
        Z = self.get_Vega(T.ravel(), X.ravel()).reshape(X.shape)

        return self.plot_option_surface(T, X, Z, r"$\rho$")
        

# Arithmetic Asian Call Option
# This could be extended to take in a parameter called "mean" to signify whether you want to use
# an arithmetic or a geometric mean
class AsianOption(Option):

    def __init__(self, gbm, strike, interest, call):

        def AsianValue(S, K, c):
            return np.maximum(np.mean(S) - K, 0) * c + np.maximum(K - np.mean(S),0) * (1-c)
        Option.__init__(self, gbm, strike, interest, call, AsianValue)

    
