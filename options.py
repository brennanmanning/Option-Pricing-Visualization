import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Stock Price Processes
class GeometricBrownianMotion:

    def __init__(self, drift, volatility, initial_val):
        self.mu = drift
        self.sigma = volatility
        self.initial_val = initial_val
        self.true_path = np.array([])

    def generate_path(self, steps):
        dt = 1 / steps
        S = np.empty([steps + 1])
        S[0] = self.initial_val

        for i in range(steps):
            S[i+1] = S[i] * np.exp((self.mu - self.sigma**2 / 2) * dt +
                                   self.sigma * np.random.normal(0, np.sqrt(dt)))

        return S

    def set_drift(self, new_drift):
        self.mu = new_drift

    def plot_sample_paths(self, samples, steps, alpha = 0.1, ax = None):
        S = np.empty([steps + 1, samples])

        for j in range(samples):
            S[:, j] = self.generate_path(steps)

        t = np.linspace(0, 1, num = steps + 1)
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(t, S, color= "#000000", alpha = alpha)
        ax.set_xlabel("Time")
        ax.set_ylabel("Stock Price")

        return ax

class BinomialTree:

    def __init__(self, volatility, initial_val, steps):
        self.sigma = volatility
        self.initial_val = initial_val
        self.steps = steps
        self.delta_t = 1 / self.steps
        self.u = np.exp(self.sigma * np.sqrt(self.delta_t))
        self.d = np.exp(- self.sigma * np.sqrt(self.delta_t))

    def get_binomial_tree(self):
        S = np.zeros([self.steps + 1, self.steps + 1])
        S[0, 0] = self.initial_val
        for i in range(1, self.steps + 1):
            for j in range(i + 1):
                S[j, i] = S[0, 0] * self.u**(i - j) * self.d**j

        return S 

    def get_binomial_path(self, q):
        # q is the probability of going up at a step and is not necessarily the risk-neutral probability 
        rand_draw = np.random.uniform(0, 1, self.steps)
        up_seq = rand_draw < q
        S = np.empty([self.steps + 1])
        S[0] = self.initial_val
        for i in range(1, self.steps + 1):
            S[i] = S[i-1] * self.u**up_seq[i -1] * self.d**(1 - up_seq[i - 1])

        return S
 
    def plot_sample_paths(self, q, samples, alpha = 0.1, ax = None):
        S = np.empty([self.steps + 1, samples]) 

        for i in range(samples):
            S[:, i] = self.get_binomial_path(q)

        t = np.linspace(0, 1, self.steps + 1)
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(t, S, color= "#000000", alpha = alpha)
        ax.set(xlabel = "Time", ylabel = "Stock Price")

        return ax

class TrinomialTree:

    def __init__(self, volatility, initial_val, steps, interest):
        self.sigma = volatility
        self.initial_val = initial_val
        self.steps = steps
        self.delta_t = 1 / steps
        self.interest = interest
        self.u = np.exp(self.sigma * np.sqrt(2 * self.delta_t))
        self.d = np.exp(-self.sigma * np.sqrt(2 * self.delta_t))
        self.p_u = ((np.exp(self.interest * self.delta_t / 2) - self.d) / (self.u - self.d))**2
        self.p_d = ((self.u - np.exp(self.interest * self.delta_t / 2)) / (self.u - self.d))**2
        self.p_m = 1 - self.p_u - self.p_d

    def get_trinomial_tree(self):
        S = np.zeros([2 * self.steps + 1, self.steps + 1])
        S[0, 0] = self.initial_val
        for j in range(1, self.steps + 1):
            for i in range(2 * j + 1):
                S[i,j] = self.initial_val * self.u**j * self.d**i

        return S

    def get_trinomial_path(self):
        rand_draw = np.random.uniform(0, 1, self.steps)
        def interval(x):
            if x < self.p_u:
                return 0
            elif x < self.p_u + self.p_d:
                return 1
            else:
                return 2
        vec_interval = np.vectorize(interval)
        seq = vec_interval(rand_draw)
        up_seq = np.zeros(self.steps)
        down_seq = np.zeros(self.steps)
        for i in range(self.steps):
            if seq[i] == 0:
                up_seq[i] = 1
            elif seq[i] == 1:
                down_seq[i] = 1

        S = np.empty(self.steps + 1)
        S[0] = self.initial_val
        for i in range(1, self.steps + 1):
            S[i] = S[i-1] * self.u ** up_seq[i - 1] * self.d ** down_seq[i - 1] 

        return S

    def plot_sample_paths(self, samples, alpha = 0.1, ax = None):
        S = np.empty([self.steps + 1, samples])

        for i in range(samples):
            S[:, i] = self.get_trinomial_path()

        t = np.linspace(0, 1, self.steps + 1)
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(t, S, alpha = alpha, color = "#000000")
        ax.set(xlabel = "Time", ylabel = "Stock Price")
        
        return ax
        
class Option:

    def __init__(self, gbm, strike, interest, call, value_fn):
       self.gbm = gbm
       self.strike = strike
       self.interest = interest
       self.mu = gbm.mu
       self.sigma = gbm.sigma
       self.initial_value = gbm.initial_val
       self.call = call
       self.value_fn = value_fn

    def get_monte_carlo_price(self, steps, samples):
        # Need to switch to risk-neutral measure for generating sample paths
        S = np.empty([steps +1, samples])
        C = np.empty([samples])

        self.gbm.set_drift(self.interest)

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

        self.gbm.set_drift(self.interest)

        for j in range(samples):
            S[:, j] = self.gbm.generate_path(steps)
            C[j] = self.value_fn(S[:,j], self.strike, self.call)

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

    def plot_option_surface(self, X, Y, Z, zlabel, ax):
        if ax is None:
            fig, ax = plt.subplots(projection = "3d")
        surf = ax.plot_surface(X, Y, Z, cmap = "viridis")
        ax.set_xlabel("Time")
        ax.set_ylabel("Stock Price")
        ax.set_zlabel(zlabel)

        return ax

    def get_pricing_surface(self, low_price, high_price, ax = None):
        t = np.linspace(0, 0.999, 999)
        x = np.linspace(low_price, high_price, 1000)
        T, X = np.meshgrid(t, x)
        c = self.get_value(T.ravel(), X.ravel()).reshape(X.shape)
        c1 = np.maximum(x - self.strike, 0) * self.call + np.maximum(self.strike - x, 0) * (1 - self.call)
        C = np.c_[c, c1]
        T = np.c_[T, np.ones(1000)]
        X = np.c_[X, x]

        return self.plot_option_surface(T, X, C, "Option Price", ax)

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
        
    def get_Delta_surface(self, low_price, high_price, ax = None):
        t = np.linspace(0, 0.999, 1000)
        x = np.linspace(low_price, high_price, 1000)
        T, X = np.meshgrid(t, x)
        Z = self.get_Delta(T.ravel(), X.ravel()).reshape(X.shape)

        return self.plot_option_surface(T, X, Z, r"$\Delta$", ax)

    def get_Gamma_surface(self, low_price, high_price, ax = None):
        t = np.linspace(0, 0.999, 1000)
        x = np.linspace(low_price, high_price, 1000)
        T, X = np.meshgrid(t, x)
        Z = self.get_Gamma(T.ravel(), X.ravel()).reshape(X.shape)

        return self.plot_option_surface(T, X, Z, r"$\Gamma$", ax)

    def get_Theta_surface(self, low_price, high_price, ax = None):
        t = np.linspace(0, 0.999, 1000)
        x = np.linspace(low_price, high_price, 1000)
        T, X = np.meshgrid(t, x)
        Z = self.get_Theta(T.ravel(), X.ravel()).reshape(X.shape)

        return self.plot_option_surface(T, X, Z, r"$\Theta$", ax)

    def get_Vega_surface(self, low_price, high_price, ax = None):
        t = np.linspace(0, 0.999, 1000)
        x = np.linspace(low_price, high_price, 1000)
        T, X = np.meshgrid(t, x)
        Z = self.get_Vega(T.ravel(), X.ravel()).reshape(X.shape)

        return self.plot_option_surface(T, X, Z, r"$\mathcal{V}$", ax)

    def get_rho_surface(self, low_price, high_price, ax = None):
        t = np.linspace(0, 0.999, 1000)
        x = np.linspace(low_price, high_price, 1000)
        T, X = np.meshgrid(t, x)
        Z = self.get_Rho(T.ravel(), X.ravel()).reshape(X.shape)

        return self.plot_option_surface(T, X, Z, r"$\rho$", ax)
        

# Arithmetic Asian Call Option
# This could be extended to take in a parameter called "mean" to signify whether you want to use
# an arithmetic or a geometric mean
class AsianOption(Option):

    def __init__(self, gbm, strike, interest, call, fixed):

        def AsianValue(S, K, c):
            if fixed:
                return np.maximum(np.mean(S) - K, 0) * c + np.maximum(K - np.mean(S),0) * (1-c)
            return np.maximum(S[-1] - np.mean(S), 0) * c + np.maximum(np.mean(S) - S[-1], 0) * (1 - c)
        Option.__init__(self, gbm, strike, interest, call, AsianValue)

    

