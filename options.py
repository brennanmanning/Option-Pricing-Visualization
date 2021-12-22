import numpy as np

# Number of simulations
B = 10000

# Number of time steps
N = 1001

 # Parameters for the option

 S_0 = 100
 mu = 1.05
 sigma = 0.2
 r = 0.01

delta_t = 0.001

S = np.zeros((B, N))
S[0,:] = S_0

Z = np.rand.normal((B, N-1))
 
 for b in range(B):
     for t in range(1,N): 
         S[t,b] = S[t-1,b] * np.exp((r - sigma**2/2)*delta_t + Z[t,b] * sigma * np.sqrt(delta_t))


# Then plot each path with an alpha of around 0.01 or 0.001
