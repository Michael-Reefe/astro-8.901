import numpy as np
import matplotlib.pyplot as plt

# constants
me = 9.10939e-28
mp = 1.67262e-24
c = 2.99792458e10
h = 6.62608e-27

# f function
def f(x):
    return x*(2*x**2-3)*np.sqrt(x**2+1) + 3*np.arcsinh(x)

# Pressure
def P(x):
    return np.pi*me**4*c**5/(3*h**3) * f(x)

# density over mu_e as a function of x
def rho_mue(x):
    return 8*np.pi*x**3*me**3*mp*c**3 / (3*h**3)

# evaluate for range of x
x = np.geomspace(5e-4, 1e5, 1000)
logP = np.log10(P(x))
logrho = np.log10(rho_mue(x))

# P is proportional to rho^(power) corresponds to a slope of (power) on a log-log plot
logP_nr = (5/3) * logrho + (logP[0] - (5/3) * logrho[0]) + .5
logP_ur = (4/3) * logrho + (logP[-1] - (4/3) * logrho[-1]) + .5

# Make plot
fig, ax = plt.subplots()
ax.plot(logrho, logP, "k-")
ax.plot(logrho, logP_nr, "k--", label="NR limit")
ax.plot(logrho, logP_ur, "k:", label="UR limit")
ax.set_xlabel("$\\log_{10}(\\rho\\,\\mu_e^{-1}$ / g cm$^{-3})$")
ax.set_ylabel("$\\log_{10}(P$ / dyne cm$^{-2})$")
ax.legend()
plt.show()
plt.close()

