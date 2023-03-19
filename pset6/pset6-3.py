import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spint
import sys
sys.path.append("..")
import constants as const

# Plot options
plt.rc("font", family="Times New Roman")
plt.rc("text", usetex=True)
plt.rc("axes", labelsize=14)

# Constants for this problem
Teff = 5777
g_ad = 5/3
mu = 0.6

def T(r):
    return Teff * (1 + (1 - 1/g_ad) * (mu * const.mp) / (const.k * Teff) * (const.G * const.Ms / const.Rs) * (const.Rs / r - 1))

def P(C, r):
    return C ** (1/g_ad)  * (mu * const.mp / (const.k * T(r))) ** (g_ad/(1-g_ad))

def rho(C, r):
    return C ** (1/g_ad) * (mu * const.mp / (const.k * T(r)))**(1/(1-g_ad))

# Determine the constant of proportionality C with numerical integration
C = (0.02*const.Ms / spint.quad(lambda r: 4*np.pi*r**2 * rho(1.0, r), 0.7*const.Rs, const.Rs)[0]) ** g_ad
print(f"{C=}")

# Create array of r/R from 0.7 to 1.0
R = np.linspace(0.7, 1.0, 1000)
r = const.Rs * R
# Evaluate functions at the different values of r
logT = np.log10(T(r))
logP = np.log10(P(C, r))
logrho = np.log10(rho(C, r))

# Make semilog plot
fig, ax = plt.subplots(1,3,figsize=(12,4))
ax[0].plot(R, logrho)
ax[0].set_ylabel('$\\log(\\rho$ / g cm$^{-3})$')
ax[1].plot(R, logP)
ax[1].set_ylabel('$\\log(P$ / dyne cm$^{-2})$')
ax[2].plot(R, logT)
ax[2].set_ylabel('$\\log(T$ / K$)$')
ax[0].set_xlabel('$r/R_{\\odot}$')
ax[1].set_xlabel('$r/R_{\\odot}$')
ax[2].set_xlabel('$r/R_{\\odot}$')
# fig.subplots_adjust(hspace=6.0)
plt.tight_layout()
plt.show()
plt.close()

# Get values of rho, P, and T at 0.7Rs and 1Rs
print("Values at 0.7Rsun:")
print(f"rho: {rho(C, 0.7*const.Rs):.5e}")
print(f"P: {P(C, 0.7*const.Rs):.5e}")
print(f"T: {T(0.7*const.Rs):.1f}")

print("Values at 1Rsun:")
print(f"rho: {rho(C, const.Rs):.5e}")
print(f"P: {P(C, const.Rs):.5e}")
print(f"T: {T(const.Rs):.1f}")

