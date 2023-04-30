import numpy as np
import matplotlib.pyplot as plt

# Plot options
plt.rc("font", family="Times New Roman")
plt.rc("text", usetex=True)
plt.rc("axes", labelsize=14)

# Constants
Msun = 1.989e33
Lsun = 3.846e33
Rsun = 6.957e10
c = 2.99792458e10
mc_i = 0.1
sb = 5.6705e-5

# Helium Core mass as a function of time (solar units)
def mc(t):
    return ((0.007*c**2 * Msun*mc_i**5)/(0.007*c**2*Msun - 1e6*Lsun*mc_i**5*t))**(1/5)

# Luminosity
def L_func(t):
    return 2e5*Lsun*mc(t)**6

# Radius
def R_func(t):
    return 3700*Rsun*mc(t)**4

# Effective temperature
def Teff_func(t):
    return (L_func(t)/(4*np.pi*R_func(t)**2*sb))**(1/4)

# Arrays
t = np.linspace(0, 10.3, 1000) 
L = L_func(t*1e9*365.25*24*3600)/Lsun
R = R_func(t*1e9*365.25*24*3600)/Rsun
Teff = Teff_func(t*1e9*365.25*24*3600)

fig, ax = plt.subplots(1,3, figsize=(9,3))
ax[0].plot(t, L, "k-")
ax[0].set_ylabel("$L$ ($L_{\\odot}$)")
ax[0].set_xlabel("$t$ (Gyr)")
ax[1].plot(t, R, "k-")
ax[1].set_ylabel("$R$ ($R_{\\odot}$)")
ax[1].set_xlabel("$t$ (Gyr)")
ax[2].plot(t, Teff, "k-")
ax[2].set_ylabel("$T_{\\rm eff}$ (K)")
ax[2].set_xlabel("$t$ (Gyr)")
plt.tight_layout()
plt.show()
plt.close()

# Plot on the HR diagram
fig, ax = plt.subplots()
ax.invert_xaxis()
ax.plot(np.log10(Teff), np.log10(L), "k-")
ax.set_xlabel("$\\log_{10}(T_{\\rm eff} / $K$)$")
ax.set_ylabel("$\\log_{10}(L / L_{\\odot})$")
plt.show()
plt.close()

