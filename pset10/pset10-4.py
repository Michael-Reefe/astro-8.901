import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")

# Physical constants
import constants as c

# Plot options
plt.rc("font", family="Times New Roman")
plt.rc("text", usetex=True)
plt.rc("axes", labelsize=14)

# molecular weights
X = 0
Y = 0.9
Z = 0.1
mu = 1/(2*X + 0.75*Y + 0.5*(7/12+9/16)*Z)
mu_e = 2
mu_i = 1/(0.75*7/12 + 0.25*9/16)

# normalization constants
T0 = 1
rho0 = 1
D = 2.01e5
kappa0 = 4.34e25 * Z * (1 + X)
M = c.Msun
wow = 64*np.pi*c.a*c.c*rho0*c.mp*c.G*D**3/(51*T0**(7/2)*c.k)

# Luminosity function
def L(Tc):
    return wow * (mu/mu_e**2) * M / kappa0 * Tc**(7/2) / c.Lsun

# Evaluate for the given Tc
L_omg = L(1e7)
print(f"L = {L_omg} Lsun")

def Tc(L):
    return (L / wow * (mu_e**2/mu) * kappa0 / M * c.Lsun)**(2/7)

Tc0 = Tc(1e4)
print(f"Tc0 = {Tc0} K")

# Timescale
tau = (3/2)/wow/Tc0**(5/2)*mu_e**2/mu/mu_i * c.k * kappa0 / c.mp / 365.25 / 24 / 3600 / 1e6
print(f"tau = {tau} Myr")

# White dwarf cooling time
def twd(L, Mwd):
    return 9.2e6*(mu_i/12)**-1*(mu_e/2)**(4/7)*(mu/2)**(-2/7)*(Z*(1+X))**(2/7)*(Mwd)**(5/7)*(L)**(-5/7)

# Main sequence lifetime
def tpn(M):
    return 10**(9.921 - 3.6648*np.log10(M) + 1.9697*np.log10(M)**2 - 0.9369*np.log10(M)**3)

# Galaxy age estimate
gal_age = tpn(8) + twd(10**-4.4, 0.6)
gal_age /= 1e9
print(tpn(8))
print(twd(10**-4.4, 0.6))
print(f"Galaxy disk age estimate: {gal_age} Gyr")

