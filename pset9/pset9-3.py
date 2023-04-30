import numpy as np
import matplotlib.pyplot as plt

# Plot options
plt.rc("font", family="Times New Roman")
plt.rc("text", usetex=True)
plt.rc("axes", labelsize=14)

# Constants
G = 6.674e-8
c = 2.99792458e10

# Mdot function
def logMdot(M, L, Gamma):
    return -1.37 + 2.07*np.log10(L/1e6) - 0.5*np.log10(2*(1-Gamma)*G*M*1.989e33) + np.log10(2.6e5)

# Create arrays
Lsolar = np.array([1e5, 3e5, 1e6, 2e6])
Msolar = np.linspace(0.1, 100, 1000)

fig, ax = plt.subplots()
# Loop over logL
for Li in Lsolar:
    logMdoti = logMdot(Msolar, Li, 0.)
    ax.plot(np.log10(Msolar), logMdoti, label="$\\log_{10}(L / L_{\\odot}) = %.3f$" % np.log10(Li))

ax.set_xlabel("$\\log_{10}(M$ / $M_{\\odot}$)")
ax.set_ylabel("$\\log_{10}(\\dot{M}$ / $M_{\\odot}$ yr$^{-1}$)")
ax.legend()
plt.show()
plt.close()

# Redo with nonzero Gamma_e
fig, ax = plt.subplots()
for Li in Lsolar:
    Gammai = 0.3*Li*3.846e33/(4*np.pi*c*G*Msolar*1.989e33)
    logMdoti = logMdot(Msolar, Li, Gammai)
    ax.plot(np.log10(Msolar), logMdoti, label="$\\log_{10}(L / L_{\\odot}) = %.3f$" % np.log10(Li))

ax.set_xlabel("$\\log_{10}(M$ / $M_{\\odot}$)")
ax.set_ylabel("$\\log_{10}(\\dot{M}$ / $M_{\\odot}$ yr$^{-1}$)")
ax.legend()
plt.show()
plt.close()

# Reimers law
def mass_reimer(t, M0, L, R, eta):
    return np.sqrt(M0**2 - (8e-13*eta*L*R*t)**2)

M0 = 1
L = 7000
R = 310
eta = 1

# Time in Megayears
t = np.linspace(0, 0.7, 1000)
# Mass over time
Mt = mass_reimer(t*1e6, M0, L, R, eta)

fig, ax = plt.subplots()
ax.plot(t, Mt, "k-")
ax.set_xlabel("$t$ (Myr)")
ax.set_ylabel("$M$ ($M_{\\odot}$)")
plt.show()
plt.close()


