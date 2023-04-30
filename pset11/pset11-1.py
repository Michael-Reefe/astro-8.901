import numpy as np
import matplotlib.pyplot as plt

# Plot options
plt.rc("font", family="Times New Roman")
plt.rc("text", usetex=True)
plt.rc("axes", labelsize=14)

c = 2.99792458e10
G = 6.67430e-8

def M_upplim(rho):
    return 4/(9*np.sqrt(3*np.pi)) * c**3 / np.sqrt(rho*G**3)

logrho = np.geomspace(14, 16.5, 1000)
logM = np.log10(M_upplim(10**logrho) / 1.989e33)

fig, ax = plt.subplots()
ax.plot(logrho, logM, 'k-')
ax.set_xlabel('$\\log_{10}(\\rho$ / g cm$^{-3}$)')
ax.set_ylabel('$\\log_{10}(M$ / $M_{\\odot}$)')
plt.show()
plt.close()

