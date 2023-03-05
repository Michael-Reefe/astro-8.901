import numpy as np
import matplotlib.pyplot as plt

# Bohr radius in cm
a0 = 5.29177e-9
# Proton mass in g
mp = 1.67262e-24

def nmax(rho):
    return np.sqrt(1/a0*(3*mp/(4*np.pi*rho))**(1/3))

# density vector
rho = np.geomspace(1e-6, 1e3, 1000)
n = nmax(rho)

# plot
fig, ax = plt.subplots()
ax.plot(np.log10(rho), n, 'k-')
ax.set_xlabel('$\\log(\\rho$ / g cm$^{-3})$')
ax.set_ylabel('$n_{\\rm max}$')
plt.show()
plt.close()

