import numpy as np 
import scipy.optimize as opt
import matplotlib.pyplot as plt
import sys
sys.path.append("..")

# Physical constants
import constants as c

# Plot options
plt.rc("font", family="Times New Roman")
plt.rc("text", usetex=True)
plt.rc("axes", labelsize=14)

# Problem-specific constants
rho = 1e7
Q = 87.851 * 1e6 * c.eV_to_erg
mhe = 4.002602 * 931.494 * 1e6 * c.eV_to_erg / c.c**2
mni = 55.942128 * 931.494 * 1e6 * c.eV_to_erg / c.c**2

# System of equations to find the zeros of
def SahaButNotSaha(param, T):
    X4, X56 = param
    f1 = X4 + X56 - 1
    f2 = 14*np.log(X4) - np.log(X56) - 39/2*np.log(c.k*T/(2*np.pi*c.hbar**2)) + 13*np.log(rho) + 5/2*np.log(mni) - 35*np.log(mhe) + Q/(c.k*T)
    return f1, f2

# Range of temps
T9 = np.linspace(4.5, 6.5, 1000)
X4 = np.zeros(len(T9))
X56 = np.zeros(len(T9))

# Loop over T and solve for X4, X56
for i in range(len(T9)):
    roots = opt.root(SahaButNotSaha, [0.5, 0.5], args=(T9[i]*1e9,), method='lm')
    X4[i], X56[i] = roots.x

# Plot
fig, ax = plt.subplots()
ax.plot(T9, X4, label="$X_4$")
ax.plot(T9, X56, label="$X_{56}$")
ax.set_xlabel("$T$ (10$^9$ K)")
ax.set_ylabel("Mass Fraction")
ax.legend()
plt.show()
plt.close()

# Find value where X4 = X56
wh = np.nanargmin(np.abs(X4 - X56))
Teq = T9[wh]
print(f"X4 = X56 at a temperature of {Teq} x 10^9 K")

