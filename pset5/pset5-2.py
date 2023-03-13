# Import the stuff
import numpy as np
import matplotlib.pyplot as plt

# Plot options
plt.rc("font", family="Times New Roman")
plt.rc("text", usetex=True)
plt.rc("axes", labelsize=14)

# Constants
h = 6.62608e-27
c = 2.99792458e10
k = 1.38066e-16

################## PART A #####################

I0 = 4.03e14
a0 = 0.2593
a1 = 0.8724
a2 = -0.1336/2

# Source function
def S_lam(I0, a0, a1, a2, tau):
    return I0 * (a0 + a1 * tau + a2 * tau**2)

# Temperature
def T_lam(I0, a0, a1, a2, tau, lam):
    return h*c/(lam*k) / np.log(2*h*c**2 / lam**5 / S_lam(I0, a0, a1, a2, tau) + 1)

# Specific wavelength we're interested in
lam = 5010 * 1e-8
# Plot over range of taus
tau = np.linspace(0., 13, 1000)
S = S_lam(I0, a0, a1, a2, tau)
T = T_lam(I0, a0, a1, a2, tau, lam)

fig, ax = plt.subplots(1, 2)
ax[0].plot(tau, S / 1e14)
ax[1].plot(tau, T)
ax[0].set_xlabel("$\\tau_{\\lambda}$")
ax[0].set_ylabel("$S(\\tau_{\\lambda})$ [$10^{14}$ erg s$^{-1}$ cm$^{-2}$ cm$^{-1}$ sr$^{-1}$]")
ax[1].set_xlabel("$\\tau_{\\lambda}$")
ax[1].set_ylabel("$T(\\tau_{\\lambda})$ [$K$]")
ax[1].sharex(ax[0])
plt.show()
plt.close()



################## PART B #####################

# Constants for different wavelengths
lam = np.array([3737, 5010, 6990, 8660, 12250, 16550, 20970]) * 1e-8
a0 = np.array([0.1435, 0.2593, 0.4128, 0.5141, 0.5969, 0.6894, 0.7249])
a1 = np.array([0.9481, 0.8742, 0.7525, 0.6497, 0.5667, 0.4563, 0.4100])
a2 = np.array([-0.0920, -0.1336, -0.1761, -0.1657, -0.1646, -0.1472, -0.1360]) / 2
I0 = np.array([42, 40.3, 25, 15.5, 7.7, 3.6, 1.6]) * 1e13

tau = np.linspace(0, 3, 1000)
t60 = np.zeros(len(lam))
t65 = np.zeros(len(lam))
fig, ax = plt.subplots()
# Loop over each point and add to the plot
for i in range(len(lam)):
    Ti = T_lam(I0[i], a0[i], a1[i], a2[i], tau, lam[i])
    ax.plot(tau, Ti, "k-")
    y_annot = Ti[-1] if i not in (0,4) else Ti[-1]-100 if i == 0 else Ti[-1]+100
    ax.annotate("%.0f $\\mathring{A}$" % (lam[i]*1e8), xy=(3.2, y_annot), annotation_clip=False, ha='center', va='center')
    # Find the tau at which T = 6000 and 6500 K
    t60[i] = tau[np.argmin(np.abs(Ti - 6000))]
    t65[i] = tau[np.argmin(np.abs(Ti - 6500))]

ax.set_xlabel("$\\tau_{\\lambda}$")
ax.set_ylabel("$T(\\tau_{\\lambda})$ [$K$]")
ax.set_xlim(0, 3)
ax.axhline(6000, color="k", alpha=0.5, linestyle="--")
ax.axhline(6500, color="k", alpha=0.5, linestyle="--")
plt.show()
plt.close()

# Plot out t60 and t65 as a function of wavelength
fig, ax = plt.subplots()
ax.plot(np.log10(lam*1e8), np.log10(t60), "k+", linestyle="--")
ax.annotate("%.0f $K$" % 6000, xy=(np.log10(lam[-1]*1e8+5000), np.log10(t60)[-1]), annotation_clip=False, ha='center', va='center')
ax.plot(np.log10(lam*1e8), np.log10(t65), "kx", linestyle="--")
ax.annotate("%.0f $K$" % 6500, xy=(np.log10(lam[-1]*1e8+5000), np.log10(t65)[-1]), annotation_clip=False, ha='center', va='center')
ax.set_xlabel("$\\log_{10}(\\lambda)$ [$\\mathring{A}$]")
ax.set_ylabel("$\\log_{10}(\\tau_{\\lambda})$")
plt.show()
plt.close()



