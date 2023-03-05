import numpy as np
import matplotlib.pyplot as plt

mH = 1.67262e-24
me = 9.10938e-28
h = 6.62608e-27
k = 1.38066e-16
chi0 = 13.6 * 1.60218e-12

def logrho(T):
    return np.log10(2*mH) + 1.5*np.log10(2*np.pi*me/h**2) + 1.5*np.log10(k*T) - chi0/(k*T)*np.log10(np.e)

T = np.geomspace(10., 1e6, 10000)
r = logrho(T)

fig, ax = plt.subplots()
ax.plot(r, np.log10(T), 'k-')
ax.set_xlabel('$\\log(\\rho$ / g cm$^{-3})$')
ax.set_ylabel('$\\log(T$ / $K)$')
ax.set_xlim(-10, -2)
plt.show()
plt.close()

Pe = 200

def fracHI(T):
    saha = k*T/Pe*(2*np.pi*me*k*T/(h**2))**(3/2)*np.exp(-chi0/(k*T))
    return 1/(1+saha)

T = np.linspace(0., 20000., 1000)
f = fracHI(T)

p90 = np.argmin(np.abs(f - 0.9))
p10 = np.argmin(np.abs(f - 0.1))

fig, ax = plt.subplots()
ax.plot(T, f, 'k-')
ax.axvline(T[p10], color='r', alpha=0.5, linestyle='--')
ax.axvline(T[p90], color='r', alpha=0.5, linestyle='--')
ax.set_xlabel('$T$ $(K)$')
ax.set_ylabel('$n_0/n$')
plt.show()
plt.close()

def fracHe(T):
    n0 = fracHI(T)
    return np.exp(-12.09 * 1.60218e-12 / (k*T)) * n0

fe = fracHe(T) * 1e7

fig, ax = plt.subplots()
ax.plot(T, fe, 'k-')
ax.set_xlabel('$T$ $(K)$')
ax.set_ylabel('$n_{0,*}/n$ $(\\times 10^{-7})$')
plt.show()
plt.close()
