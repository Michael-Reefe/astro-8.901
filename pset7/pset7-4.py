import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spint

# Constants
rho = 1e4
T9 = 0.15
mp = 1.67262e-24
MeV_to_erg = 1.60218e-12 * 1e6
# Heats from each reaction in ergs
Q3a = 7.275 * MeV_to_erg
Qa12 = 7.161 * MeV_to_erg
Qa16 = 4.734 * MeV_to_erg
s_per_year = 365.25*24*3600

# Energy generation rates
def epsm_3a(X4):
    return 5.1e8*rho**2*X4**3/T9**3 * np.exp(-4.4027/T9)

def epsm_a12(X4, X12):
    return 1.5e25*rho*X4*X12/T9**2 * (1 + 0.0489*T9**(-2/3))**(-2) * np.exp(-32.19*T9**(-1/3) - (0.286*T9)**2)

def epsm_a16(X4, X16):
    return 6.69e26*rho*X4*X16/T9**(2/3) * np.exp(-39.757*T9**(-1/3) - (0.631*T9)**2)

# Coupled differential equations for the mass fractions X4, X12, X16, X20
def dXdt(t, X):
    # Unpack parameters
    X4, X12, X16, X20 = X
    # Calculate each energy generation rate
    e3a = epsm_3a(X4)
    ea12 = epsm_a12(X4, X12)
    ea16 = epsm_a16(X4, X16)
    # Calculate the derivatives of each based on the others
    dX4  = -4*mp*(3*e3a/Q3a + ea12/Qa12 + ea16/Qa16)
    dX12 = 12*mp*(e3a/Q3a - ea12/Qa12)
    dX16 = 16*mp*(ea12/Qa12 - ea16/Qa16)
    dX20 = 20*mp*(ea16/Qa16)
    # Return an array of derivatives
    return dX4, dX12, dX16, dX20

# Initial conditions
X0 = np.array([1., 0., 0., 0.])
t0 = 0.
tmax = 10**8

t = np.geomspace(t0+1, tmax-1, 1000) * s_per_year

# Integrate with Runge-Kutta
print("Integrating with Runge-Kutta...")
soln = spint.solve_ivp(dXdt, (t0*s_per_year, tmax*s_per_year), X0, method="RK45", t_eval=t)
t = soln.t
X = soln.y
print("Done!")

# Cutoff point
he0 = np.where(X[0,:] < 1e-6)[0][0]

# Plot
fig, ax = plt.subplots()
ax.semilogx(t[:he0]/s_per_year, X[0, :he0], label="$X_{4}$")
ax.semilogx(t[:he0]/s_per_year, X[1, :he0], label="$X_{12}$")
ax.semilogx(t[:he0]/s_per_year, X[2, :he0], label="$X_{16}$")
ax.semilogx(t[:he0]/s_per_year, X[3, :he0], label="$X_{20}$")
ax.set_xlabel("$t$ (yr)")
ax.set_ylabel("$X$")
ax.legend()
plt.show()
plt.close()

# Sanity check to make sure they all add up to 1 consistently
# print(X[0, :he0] + X[1, :he0] + X[2, :he0] + X[3, :he0])
print("Final values:")
print("X4 = ", X[0, -1])
print("X12 = ", X[1, -1])
print("X16 = ", X[2, -1])
print("X20 = ", X[3, -1])


