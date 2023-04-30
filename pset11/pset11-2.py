import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import sys
sys.path.append("..")

# Physical constants
import constants as c

# Plot options
plt.rc("font", family="Times New Roman")
plt.rc("text", usetex=True)
plt.rc("axes", labelsize=14)

############# PART A ##############

def polytrope(logK, logrho, gamma):
    return logK + gamma*logrho

logrho = np.geomspace(14.4, 16, 1000)
gamma = 2
logK = 5
logP = polytrope(logK, logrho, gamma)

fig, ax = plt.subplots()
ax.plot(logrho, logP, 'k-')
ax.set_xlabel('$\\log_{10}(\\rho$ / g cm$^{-3}$)')
ax.set_ylabel('$\\log_{10}(P$ / dyne cm$^{-2}$)')
plt.show()
plt.close()

############# PART B ##############

# Non-dimensionalizing
rho0 = 1e15
M0 = c.Msun
R0 = (M0/rho0)**(1/3)
T0 = (c.G*rho0)**(-1/2)

K0 = 1e5 / M0 * R0 * T0**2 * rho0**gamma
G0 = c.G * M0 / R0**3 * T0**2
c0 = c.c / R0 * T0
print("Normalized constants:")
print(f"K = {K0} M0 R0^-1 T0^-2 rho0^-gamma")
print(f"G = {G0} M0^-1 R0^3 T0^-2")
print(f"c = {c0} R0 T0^-1")

# All functions below are assuming inputs in normalized units
def polytrope_norm(K, rho, gamma):
    return K*rho**gamma

def deriv_polytrope_norm(K, rho, gamma):
    return K*gamma*rho**(gamma-1)

def dPdr_TOV(M, P, rho, r):
    return -G0*(M + 4*np.pi*r**3*P/c0**2)*(rho + P/c0**2)/(r**2*(1-2*G0*M/(r*c0**2)))

def dMdr(r, rho):
    return 4*np.pi*r**2*rho

def M_enclosed(r, rho):
    return scipy.integrate.quad(dMdr, 0, r, args=(rho,))[0]

def drhodr(r, rho):
    M = M_enclosed(r, rho)
    P = polytrope_norm(K0, rho, gamma)
    return 1/deriv_polytrope_norm(K0, rho, gamma) * dPdr_TOV(M, P, rho, r)

# Different central densities
rho_c = 10**np.geomspace(14, 16.5, 5)

# Prepare plot
fig, ax = plt.subplots()

for i, rc in enumerate(rho_c):
    print(f"Beginning integration for rho_c={rc} ({i+1}/{len(rho_c)})")
    # Initial steps
    r_step = 1e-10/R0
    rho_step = rc/rho0
    # Prepare output arrays
    r_out = [r_step]
    rho_out = [rho_step]
    # Runge-Kutta solver
    solver = scipy.integrate.RK45(drhodr, r_step, (rho_step,), np.inf, max_step=1e3/R0)
    while solver.y > 1e-1/rho0:
        print(solver.status, solver.t, solver.y)
        solver.step()
        r_out.append(solver.t)
        rho_out.append(solver.y[0])
    # Add to plot
    ax.plot(np.array(r_out) * R0 / 1e6, np.log10(np.array(rho_out) * rho0))

# Add labels to plot
ax.set_xlabel('$r$ ($10^6$ cm)')
ax.set_ylabel('$\\log_{10}(\\rho$ / g cm$^{-3}$)')
ax.set_ylim(-1, 17)
ax.set_xlim(0)
plt.show()
plt.close()

