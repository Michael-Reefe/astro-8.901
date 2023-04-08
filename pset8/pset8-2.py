
##### COPIED FROM PSET 7-5 #####

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spint

# Plot options
plt.rc("font", family="Times New Roman")
plt.rc("text", usetex=True)
plt.rc("axes", labelsize=14)

# Define the lane-emden syste of differential equations
def lane_emden(xi, params, n):
    phi, u = params
    dphi = u
    du = -phi**n - 2*u/xi
    return dphi, du

# Approximation for small xi
def phi_series(xi, n):
    return 1 - 1/6*xi**2 + n/120*xi**4

def u_series(xi, n):
    return -1/3*xi + n/30*xi**3

# F(n) function for the central pressure
def F(xi_final, u_final, n):
    return 1/(n+1) * (xi_final**2 * np.abs(u_final))**(-2/3)

#####################################
############ PARTS A-C ##############
#####################################

# Start at a small nonzero xi
xi0 = 1e-10
# Calculate each phi at an appropriately spaced array of points
xi = np.linspace(xi0, 10., 10000)

# Prepare output arrays for quantities to calculate for each model
xi_final = np.zeros(6)
u_final = np.zeros(6)
F_n = np.zeros(6)

# Loop over different polytrope indices
for i, n in enumerate([1.0, 1.5, 2.0, 2.5, 3.0, 3.5]):
    # Boundary conditions
    # Power series approximation for the starting point boundary conditions
    phi0 = phi_series(xi0, n)
    u0 = u_series(xi0, n)
    # Runge-Kutta numerical solver
    soln = spint.solve_ivp(lane_emden, (xi0, 10.0), (phi0, u0), method="RK45", t_eval=xi, args=(n,))
    xi_n = soln.t
    phi_n = soln.y[0, :]

    # Find where phi goes to 0 and cut it off there
    surf = np.where(phi_n <= 0)[0]
    if len(surf) > 0:
        surf = surf[0]
    else:
        surf = -1
    xi_n = xi_n[:surf]
    phi_n = phi_n[:surf]

    # This defines our xi_final and u_final
    xi_final[i] = xi_n[-1]
    u_final[i] = soln.y[1, surf]

    # Calculate F(n)
    F_n[i] = F(xi_final[i], u_final[i], n)

# Print out F(n) and its average
print(f"{F_n=}")
print(f"Average = {np.mean(F_n)}")

