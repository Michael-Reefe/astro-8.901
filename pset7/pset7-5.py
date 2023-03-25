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

# Dimensionless moment of inertia
def k_func(n, xi_n, phi_n, xi_final, u_final):
    # Numerically integrate xi^4 * phi^n over xi
    integral = spint.simpson(xi_n**4 * phi_n**n, x=xi_n)
    # Return the dimensionless moment of inertia
    return  integral / (xi_final**4 * np.abs(u_final))

# Dimensionless gravitational potential energy
def omega_func(n, xi_n, phi_n, xi_final, u_final):
    # Make a function to evaluate the first integral up to an upper bound index
    int_func = lambda index: spint.simpson(xi_n[:index]**2 * phi_n[:index]**n, x=xi_n[:index])
    # Do the integral for all possible upper bounds
    enclosed_int = np.array([int_func(i) for i in range(1, len(xi_n))])
    # Append a 0 to the beginning since the first integral does not cover any space in xi
    enclosed_int = np.append(0., enclosed_int)
    # Do a second integral over the first integral
    integral = spint.simpson(xi_n * phi_n**n * enclosed_int, x=xi_n)
    # Return the dimensionless energy
    return integral / (xi_final**3 * u_final**2)

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
k = np.zeros(6)
omega = np.zeros(6)

# Prepare plots
fig, ax = plt.subplots(1, 2)

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

    # Use these to calculate k and omega
    k[i] = k_func(n, xi_n, phi_n, xi_final[i], u_final[i])
    omega[i] = omega_func(n, xi_n, phi_n, xi_final[i], u_final[i])

    # Add to plot
    ax[0].plot(xi_n, phi_n, label=f"$n = {n:.1f}$")
    ax[1].plot(xi_n, phi_n**n, label=f"$n = {n:.1f}$")

    # Save the n=3 polytrope for later
    if n == 3.0:
        xi_sun = xi_n
        phi_sun = phi_n
        xi_final_sun = xi_final[i]
        u_final_sun = u_final[i]

# Pretty plots
ax[0].set_xlabel("$\\xi$")
ax[1].set_xlabel("$\\xi$")
ax[0].set_ylabel("$\\phi_n(\\xi)$")
ax[1].set_ylabel("$\\phi_n^n(\\xi)$")
ax[0].legend(loc="upper right")
ax[1].legend(loc="upper right")
ax[0].set_xlim(0., 10.)
ax[1].set_xlim(0., 5.)
ax[0].set_ylim(0., 1.)
ax[1].set_ylim(0., 1.)
plt.tight_layout()
plt.show()
plt.close()

# Save xi_final, -u_final, omega, and k for each polytrope
np.savetxt("polytrope_table.txt", np.column_stack((xi_final, -u_final, k, omega)), fmt="%.18f", delimiter=' ', newline='\n',
           header='xi_1 -u_1 k omega')


#####################################
############ PARTS D-F ##############
#####################################

# Constants
kB = 1.38066e-16
mp = 1.67262e-24
G = 6.674e-8

# Parameters
rhoc = 158
Tc = 15.7e6
X = 0.6
mu = 1/(2*X + 3/4*(1-X))

# Take our n=3 polytrope model from before and compute the total mass and radius
n = 3
lambda_n = ((n+1)*kB*Tc/(mu*mp)/(4*np.pi*G*rhoc))**(1/2)
print(f"lambda_n = {lambda_n:.5e}")

# Compute the total mass and radius of the star with dimensions
M = 4*np.pi*rhoc*lambda_n**3*xi_final_sun**2*np.abs(u_final_sun)
R = lambda_n*xi_final_sun
print(f"M = {M:.5e} g = {M/1.989e33:.3f} Msun")
print(f"R = {R:.5e} cm = {R/6.96e10:.3f} Rsun")

# Plot the dimensionful temperature and density
T = Tc * phi_sun
rho = rhoc * phi_sun**n
r = lambda_n*xi_sun

fig, ax = plt.subplots(1, 2)
ax[0].plot(r/R, np.log10(T))
ax[1].plot(r/R, np.log10(rho))
ax[0].set_xlabel("$r/R$")
ax[0].set_ylabel("$\\log_{10}(T$ / K$)$")
ax[1].set_xlabel("$r/R$")
ax[1].set_ylabel("$\\log_{10}(\\rho$ / g cm$^{-3})$")
ax[0].set_xlim(left=0.)
ax[0].set_ylim(bottom=np.min(np.log10(T)))
ax[1].set_xlim(left=0.)
ax[1].set_ylim(bottom=np.min(np.log10(rho)))
plt.tight_layout()
plt.show()
plt.close()

# Integration constant for Luminosity
C = 4*np.pi*(2.46e6)*lambda_n**3*rhoc**2*X
print(f"C = {C:.5e} erg/s")

# Numerically integrate for the luminosity
L = C * spint.simpson(xi_sun**2*phi_sun**(2*n-2/3)*(Tc/1e6)**(-2/3)*np.exp(-33.81*(Tc/1e6)**(-1/3)*phi_sun**(-1/3)), x=xi_sun)
print(f"L = {L:.5e} erg/s = {L/3.839e33:.3f} Lsun")
