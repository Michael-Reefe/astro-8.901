import numpy as np
import scipy.integrate as spint
import scipy.interpolate as interp
import matplotlib.pyplot as plt

# Plot options
plt.rc("font", family="Times New Roman")
plt.rc("text", usetex=True)
plt.rc("axes", labelsize=14)

# Constants
G = 6.674e-8
mue = 2
KNR = 1.00e13
a = 1.557e8
rho0 = 3.79e6
Rsun = 6.957e10

# Coupled differential equations for theta and V
def theta_V_diff_eqs(s, params):
    # unpack params
    theta, V = params
    # first coupled equation
    dtheta = V
    # second coupled equation (end me)
    dV = 3*(1+theta**(2/3))**(3/2)/((4*theta**(2/3)+5)*theta**(2/3))*theta*(
            -4*np.pi*G*a**2*rho0**(1/3)/(KNR*mue**(-5/3))*theta - (4*theta**(4/3) + 11*theta**(2/3) + 10)/(9*(1+theta**(2/3))**(5/2)*theta**(1/3))*V**2/theta
            ) + V**2/theta
    if s == 0. and V == 0.:
        dV -= 2
    else:
        dV -= 2*V/s
    return dtheta, dV

# integral for dimensionless mass
def dimensionless_mass(s, theta):
    # use simpson's method
    return spint.simpson(theta*s**2, x=s)


# All choices for the critical density
rhoc = 10 ** np.arange(4, 12.5, 0.5)
ss = []
thetas = []
Vs = []
Ms = []

# Function for integrating a specific model
def integrate_model(rhoc_i, plot=False):

    # Boundary conditions
    thetac = theta0 = rhoc_i / rho0
    V0 = 0.

    # Set up the integrator
    solver = spint.RK45(theta_V_diff_eqs, 0, [theta0, V0], np.inf, rtol=1e-12, atol=1e-14)

    # Set up output arrays
    s = np.array([0.])
    theta = np.array([theta0])
    V = np.array([V0])

    # Step until we reach an ending threshold of theta/thetac = 1e-8
    i = 0
    while (theta[-1]/thetac) > 1e-8:
        solver.step()
        s = np.append(s, solver.t)
        theta = np.append(theta, solver.y[0])
        V = np.append(V, solver.y[1])
        if i % 100 == 0:
            print(f"Step {i}: theta={theta[-1]}, V={V[-1]}")
        i += 1

    # Calculate mass
    M = 0.089 * dimensionless_mass(s, theta)
    
    # plot for sanity checking
    if plot:
        fig, ax = plt.subplots()
        ax.plot(s*a, theta*rhoc_i, "k-")
        ax.set_xlabel("$r$ (cm)")
        ax.set_ylabel("$\\rho$ (g cm$^{-3}$)")
        ax.set_title("$M = %.2f$ M$_{\\odot}$" % M)
        fig.tight_layout()
        plt.show()
        plt.close()

    return s, theta, V, M

# Loop over each model and integrate it
for rhoc_i in rhoc:

    s, theta, V, M = integrate_model(rhoc_i)
    # append to overall output arrays
    ss.append(s)
    thetas.append(theta)
    Vs.append(V)
    Ms.append(M)


# Plot mass as a function of rho_crit
fig, ax = plt.subplots()
ax.plot(np.log10(rhoc), Ms, "k.")
ax.set_xlabel("$\\log_{10}(\\rho_{\\rm c} / $ g cm$^{-3})$")
ax.set_ylabel("$M$ (M$_{\\odot})$")
plt.show()
plt.close()

# Maximum stable mass: extrapolate the derivatives
dM = (Ms[-1] - Ms[-2]) / 0.5
d2M = (Ms[-1] - 2*Ms[-2] + Ms[-3]) / 0.5**2
# Find how far for the first derivative to go to 0 (i.e. the mass flattens out)
dist = dM / d2M
# Extrapolate the mass at this location
M_max = Ms[-1] + dist * dM

print(f"Maximum stable mass: M = {Ms[-1]} Msun")

# Radii of each model in units of Rsun/100
Rs =[s[-1]*a*100/Rsun for s in ss]

# Radius as a function of rho_crit
fig, ax = plt.subplots()
ax.plot(np.log10(rhoc), Rs, "k.")
ax.set_xlabel("$\\log_{10}(\\rho_{\\rm c} / $ g cm$^{-3})$")
ax.set_ylabel("$R$ (R$_{\\odot}$/100)")
plt.show()
plt.close()

# Radius as a function of mass
fig, ax = plt.subplots()
ax.plot(Ms, Rs, "k.")
ax.set_xlabel("$M$ (M$_{\\odot})$")
ax.set_ylabel("$R$ (R$_{\\odot}$/100)")
plt.show()
plt.close()

# Interpolate to find the critical density needed for a model with M = 1 Msun and 1.3 Msun
interp_func = interp.interp1d(Ms, np.log10(rhoc), kind='cubic')
rho_M1 = interp_func(1.0)
rho_M13 = interp_func(1.3)
print(f"Critical density for 1 Msun: {rho_M1}, 1.3 Msun: {rho_M13}")

# Integrate these models
s1, theta1, V1, M1 = integrate_model(10**rho_M1, plot=True)
s13, theta13, V13, M13 = integrate_model(10**rho_M13, plot=True)

# Sanity check:
print(f"M1 = {M1} Msun")
print(f"M13 = {M13} Msun")

