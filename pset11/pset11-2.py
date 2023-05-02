# Import packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy.integrate
import tqdm

# Physical constants we will need
c = 2.99792458e10    # cm/s
G = 6.674e-8         # g^-1 cm^3 s^-2
Msun = 1.989e33      # g

# Plot options
plt.rc("font", family="Times New Roman")
plt.rc("text", usetex=True)
plt.rc("axes", labelsize=14)

############# PART A ##############

# Polytrope equation of state:
def polytrope_norm(K, rho, gamma):
    return K*rho**gamma

rho = 10**np.geomspace(14.4, 16, 1000)
gamma = 2
K = 1e5
logP = polytrope_norm(K, rho, gamma)

fig, ax = plt.subplots()
ax.plot(np.log10(rho), np.log10(logP), 'k-')
ax.set_xlabel('$\\log_{10}(\\rho$ / g cm$^{-3}$)')
ax.set_ylabel('$\\log_{10}(P$ / dyne cm$^{-2}$)')
plt.show()
plt.close()

############# PART B ##############

# Non-dimensionalizing
rho0 = 1e15               # density scale (g/cm^3)
M0 = Msun                 # mass scale (g)
R0 = (M0/rho0)**(1/3)     # length scale (cm)
T0 = (G*rho0)**(-1/2)     # time scale (s)

G0 = 1.0                  # these units have been specially chosen such that G=1
c0 = c / R0 * T0          # rescaled speed of light in units of R0/T0, becomes of order unity

print("Normalized constants:")
print(f"G = {G0} M0^-1 R0^3 T0^-2")
print(f"c = {c0} R0 T0^-1")

# All functions below are assuming inputs in normalized units

# Derivative of the equation of state:
def polytrope_norm_deriv(K, rho, gamma):
    return K*gamma*rho**(gamma-1)

# Tolman-Oppenheimer-Volkoff equation:
def dPdr_TOV(M, P, rho, r):
    return -G0*(M + 4*np.pi*r**3*P/c0**2)*(rho + P/c0**2)/(r**2*(1-2*G0*M/(r*c0**2)))

# Enclosed mass
# Here, r and rho must be arrays that go from (0, rho_c) to (r_current, rho_current)
def M_enclosed(r, rho):
    return scipy.integrate.simpson(4*np.pi*r**2*rho, x=r)

# Here, rho_current is the current step in rho (1 value) and rho_out is an array containing
# all of the previous steps in rho (and similarly for r_current and r_out). This is necessary
# to perform the integral for the mass at each step
def drhodr(r_current, rho_current, gamma, K, polytrope_model, polytrope_deriv):
    # non-dimensionalize the constant in the equation of state
    K0 = K / M0 * R0 * T0**2 * rho0**gamma
    # get the enclosed mass and the pressure
    M = M_enclosed(np.append(r_out, r_current), np.append(rho_out, rho_current))
    P = polytrope_model(K0, rho_current, gamma)
    # calculate the derivative of rho from the TOV equation
    return 1/polytrope_deriv(K0, rho_current, gamma) * dPdr_TOV(M, P, rho_current, r_current)

# Function for integrating the density
def integrate_density(rho_c, gamma, K, polytrope_model, polytrope_deriv):
    # Global arrays for r and rho that can be accessed by the drhodr function during the integration
    global r_out, rho_out

    # Initial steps
    r_step = 1e-10/R0
    rho_step = rho_c/rho0
    r_out = np.array([r_step])
    rho_out = np.array([rho_step])

    # Runge-Kutta solver -- since it doesn't support "args" we need to use an anonymous function
    solver = scipy.integrate.RK45(lambda r, rho: drhodr(r, rho, gamma, K, polytrope_model, polytrope_deriv), 
                                  r_step, (rho_step,), np.inf, rtol=1e-10, atol=1e-18)
    while solver.y[0] * rho0 > 1e-5:
        try:
            solver.step()
            r_out = np.append(r_out, solver.t)
            rho_out = np.append(rho_out, solver.y[0])
            # print(solver.status, solver.t, solver.y[0] * rho0)
        except:
            print("Warning: Solver failed before reaching the end threshold of 10^-5 x rho0")
            break

    # Convert outer radius to kilometers
    R_tot = r_out[-1] * R0 / 1e5
    # Integrate the total mass (in solar units)
    M_tot = scipy.integrate.simpson(4*np.pi*r_out**2*rho_out, x=r_out)

    return r_out, rho_out, R_tot, M_tot

# Prepare array of central densities
rho_c = 10**np.geomspace(14, 16.5, 50)

# Prepare output arrays for mass
M_tot = np.zeros(len(rho_c))
R_tot = np.zeros(len(rho_c))

# Prepare plot
fig, ax = plt.subplots()

# Loop over each central density and integrate it
for i in tqdm.trange(len(rho_c)):
    rc = rho_c[i]
    r_out, rho_out, R_tot[i], M_tot[i] = integrate_density(rc, 2, 1e5, polytrope_norm, polytrope_norm_deriv)
    # Only add every 5 models to the plot to eliminate overcrowding
    if i % 5 == 0:
        ax.plot(r_out * R0 / 1e5, np.log10(rho_out * rho0))

# Add labels to plot
ax.set_xlabel('$r$ (km)')
ax.set_ylabel('$\\log_{10}(\\rho$ / g cm$^{-3}$)')
ax.set_ylim(10, 17)
ax.set_xlim(0, 15)
plt.show()
plt.close()

################# PART C #########################

# Plot mass as a function of rho_c
fig, ax = plt.subplots()
ax.plot(np.log10(rho_c), M_tot, 'k', linestyle='-')
ax.set_xlabel('$\\log_{10}(\\rho$ / g cm$^{-3}$)')
ax.set_ylabel('$M$ ($M_{\\odot}$)')
plt.show()
plt.close()

# Maximum mass
print(f"Maximum mass = {np.max(M_tot)}")

################# PART D #########################

# Plot radius as a function of rho_c
fig, ax = plt.subplots()
ax.plot(np.log10(rho_c), R_tot, 'k', linestyle='-')
ax.set_xlabel('$\\log_{10}(\\rho$ / g cm$^{-3}$)')
ax.set_ylabel('$R$ (km)')
plt.show()
plt.close()

################# PART E #########################

# Plot mass-radius relationship
fig, ax = plt.subplots()
ax.plot(R_tot, M_tot, 'k', linestyle='-')
ax.set_xlabel('$R$ (km)')
ax.set_ylabel('$M$ ($M_{\\odot}$)')
plt.show()
plt.close()

################# PART F #########################

# Dont forget to normalize this too!
rho1 = 4.6e14/rho0

def polytrope_stiff(K, rho, gamma):
    return K*rho**gamma if rho < rho1 else K*rho1**gamma + (rho - rho1)*c0**2

def polytrope_stiff_deriv(K, rho, gamma):
    return K*gamma*rho**(gamma-1) if rho < rho1 else c0**2

fig, ax = plt.subplots()
M_stiff = np.zeros(len(rho_c))
R_stiff = np.zeros(len(rho_c))

for i in tqdm.trange(len(rho_c)):
    rc = rho_c[i]
    r_out, rho_out, R_stiff[i], M_stiff[i] = integrate_density(rc, 5/3, 5.5e9, polytrope_stiff, polytrope_stiff_deriv)
    # Only add every 5 models to the plot to eliminate overcrowding
    if i % 5 == 0:
        ax.plot(r_out * R0 / 1e5, np.log10(rho_out * rho0))

# Add labels to plot
ax.set_xlabel('$r$ (km)')
ax.set_ylabel('$\\log_{10}(\\rho$ / g cm$^{-3}$)')
ax.set_ylim(10, 17)
ax.set_xlim(0, 23)
plt.show()
plt.close()

# Plot mass as a function of rho_c
fig, ax = plt.subplots()
ax.plot(np.log10(rho_c), M_stiff, 'k', linestyle='-')
ax.set_xlabel('$\\log_{10}(\\rho$ / g cm$^{-3}$)')
ax.set_ylabel('$M$ ($M_{\\odot}$)')
plt.show()
plt.close()

# Maximum mass
print(f"Maximum mass = {np.max(M_stiff)}")

