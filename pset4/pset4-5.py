# Import packages
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

# Define constants
mHe = 4*1.67262e-24
me = 9.10938e-28
h = 6.62608e-27
k = 1.38066e-16
eV_to_erg = 1.60218e-12
chi0 = 24.6 * eV_to_erg
chi1 = 54.4 * eV_to_erg

# Function to solve the pair of equations
def Saha(x, rho, T):
    # Unpack parameters
    x1, x2 = x
    # First function
    f1 = (x1**2 + 2*x1*x2) - (1 - x1 - x2)*mHe/rho*(2*np.pi*me*k*T/h**2)**(3/2) * np.exp(-chi0/(k*T))
    # Second function
    f2 = (2*x2**2 + x1*x2) - x1*mHe/rho*(2*np.pi*me*k*T/h**2)**(3/2) * np.exp(-chi1/(k*T))
    # Return both equations
    return f1, f2

# Define the temperature vector
T = np.geomspace(1e4, 2e5, 10000)
rho = 1e-6

# Prepare output vectors
x1 = np.zeros_like(T)
x2 = np.zeros_like(T)

# Loop over T and solve the nonlinear equations for x1 and x2 using scipy
for i in range(len(T)):
    # Minimize using the Levenberg-Marquardt algorithm
    soln = opt.root(Saha, [0.2, 0.1], args=(rho, T[i]), method='lm')
    x1[i], x2[i] = soln.x

# Calculate x0 and xe based on x1 and x2
x0 = 1 - x1 - x2
xe = x1 + 2*x2

# Plot
fig, ax = plt.subplots(1,2)
ax[0].plot(T/1e4, x0, label='$x_0$')
ax[0].plot(T/1e4, x1, label='$x_1$')
ax[0].plot(T/1e4, x2, label='$x_2$')
ax[0].set_xlabel('$T$ $(10^4 K)$')
ax[0].set_ylabel('$x_i$')
ax[0].legend()
ax[1].plot(T/1e4, xe, label='$x_e$')
ax[1].set_xlabel('$T$ $(10^4 K)$')
ax[1].legend()
plt.show()
plt.close()

# Transition temperatures
T1 = T[T < 3e4][np.argmin(np.abs(x1 - x0)[T < 3e4])]
T2 = T[T > 3e4][np.argmin(np.abs(x2 - x1)[T > 3e4])]
print("Transition temperatures: T1=%.0f, T2=%.0f" % (T1, T2))

