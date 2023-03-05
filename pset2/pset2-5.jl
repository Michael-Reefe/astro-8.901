using Roots
using PyPlot
using LaTeXStrings

plt.switch_backend("tkagg")

# Constants
Mjup = 1.899e30  # Jupiter mass in g
Msun = 1.989e33  # Solar mass in g
d = 24*60^2      # Seconds per day
deg = π/180      # Radians per degree
G = 6.674e-8     # Gravitational constant in CGS units

# Orbital parameters and masses
Mp = 4.2 * Mjup     # Planet mass
Ms = 1.05 * Msun    # Star mass
P = 111.44 * d      # Orbital period
e = 0.933           # Eccentricity
i = 89.32 * deg     # Inclination
ω = 300.83 * deg    # Argument of periastron
t_p = 0.            # Time of periastron

# Kepler's Equation for the eccentric anomaly
function Kepler(e, ψ, ω_m, t)
    M = ω_m * t        # --> mean anomaly
    ψ - e*sin(ψ) - M
end

# Write a function to solve Kepler's equation numerically for the eccentric
# anomaly ψ for a given time t
function solveKepler(e, ω_m, t_p, t; acc=1.)
    # Take the modulus with the period to keep t in the range (0, P)
    t = mod(t - t_p, 2π/ω_m)
    if abs(t) ≤ acc
        0.
    else
        find_zero(x -> Kepler(e, x, ω_m, t), (0., 2π))
    end
end

# Convert the eccentric anomaly to the true anomaly
function trueAnomaly(e, ψ)
    tan_2phi = √((1+e)/(1-e)) * tan(ψ/2)
    2atan(tan_2phi)
end

# Use the true anomaly to calculate the radial velocity at time t
function vrad1(P, M1, M2, I, e, ω, t_p, t; acc=1.)
    ω_m = 2π/P                                  # First find the mean angular frequency
    ψ = solveKepler(e, ω_m, t_p, t; acc=acc)    # Numerically solve Kepler's equation for the eccentric anomaly
    ϕ = trueAnomaly(e, ψ)                       # Convert eccentric anomaly to true anomaly
    (2π*G/P)^(1/3) * (M2*sin(I)) / (M1+M2)^(2/3) * 1/√(1-e^2) * (cos(ϕ+ω) + e*cos(ω))  # Calculate radial velocity (cm/s)
end

# Apply across linearly spaced points in time over 2 orbits
t_p = P
t = 0:1:(2P)                                  # Resolution of 1 second(!) to get accurate shape at the turnaround points
v = vrad1.(P, Ms, Mp, i, e, ω, t_p, t)        # (convert to m/s)

# Plot
fig, ax = plt.subplots()
ax.plot((t .- t_p) ./ 86400, v ./ 100, "k-")
ax.set_xlabel(L"$T - T_P$ (days)")
ax.set_ylabel(L"$v_{\rm rad,1}$ (m s$^{-1}$)")
plt.show()
plt.close()

