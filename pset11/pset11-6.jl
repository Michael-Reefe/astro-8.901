# Get package for integrating ODEs
using DifferentialEquations
using PyPlot
using LaTeXStrings

################## PART A #####################

# Define our differential equations
function photon_orbit!(ds, s, p, λ)
    # Unpack the vector s in (r, φ) components
    r, φ = s
    # Unpack parameters
    b, sense = p
    # Photon potential
    V_phot = 1/r^2 * (1 - 2/r)
    # Calculate new derivatives
    dr = 1/b^2 - V_phot ≥ 0 ? sense * √(1/b^2 - V_phot) : -sense * √(V_phot - 1/b^2)
    dφ = 1/r^2
    # Update ds in-place
    ds .= [dr, dφ]
end

# Choose an initial r between 2.1-2.5, and a range of angles
r₀ = 2.3
φ₀ = 0
s₀ = [r₀, φ₀]

# Choose a range of angles ψ to start the photon at different trajectories
# ψ = 0:0.1:(2π-0.05)
ψ = 0:0.05:π
# Impact parameters
b = @. r₀ * sin(ψ) / √(1 - 2/r₀)
# sense of motion (inwards or outwards)
sense = sign.(π/2 .- ψ)

# Prepare output arrays in cartesian space
λ = 0:0.01:1000
x = zeros(length(λ), length(b))
y = zeros(length(λ), length(b))

# Loop over b and integrate
for i ∈ 1:length(b)
    println("Integrating photon trajectory for impact parameter $(b[i]), sense $(sense[i])")
    # Create the problem
    prob = ODEProblem{true}(photon_orbit!, s₀, (0, 1000), [b[i], sense[i]])
    # Integrate
    soln = solve(prob, RK4(), saveat=λ)
    # Unpack results
    xi = soln[1, :] .* cos.(soln[2, :])
    yi = soln[1, :] .* sin.(soln[2, :])
    # Pad with zeros if the solution became unstable
    if length(xi) < length(λ)
        xi = [xi; zeros(length(λ)-length(xi))]
        yi = [yi; zeros(length(λ)-length(yi))]
    end
    x[:, i] .= xi
    y[:, i] .= yi
end

# Plot the trajectories with matplotlib
fig, ax = plt.subplots()
# Black circle to indicate the event horizon
BH = plt.matplotlib.patches.Circle((0,0), 2, color="k")
ax.add_patch(BH)
for i ∈ 1:length(b)
    ax.plot(x[:, i], y[:, i], "r-")
end
ax.set_xlabel(L"x")
ax.set_ylabel(L"y")
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_aspect("equal")
plt.show()
plt.close()

################## PARTS B & C #####################

# Same as part a, but with specific values for b
r₀ = [3.5, 3.5, 2.5, 2.5]
sinψ = @. 3√(3)/r₀*√(1 - 2/r₀)
sinψ .+= [-1e-3, 1e-3, 1e-3, -1e-3]
b = @. r₀ * sinψ / √(1 - 2/r₀)
# reverse sign for inward moving photons
sense = [-1, -1, 1, 1]

λ = 0:0.01:10000
x = zeros(length(λ), length(b))
y = zeros(length(λ), length(b))

for i ∈ 1:length(b)
    println("Integrating test photon trajetory at r=3")
    prob = ODEProblem{true}(photon_orbit!, [r₀[i], 0], (0, 10000), [b[i], sense[i]])
    soln = solve(prob, RK4(), saveat=λ)
    # Unpack results
    xi = soln[1, :] .* cos.(soln[2, :])
    yi = soln[1, :] .* sin.(soln[2, :])
    # Pad with zeros if the solution became unstable
    if length(xi) < length(λ)
        xi = [xi; zeros(length(λ)-length(xi))]
        yi = [yi; zeros(length(λ)-length(yi))]
    end
    x[:, i] .= xi
    y[:, i] .= yi
end

# Plot the trajectories with matplotlib
fig, ax = plt.subplots(1, 2)
for i ∈ 1:2
    # Black circle to indicate the event horizon
    BH = plt.matplotlib.patches.Circle((0,0), 2, color="k")
    ax[i].add_patch(BH)
    # Red = falls in, blue = escapes
    ax[i].plot(x[:, 2i-1], y[:, 2i-1], "r-")
    ax[i].plot(x[:, 2i], y[:, 2i], "b-")
end
for i ∈ 1:2
    ax[i].set_xlabel(L"x")
    ax[i].set_ylabel(L"y")
    ax[i].set_xlim(-5, 5)
    ax[i].set_ylim(-5, 5)
    ax[i].set_aspect("equal")
end
plt.show()
plt.close()
