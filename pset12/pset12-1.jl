# Get package for integrating ODEs
using DifferentialEquations
using PyPlot
using LaTeXStrings

################## PART D #####################

# Plot options
plt.rc("font", family="Times New Roman")
plt.rc("text", usetex=true)
plt.rc("axes", labelsize=14)

Veff(r, ℓ) = (1 - 2/r) * (1 + ℓ^2/r^2)


# Define out differential equations
function particle_orbit_gr!(ds, s, p, τ)
    # Unpack the vector s into (r, u, φ) components
    r, u, φ, t = s
    # Unpack the parameters (energy and angular momentum)
    ℓ = p[1]
    # Calculate new derivatives
    dr = u
    du = ℓ^2/r^3 * (1 - 2/r) - 1/r^2 * (1 + ℓ^2/r^2)
    dφ = ℓ/r^2
    dt = r ≈ 2 ? 0 : ε / (1 - 2/r)
    # Update ds in-place
    ds .= [dr, du, dφ, dt]
end


function integrate_particle_orbit_gr(s₀, ℓ, ε; cartesian=false, verbose=false, n_orbits=0,
    rtol=1e-5, atol=1e-8, dtmax=1)
    # Prepare output arrays 
    τ = Float64[]  
    r = Float64[]
    drdτ = Float64[]
    φ = Float64[]
    t = Float64[]

    println("Integrating particle trajectory for ℓ = $ℓ and ε = $ε")
    println("Initial conditions: $s₀")

    # Create the problem with an unlimited proper time range
    prob = ODEProblem{true}(particle_orbit_gr!, s₀, (0, Inf), (ℓ,))
    # Initialize the integrator
    integrator = init(prob, RK4(), save_everystep=false, abstol=atol, reltol=rtol, dtmax=dtmax)
    stopping_condition = false
    # Loop over steps and continue until we reach r = 0 or some other stopping condition
    while !stopping_condition
        push!(τ, integrator.t)
        push!(r, integrator.u[1])
        push!(drdτ, integrator.u[2])
        push!(φ, integrator.u[3])
        push!(t, integrator.u[4])
        step!(integrator)
        # Stop if r goes to 0 or some large number
        ded = integrator.u[1] < 1e-4
        gon = integrator.u[1] > 100
        # Also stop if we've done n_orbits
        loopy = n_orbits > 0 && integrator.u[3] > n_orbits*2π

        stopping_condition = ded || gon || loopy
        if verbose
            println("$(integrator.t) $(integrator.u)")
        end
    end

    !cartesian ? (τ, t, r, φ) : (τ, t, r .* cos.(φ), r .* sin.(φ))
end


function plot_trajectory(x, y; limits=(-10, 10, -10, 10))
    # Plot
    fig, ax = plt.subplots()
    BH = plt.matplotlib.patches.Circle((0,0), 2, color="k")
    ax.add_patch(BH)
    ax.plot(x, y, "r-")
    ax.set_xlim(limits[1:2]...)
    ax.set_ylim(limits[3:4]...)
    ax.set_xlabel(L"$x$")
    ax.set_ylabel(L"$y$")
    ax.set_aspect("equal")
    plt.show()
    plt.close()
end

# Initial conditions
r₀ = 6
u₀ = 0
φ₀ = 0
t₀ = 0
s₀ = [r₀, u₀, φ₀, t₀]

# Angular momentum and energy
ℓ = 0
ε = √(u₀^2 + Veff(r₀, ℓ))

τ, t, r, φ = integrate_particle_orbit_gr(s₀, ℓ, ε)
t_good = r .> 2

println("Proper time to fall to r=0: τ = $(τ[end])")

# Plot r as a function of τ and t
fig, ax = plt.subplots()
ax.plot(τ, r, "k-")
ax.plot(t[t_good], r[t_good], "k-")
ax.plot(t[t_good], 2 .* ones(sum(t_good)), "k--")
ax.set_xlim(0, 30)
ax.set_ylim(0)
ax.set_xlabel(L"$t$ (Normalized units)")
ax.set_ylabel(L"$r$ (Normalized units)")
plt.show()
plt.close()

# Veff plot for ℓ = 5
ℓ = 5
r = 0:0.001:50
V = Veff.(r, ℓ)
# Find max and min
rmax = (ℓ^2 - ℓ * √(ℓ^2 - 12))/2
rmin = (ℓ^2 + ℓ * √(ℓ^2 - 12))/2
Vmin = Veff(rmin, ℓ)
Vmax = Veff(rmax, ℓ)
println("Effective potential peak height V=$Vmax at r=$rmax")
println("Effective potential minimum V=$Vmin at r=$rmin")

fig, ax = plt.subplots()
ax.plot(r, V, "k-")
ax.plot(r, Vmax .* ones(length(r)), "k--")
ax.plot(r, Vmin .* ones(length(r)), "k--")
ax.set_ylim(0.5, 1.6)
ax.set_xlabel(L"$r$")
ax.set_ylabel(L"$V_{\rm eff}$")
plt.show()
plt.close()

################## PART E #####################

# Choose large r to represent coming in from infinity
r₀ = 100
# Choose energy slightly smaller than sqrt(Vmax)
ε = √(Vmax - 1e-3)
# Initial u is constrained by energy
u₀ = -√(ε^2 - Veff(r₀, ℓ))
# Integrate
τ, t, x, y = integrate_particle_orbit_gr([r₀, u₀, 0, 0], ℓ, ε, cartesian=true)
# Calculate closest approach
rclose = minimum(sqrt.(x.^2 .+ y.^2))
println("Closest approach r=$rclose")
# Plot
plot_trajectory(x, y)

################## PART F #####################

r₀ = 100
ε = √(Vmax + 1e-3)
u₀ = -√(ε^2 - Veff(r₀, ℓ))
τ, t, x, y = integrate_particle_orbit_gr([r₀, u₀, 0, 0], ℓ, ε, cartesian=true)
plot_trajectory(x, y)

################## PART G #####################

# energy slightly larger than Vmin
ε = √(Vmin + 1e-2)
r₀ = rmin
u₀ = √abs(ε^2 - Veff(r₀, ℓ))
τ, t, x, y = integrate_particle_orbit_gr([r₀, u₀, 0, 0], ℓ, ε, cartesian=true, n_orbits=10)
rfar = maximum(sqrt.(x.^2 .+ y.^2))
plot_trajectory(x, y, limits=(-rfar-1, rfar+1, -rfar-1, rfar+1))

################## PART H #####################

ε = √Vmin
r₀ = rmin
u₀ = 0
τ, t, x, y = integrate_particle_orbit_gr([r₀, u₀, 0, 0], ℓ, ε, cartesian=true, n_orbits=10,
                                        atol=1e-12, rtol=1e-8)
plot_trajectory(x, y, limits=(-r₀-1, r₀+1, -r₀-1, r₀+1))

################## PART I #####################

# Unstable cirular orbit: exactly at Vmax
ε = √Vmax
# Small perturbation
r₀ = rmax - 1e-5
u₀ = 0
τ, t, x, y = integrate_particle_orbit_gr([r₀, u₀, 0, 0], ℓ, ε, cartesian=true, n_orbits=10,
                                         atol=1e-12, rtol=1e-8)
plot_trajectory(x, y, limits=(-r₀-1, r₀+1, -r₀-1, r₀+1))

