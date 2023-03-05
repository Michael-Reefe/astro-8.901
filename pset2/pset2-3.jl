using PyPlot
using Roots
using LaTeXStrings

plt.switch_backend("tkagg")

function energy(s)
    α = 110.48
    j = 1.25
    -1/s + α*(j-√s)^2
end

s = 0:0.01:2
ε = energy.(s)

maxval = argmax(ε)
minval = argmin(ε[s .> 0.25])

plt.plot(s, ε, "k-")
plt.axvline(s[maxval], color="r", linestyle="--")
plt.axvline(s[s .> 0.25][minval], color="r", linestyle="--")
plt.xlabel(L"s")
plt.ylabel(L"\varepsilon")
plt.ylim(bottom=-2.)
plt.show()

function rootEnergy(s)
    α = 110.48
    j = 1.25
    1 - α*s^1.5*(j-s^0.5)
end

z1 = find_zero(rootEnergy, (0., 0.05))
z2 = find_zero(rootEnergy, (0.05, 2.))

println("Zeros located at $z1 and $z2")

