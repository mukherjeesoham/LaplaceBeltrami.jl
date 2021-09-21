#-----------------------------------------------------
# Check spherical harmonics with finite differences 
# Soham 07/2021
#-----------------------------------------------------

using FastSphericalHarmonics, LinearAlgebra, Plots, StaticArrays, Test

# Laplace operator with Spherical Harmonics and spherical basis
function laplaceSH(u::Function, lmax::Int)
    U       = map(u, lmax)
    qmetric = map(q, lmax)
    hmetric = map(h, lmax)
    dU      = grad(spinsph_transform(U, 0), lmax) 
    SdU     = map(S1, qmetric, hmetric, dU) 
    dSdU    = div(SdU, lmax) 
    ΔU      = map(S2, qmetric, hmetric, dSdU) 
    return (U, dU, ΔU)
end

# Laplace operator with Spherical Harmonics and Cartesian basis
function laplaceCC(u::Function, lmax::Int)
    U       = map(u, lmax)
    qmetric = map((μ,ν)->cartesian(q,μ,ν), lmax)
    hmetric = map((μ,ν)->cartesian(h,μ,ν), lmax)
    dU      = grad(U, lmax,  :Cartesian) 
    SdU     = map(S1, qmetric, hmetric, dU) 
    dSdU    = div(SdU, lmax, :Cartesian) 
    ΔU      = map(S2, qmetric, hmetric, dSdU) 
    return (U, dU, ΔU)
end

# Laplace operator with finite differences and spherical basis
function laplaceFD(u::Function, ni::Int, nj::Int)
    U       = map(u, ni, nj) 
    qmetric = map(q, ni, nj)
    hmetric = map(h, ni, nj)
    dU      = grad(U, ni, nj)
    SdU     = map(S1FD, hmetric, dU) 
    dSdU    = div(SdU, ni, nj) 
    ΔU      = map(S2FD, hmetric, dSdU) 
    return (U, dU, ΔU)
end

function compare(u::Array{T,2}, v::Array{T,2}) where {T}
    return u - v[:, begin:2:end]
end

function Y42(μ, ν)
    return sYlm(Real,0,4,2,μ,ν)
end

# function dY22(μ, ν)
    # return SVector{2, Float64}([∂θsYlm(Real,0,2,2,μ,ν), ∂ϕsYlm(Real,0,2,2,μ,ν)])
# end

function Ȳ42(μ,ν)
    return Y42(theta(μ,ν), phi(μ,ν)) 
end

lmax = 302

for l in 1:40, m in -l:l
    U(μ,ν) =  sYlm(Real,0,l,m, theta(μ,ν), phi(μ,ν)) 
    (U1, dU1, ΔU1) = laplaceSH(U, lmax)
    @show l, m, maximum(abs.(ΔU1 + l * (l + 1) * U1)) 
end


# function dȲ22(μ,ν)
    # return dY22(theta(μ,ν), phi(μ,ν)) 
# end

# FIXME: Why is the gradient zero for lmax = 40, 60?  Also, does the solution
# get inaccurate for all big l's? No, just some even numbers apparently.
# l = 4
# lmax = 102
# ni,nj = npoints(lmax) 

# μ = map((μ,ν)->μ, lmax)
# ν = map((μ,ν)->ν, lmax)
# θ = map(theta, lmax)
# ϕ = map(phi, lmax)
# p0 = plot(contourf(θ - μ), contourf(ϕ - ν))
# savefig(p0, "plots/theta-phi-difference")

# # Test if either of the Laplace operator works for an eigenfunction 
# # for the round sphere coordinates.
# (U1, dU1, ΔU1) = laplaceSH(Ȳ42, lmax)
# (U2, dU2, ΔU2) = laplaceCC(Ȳ42, lmax)
# (U3, dU3, ΔU3) = laplaceFD(Ȳ42, ni, nj)
# @show maximum(ΔU1 + l * (l + 1) * U1)
# @show maximum(ΔU2 + l * (l + 1) * U2)
# @show maximum(ΔU3 + l * (l + 1) * U3)

# # Compare with FD solution
# # @show maximum(compare(U1, U3))
# # @show maximum(compare(U2, U3))


# # Compare solutions for gradient. Easiest would be to plot the solutions. 
# # CdU  = compare(dU1, dU2)
# # CdUθ = map(x->x[1], CdU) 
# # CdUϕ = map(x->x[2], CdU) 
# # dU1θ = map(x->x[1], dU1)
# # dU1ϕ = map(x->x[2], dU1)
# # dU2θ = map(x->x[1], dU2)
# # dU2ϕ = map(x->x[2], dU2)

# # p1 = plot(contourf(log10.(abs.(CdUθ))), contourf(log10.(abs.(CdUθ))))
# # p2 = plot(contourf(dU1θ), contourf(dU2θ))
# # savefig(p1, "plots/compare-grad-difference")
# # savefig(p2, "plots/compare-grad")

# # Now compare the solutions after the divergence. This is to fix
# # the FD divergence. The solution seems to differ near the poles.
# # Shall we do some filtering?
# # Could the divergence routine be wrong? Can we check the 
# # convergence for the vector derivative operators?
# # We can also try filtering, or analytically handling the 
# # sinθ terms in the expressions
# # @show maximum(compare(ΔU1, ΔU2))
# # p3 = plot(contourf(dSdU1), contourf(dSdU2))
# # p4 = plot(contourf(compare(dSdU1, dSdU2)))
# # savefig(p3, "plots/compare-div-analytic-gradient")
# # savefig(p4, "plots/compare-div-difference-analytic-gradient")

