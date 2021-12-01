#-----------------------------------------------------
# Expand the Lapplace operator
# Soham 07/2021
#-----------------------------------------------------

using FastSphericalHarmonics, LinearAlgebra, Plots, StaticArrays, Test, ForwardDiff

# Laplace operator with Spherical Harmonics and Cartesian basis
# function laplaceCC(u::Function, lmax::Int)
    # U       = map(u, lmax)
    # qmetric = map((μ,ν)->cartesian(q,μ,ν), lmax)
    # hmetric = map((μ,ν)->cartesian(h,μ,ν), lmax) 
    # dU      = grad(U, lmax,  :Cartesian) 
    # SdU     = map(S4, qmetric, hmetric, dU)   
    # dSdU    = div(dU, lmax, :Cartesian) 
    # ΔU      = map(S5, qmetric, hmetric, dSdU) 
    # return (U, dU, ΔU)
# end

lmax = 12

function r(μ, ν)
    (x, y, z) = spherical2cartesian(1.0, μ, ν)
    return (x^3 + y^3 + z^3)
end

function ddr(μ, ν)
    (x, y, z) = spherical2cartesian(1.0, μ, ν)
    return 3*SVector{3}(x^2, y^2, z^2)
end

# u = map(r, lmax) 
# du_an = map(ddr, lmax)
# r(x::Vector) = r(x...)
# drad = map((μ, ν) -> ForwardDiff.gradient(r, [μ, ν]), lmax)
# du = grad(u, lmax, :Cartesian)
# display(du)
# display(du_an)

μ = map((μ, ν)->μ, lmax)
ν = map((μ, ν)->ν, lmax)
j = map(jacobian_xyz_of_rθϕ, lmax)

@show μ[4,7]
@show ν[4,7]
display(j[4,7])


# UP(μ,ν) =  sYlm(Real, 0, l, m, μ, ν)

# (U1, dU1, ΔU1) = laplaceCC(UP, lmin)
# @show maximum(abs.(ΔU1 + l * (l + 1) * U1)) 

# FIXME: Test the fucking gradient first. Then test the stupid scaling operator.
# Checking the divergence should be easier though.

