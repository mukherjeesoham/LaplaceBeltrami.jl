#-----------------------------------------------------
# Check spherical harmonics with finite differences 
# Soham 07/2021
#-----------------------------------------------------
# — Start with a function on the sphere.
# — Apply the Laplace operator in terms of FD
# — Do the same in terms of SH.
# — If the final output is different then
# — Compare whether the solutions agree at the gradient.
# — If yes, compare at the divergence.
# — How to compare? Compute the modes of the FD solution using SHTOOLS. Or
#   compare at the FD points throwing away half the points.
# — Do a convergence Y22 for both approaches.

using FastSphericalHarmonics, LinearAlgebra, Plots, StaticArrays

function laplaceSH(u::Function, lmax::Int)
    U       = map(u, lmax)
    qmetric = map(q, lmax)
    hmetric = map(h, lmax)
    Ulm     = spinsph_transform(U, 0) 
    dU      = grad(Ulm, lmax) 
    SdU     = map(S1, qmetric, qmetric, dU) 
    @assert all(maximum(dU - SdU) .< 1e-12)
    dSdU    = div(SdU, lmax, "Cartesian") 
    ΔU      = map(S2, qmetric, qmetric, dSdU) 
    @assert maximum(dSdU - ΔU) < 1e-12
    return (U, dU, dSdU, ΔU)
end

function laplaceFD(u::Function, ni::Int, nj::Int)
    U       = map(u, ni, nj) 
    qmetric = map(q, ni, nj)
    hmetric = map(h, ni, nj)
    dU      = grad(U, ni, nj)

    # Introduce analytic gradient here. 
    dU      = map(dY22, ni, nj) 
    vsin2θ = map((μ,ν)->SVector{2}(1, sin(μ)^2), ni, nj)
    sinθ = map((μ,ν)->SVector{2}(sin(μ), sin(μ)), ni, nj)
    onesinθ = map((μ,ν)->sin(μ), ni, nj)
    # @show typeof(vsin2θ)
    # @show typeof(dU)
    # @show typeof(sinθ)
    # @show size(vsin2θ)
    # @show size(dU)
    # @show size(sinθ)
        
    # SdU = sinθ .* (dU ./ (vsin2θ))
    SdU = map(.*, map( ./, dU, vsin2θ), sinθ)

    # SdU     = map(S1, qmetric, qmetric, dU) 
    # @assert all(maximum(dU - SdU) .< 1e-12)
    dSdU    = div(SdU, ni, nj) 
    ΔU = (1 ./ onesinθ) .* dSdU

    # ΔU      = map(S2, qmetric, qmetric, dSdU) 
    # @assert maximum(dSdU - ΔU) < 1e-12
    return (U, dU, dSdU, ΔU)
end

function compare(u::Array{T,2}, v::Array{T,2}) where {T}
    return u - v[:, begin:2:end]
end

function Y22(μ, ν)
    return sYlm(Real,0,2,2,μ,ν)
end

function dY22(μ, ν)
    return SVector{2, Float64}([∂θsYlm(Real,0,2,2,μ,ν), ∂ϕsYlm(Real,0,2,2,μ,ν)])
end

# FIXME: Convergence orders are acting funny. We're getting 
# 2nd order convergence for a 4th order method. 
# Check convergence for Dθ [Why do we get 2nd order convergence?]
function convergence(order::Int)
    L2θ = zeros(9)
    L2ϕ = zeros(9)

    for index in CartesianIndices(L2θ) 
        n   = index.I[1]
        ni  = 2^n
        nj  = 2ni
        Dθ, Dϕ = dscalar(ni, nj, order)
        u   = map((μ,ν)->cos(μ), ni, nj)
        du  = map((μ,ν)->-sin(μ), ni, nj)
        ndu = reshape(Dθ * vec(u), ni, nj)
        L2θ[index] = norm(du - ndu)
    end

    for index in CartesianIndices(L2ϕ) 
        n   = index.I[1]
        ni  = 2^n
        nj  = 2ni
        Dθ, Dϕ = dscalar(ni, nj, order)
        u   = map((μ,ν)->sin(μ)*cos(ν), ni, nj)
        du  = map((μ,ν)->-sin(μ)*sin(ν), ni, nj)
        ndu = reshape(Dϕ * vec(u), ni, nj)
        L2ϕ[index] = norm(du - ndu)
    end

    o4 = [-4*x for x in 1:9]
    o2 = [-2*x for x in 1:9]
    plot(log.(L2θ), label="numeric theta")
    plot!(log.(L2ϕ), label="numeric phi")
    plot!(o2, label="2th order")
    plot!(o4, label="4th order")
end

# FIXME: Why is the gradient zero for lmax = 40, 60? 
# Also, does the solution get inaccurate for all big l's? No, just some even
# numbers apparently.
l = 2
lmax = 82
ni,nj = npoints(lmax) 

# Test if either of the Laplace operator works for an eigenfunction 
# for the round sphere coordinates.
(U1, dU1, dSdU1, ΔU1) = laplaceSH(Y22, lmax)
@show maximum(ΔU1 + l * (l + 1) * U1)

(U2, dU2, dSdU2, ΔU2) = laplaceFD(Y22, ni, nj)
@show maximum(ΔU2 + l * (l + 1) * U2)
@show maximum(compare(U1, U2))
@show maximum(compare(dU1, dU2))

# Compare solutions for gradient. Easiest would be to plot the solutions. 
CdU  = compare(dU1, dU2)
CdUθ = map(x->x[1], CdU) 
CdUϕ = map(x->x[2], CdU) 
dU1θ = map(x->x[1], dU1)
dU1ϕ = map(x->x[2], dU1)
dU2θ = map(x->x[1], dU2)
dU2ϕ = map(x->x[2], dU2)

p1 = plot(contourf(log10.(abs.(CdUθ))), contourf(log10.(abs.(CdUθ))))
p2 = plot(contourf(dU1θ), contourf(dU2θ))
savefig(p1, "plots/compare-grad-difference")
savefig(p2, "plots/compare-grad")

# Now compare the solutions after the divergence. This is to fix
# the FD divergence. The solution seems to differ near the poles.
# Shall we do some filtering?
# Could the divergence routine be wrong? Can we check the 
# convergence for the vector derivative operators?
# We can also try filtering, or analytically handling the 
# sinθ terms in the expressions
@show maximum(compare(ΔU1, ΔU2))
p3 = plot(contourf(dSdU1), contourf(dSdU2))
p4 = plot(contourf(compare(dSdU1, dSdU2)))
savefig(p3, "plots/compare-div-analytic-gradient")
savefig(p4, "plots/compare-div-difference-analytic-gradient")

function laplaceFDregular(U::Function, ni::Int, nj::Int) where {T<:Real}
    F       = map(U, ni, nj) 
    Dθ, Dϕ  = dscalar(ni, nj, 4)
    Dθ̄, Dϕ̄  = dvector(ni, nj, 4) 
    sinθ    = map((μ,ν)->sin(μ), ni, nj)
    cotθ    = map((μ,ν)->cot(μ), ni, nj)
    csc2θ   = map((μ,ν)->csc(μ)^2, ni, nj)
    divbarF1 = (vec(cotθ) .* (Dθ * vec(F)) .+ Dθ̄ * (Dθ * vec(F)))
    divbarF2 = - vec(cotθ .* csc2θ) .* (Dϕ * vec(F)) .+ vec(csc2θ) .* (Dϕ̄ * (Dϕ * vec(F))) 
    return reshape(F, ni, nj), reshape(divbarF1 + divbarF2, ni, nj)
end

# Is it better with regularized variables? Seems worse! Or is this is a bug!
# UR, ΔUR = laplaceFDregular(Y22, ni, nj) 
# @show maximum(ΔUR + l * (l + 1) * UR)

# TODO: Test the divergence with the analytic gradients. 
# XXX: This is strange. Why is the error larger with the accurate gradient? How
# can I understand this?
dUSH = map(dY22, lmax) 
dUFD = map(dY22, ni, nj) 
@show maximum(dUSH - dU1)
@show maximum(dUFD - dU2)
