#---------------------------------------------------------------
# Use spin-weighted spherical harmonics to compute the 
# Laplace operator using FastSphericalHarmonics.jl
# Soham M 03/21
#---------------------------------------------------------------

using Test, FastSphericalHarmonics, Arpack, ForwardDiff, CairoMakie, LinearAlgebra

# FIXME: Why does the gradient become grossly inaccurate for lmax = 60?
lmax = 26 
qmetric = map(q, lmax)   
hmetric = map(h, lmax)   

A  = Laplace{Float64}(lmax, qmetric, hmetric)
λ, w = eigs(A; nev=60, which=:LR)
@show λ

if false
    # Check individual components of the Laplace operator by stepping 
    # through the functions
    gamma(x::Vector)  = sYlm(Real, 0, 2, 2, theta(x...), x[2]) 
    dgamma(x::Vector) = ForwardDiff.gradient(gamma, x) 
    
    # Gradient with s = -1 spin weighted spherical harmonics
    Γ  = map((μ,ν)->gamma([μ,ν]),  lmax)
    ∇Γ = map((μ,ν)->dgamma([μ,ν]), lmax)
    dΓ = grad(spinsph_transform(Γ, 0), lmax) 
    @show ∇Γ[3,2]
    @show dΓ[3,2]
    @show maximum(∇Γ - dΓ)
    
    # Check the expansion of the divergence after the scaling 
    # We notice that the divergence after scaling doesn't fall off easily until 60 modes.
    # Could filtering help?
    SdΓ  = map(S1, qmetric, hmetric, dΓ)
    dSdΓ = div(SdΓ, lmax)
end 

# Test the divergence in isolation. First, check whether the integral of the divergence
# over the sphere is zero. This is guaranteed by the Divergence theorem. Next one can also put
# a purely divergence-free vector field on the sphere, and check if the divergence is indeed zero.
# These can be tested with the complicated metric in there. 
# TODO: Also check vector field decompositions on the sphere. 
