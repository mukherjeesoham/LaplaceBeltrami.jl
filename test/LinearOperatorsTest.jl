#-----------------------------------------------------
# Check spherical harmonics with finite differences 
# Soham 09/2021
#-----------------------------------------------------

using FastSphericalHarmonics, LinearAlgebra, Plots, StaticArrays, Test

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


lmin = 62
@testset "Laplace Spherical Harmonics" begin 
    for l in 1:6, m in -l:l
        U(μ,ν) =  sYlm(Real,0,l,m, theta(μ,ν), phi(μ,ν)) 
        (U1, dU1, ΔU1) = laplaceSH(U, lmin)
        @test maximum(abs.(ΔU1 + l * (l + 1) * U1))  < 1e-8
    end
end;
