#-----------------------------------------------------
# Check spherical harmonics with finite differences
# and Cartesian derivatives
# Soham 07/2021
#-----------------------------------------------------

using FastSphericalHarmonics, LinearAlgebra, Plots, StaticArrays, Test

# Laplace operator with Spherical Harmonics and Cartesian basis
function laplaceCC(u::Function, lmax::Int)
    U       = map(u, lmax)
    qmetric = map((μ,ν)->cartesian(q,μ,ν), lmax)
    hmetric = map((μ,ν)->cartesian(h,μ,ν), lmax) # <=== Change frorm h to q here if you want round sphere coordinates
    dU      = grad(U, lmax,  :Cartesian) 
    # FIXME: This scaling might be different when you use Cartesian
    # derivatives. Check with FD
    SdU     = map(S4, qmetric, hmetric, dU) 
    dSdU    = div(SdU, lmax, :Cartesian) 
    ΔU      = map(S5, qmetric, hmetric, dSdU) 
    return (U, dU, ΔU)
end

lmin = 32
tol  = 1e-5
@testset "Laplace Cartesian Derivatives" begin 
    for l in 0:3, m in -l:l
        U(μ,ν) =  sYlm(Real,0,l,m, θϕ_of_μν(μ,ν)...) # <=== Change frorm h to q here if you want round sphere coordinates
        # U(μ,ν) =  sYlm(Real,0,l,m, μ,ν) 
        (U1, dU1, ΔU1) = laplaceCC(U, lmin)
        @test maximum(abs.(ΔU1 + l * (l + 1) * U1))  < tol
    end
end;
