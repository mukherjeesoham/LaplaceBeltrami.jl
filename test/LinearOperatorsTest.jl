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

# XXX: There's a tradeoff between lmin and tol.
lmin = 72
tol  = 1e-5
@testset "Laplace Spherical Harmonics" begin 
    for l in 0:10, m in -l:l
        # Test the Laplace operator put together using pieces from
        # LinearOperators.jl
        U(μ,ν) =  sYlm(Real,0,l,m, θϕ_of_μν(μ,ν)...) 
        (U1, dU1, ΔU1) = laplaceSH(U, lmin)
        @test maximum(abs.(ΔU1 + l * (l + 1) * U1))  < tol

        # Now test the operator passed to the eigenvalue solver   
        qmetric = map(q, lmin)   
        hmetric = map(h, lmin)   
        A  = Laplace{Float64}(lmin, qmetric, hmetric)
        U = map((μ,ν)->sYlm(Real,0,l,m, θϕ_of_μν(μ,ν)...), lmin)
        Ulm = spinsph_transform(U, 0)
        ΔUlm = laplace(Ulm, A)
        ΔU = spinsph_evaluate(ΔUlm, 0) 
        @test maximum(abs.(ΔU + l * (l + 1) * U))  < tol

        if false
            # Now test the operator passed to the eigenvalue solver in
            # terms of LinearAlgebra.mul!
            qmetric = map(q, lmin)   
            hmetric = map(h, lmin)   
            A  = Laplace{Float64}(lmin, qmetric, hmetric)
            U = map((μ,ν)->sYlm(Real,0,l,m, θϕ_of_μν(μ,ν)...), lmin)
            Ulm = spinsph_transform(U, 0)
            # FIXME: Why doesn't this work? Check with Erik.
            ΔUlm = A * Ulm 
            ΔU = spinsph_evaluate(ΔUlm, 0) 
            @test maximum(abs.(ΔU + l * (l + 1) * U))  < 1e-7
        end
    end

end;

