#---------------------------------------------------------------
# Test the eigenvalue solver using Arpack and IterativeSolvers 
# Soham M 09/21
#---------------------------------------------------------------

using Test, FastSphericalHarmonics, Arpack, LinearMaps, LinearAlgebra, IterativeSolvers, Plots

#---------------------------------------------------------------
# Test the eigensolver for round sphere coordinates
#---------------------------------------------------------------

if false
    println("==> Simple coordinates | Arpack")
    (l, lmin) = (1, 14)
    qmetric = map(q, lmin)   
    hmetric = map(h, lmin)   
    A       = Laplace{Float64}(lmin, qmetric, qmetric)
    U0      = map((μ,ν)->sYlm(Real,0,l,0,μ,ν), lmin)
    U0lm    = spinsph_transform(U0, 0)
    λ, w    = eigs(A; nev=9, which=:LR, v0=vec(U0lm))
    ΔUlm    = reshape(w[:,3], lmin + 1, 2lmin + 1)
    ΔU      = spinsph_evaluate(ΔUlm, 0)
    P1      = contourf(ΔU')
    # FIXME: We do not get all the eigenvectors corresponding to l = 1 
    @show λ[1:4]
    @test_broken isapprox(λ[2:4], [-2.0, -2.0, -2.0], atol=1e-8)
    @show maximum(abs.(laplace(ΔUlm, A) + l * (l + 1) * ΔUlm ))
    savefig("./plots/l2m0-simple-arpack")
end

#---------------------------------------------------------------
# Test the eigensolver for distorted coordinates
#---------------------------------------------------------------
# TODO: Remove the possibility of aliasing. Visualize the mode. 
#       These seem to grow with time. But do I really understand what's going on?
# FIXME: Fix the CC divergence. 
# FIXME: Why do 

if true
    print("Complicated coordinates | Arpack")
    (l,lmin) = (1, 13) 
    qmetric  = map(q, lmin)   
    hmetric  = map(h, lmin)   
    A        = Laplace{Float64}(lmin, qmetric, hmetric)
    U0       = map((μ,ν)->sYlm(Real,0,0,0,μ,ν), lmin) # The solver seems to work best starting from  l = 0, m = 0 as guess
    U0lm     = spinsph_transform(U0, 0)
    λ, w     = eigs(A; nev=9, which=:LR, v0=vec(U0lm), maxiter = 100)
    ΔUlm     = reshape(w[:,3], lmin + 1, 2lmin + 1)
    ΔU       = spinsph_evaluate(ΔUlm, 0)
    P2       = contourf(ΔU')
    @show λ
    @test_broken isapprox(λ[2:4], [-2.0, -2.0, -2.0], atol=1e-8)
    @show maximum(abs.(laplace(ΔUlm, A) + l * (l + 1) * ΔUlm ))
    savefig("./plots/l2m0-distorted-arpack")
end

if false
    print("Complicated coordinates | Iterative Solvers")
    # Use the power method from IterativeSolvers
    # FIXME: How do we get multiple eigenvalue/eigenvector pairs? Start with a perturbation?
    (l, lmin) = (1, 41)
    qmetric   = map(q, lmin)   
    hmetric   = map(h, lmin)   
    A         = Laplace{Float64}(lmin, qmetric, hmetric)
    U0        = map((μ,ν)->sYlm(Real,0,l,0,μ,ν), lmin)
    U0lm      = spinsph_transform(U0, 0)
    σ         = -l * (l + 1) 
    Lap       = LinearMap{Float64}((y, x) -> mul!(y, A, x), length(U0), ismutating = true, issymmetric=true)
    λ, w      = powm!(Lap, vec(U0lm), inverse = true, shift = σ + 0.1, tol = 1e-4, maxiter = 100, verbose=false)
    ΔUlm      = reshape(w, lmin + 1, 2lmin + 1)
    ΔU        = spinsph_evaluate(ΔUlm, 0)
    P3        = contourf(ΔU')

    @show λ
    @show maximum(abs.(laplace(ΔUlm, A) + l * (l + 1) * ΔUlm ))
    @show laplace(ΔUlm, A) + l * (l + 1) * ΔUlm
    savefig("./plots/l2-distorted-iterativesolvers")
end
    
