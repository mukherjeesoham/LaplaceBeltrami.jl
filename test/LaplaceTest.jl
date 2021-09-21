#---------------------------------------------------------------
# Test the eigenvalue solver 
# Soham M 03/21
#---------------------------------------------------------------

using Test

lmax = 42 
qmetric = map(q, lmax)   
hmetric = map(h, lmax)   
A  = Laplace{Float64}(lmax, qmetric, qmetric)
λ, w = eigs(A; nev=60, which=:LR)
@test isapprox(λ[2:4], [-2.0, -2.0, -2.0], atol=1e-8)

