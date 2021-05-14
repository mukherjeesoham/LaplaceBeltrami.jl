#---------------------------------------------------------------
# Use spin-weighted spherical harmonics to compute the 
# Laplace operator using FastSphericalHarmonics.jl
# Soham M 03/21
#---------------------------------------------------------------

using Test, FastSphericalHarmonics, Arpack

lmax = 27
l = rand(1:4)
m = rand(-l:l)
reYlm = map((μ, ν)->sYlm(Real,0,l,m,μ,ν), lmax) 
C⁰    = spinsph_transform(reYlm, 0)
ð̄ðC⁰  = laplace(C⁰, lmax) 
@test all(ð̄ðC⁰  .+ 1 .≈  -l * (l + 1) .* C⁰  .+ 1)

# Now test the Laplace operator for the eigenvalues
A  = Laplace{Float64}(lmax)
λ, u = eigs(A; nev=60, which=:LR)
@show λ[2:4]
ef = u[:,2:4] 

# Now compute the coordinates, do the transformation, and 
# check if you can recover a diagonal metric.
u1, u2, u3 = evaluate(ef, lmax)
x, y, z    = gramschmidt(u1, u2, u3, lmax) 
jac        = jacobian(x, y, z, lmax)

qinverse = map(inv ∘ q, lmax)   
hinverse = transform(qinverse, jac)
@test all(isdiagonal.(hinverse, 1e-10))
display(hinverse[1,1])

