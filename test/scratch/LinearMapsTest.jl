using LinearMaps, LinearAlgebra, IterativeSolvers
σ = 1.0 + 1.3im
A = rand(ComplexF64, 10, 10)
F = lu(A - σ * I)
# Why is the linear map the way it is? Does it have anything to do with the power method?
Fmap = LinearMap{ComplexF64}((y, x) -> ldiv!(y, F, x), 10, ismutating = true)
λ, x = powm(Fmap, inverse = true, shift = σ, tol = 1e-4, maxiter = 200)
# @show λ
# @show eigvals(A - σ*I)

# Understand what ldiv! does and find eigenvalues of A when it's wrapped in a function
A = rand(3,3)
σ = eigvals(A)[2]

function dummy!(y, x) 
    y .= A*x
end

Fmap = LinearMap{ComplexF64}((y, x) -> dummy!(y, x), 3, ismutating = true)
λ, x = powm(Fmap, inverse = true, shift = σ, tol = 1e-4, maxiter = 200, verbose=true)
@show λ


