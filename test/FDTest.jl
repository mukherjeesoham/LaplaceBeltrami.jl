#---------------------------------------------------------------
# LaplaceOnASphere
# Soham 3/20
# Construct finite difference operators accurate to second order 
#---------------------------------------------------------------

using LinearAlgebra

N1 = 200
N2 = 100
Dθ = D2D1(Float64, N1, N2)
Dϕ = D2D2(Float64, N1, N2)
A  = S2D(Float64, N1, N2, (θ, ϕ)->sin(θ))

sinθcosϕ = [sin(X1(Float64, i, N1))*cos(X2(Float64, j, N2)) for i in 1:N1, j in 1:N2]
cosθcosϕ = [cos(X1(Float64, i, N1))*cos(X2(Float64, j, N2)) for i in 1:N1, j in 1:N2]
sinθsinϕ = [sin(X1(Float64, i, N1))*sin(X2(Float64, j, N2)) for i in 1:N1, j in 1:N2]
cosϕ     = [                        cos(X2(Float64, j, N2)) for i in 1:N1, j in 1:N2]

@show L1(Dθ*vec(sinθcosϕ) - vec(cosθcosϕ))
@show L1(Dϕ*vec(sinθcosϕ) + vec(sinθsinϕ))
@show L1(A*vec(cosϕ) - vec(sinθcosϕ))

