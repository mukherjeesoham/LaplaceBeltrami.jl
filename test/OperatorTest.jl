#---------------------------------------------------------------
# Construct the Laplace operator in local coordinates using 
# Spherical harmonics
# We have 1/|g| d/dx^i (|g| g^ij dϕ/dx^j) 
# As an operator we write the above expression as 
#    1\|g| d/dx^i (Ψ_lm inv(Ψ_lm) S(|g|) S(gij) Ψ_lm [ϕ^lm])
#                       |---------------------------------|  
#                                   ϕ′_lm
#    S(1\|g|) l(l+1) Y_lm ϕ′_lm 
#    S(1\|g|) l(l+1) Y_lm inv(Ψ_lm) S(|g|) S(gij) Ψ_lm [ϕ^lm]
#    Δ*ϕ_lm = l(l+1) Ylm*ϕ_lm = L(l)*𝐘*ϕ
#
#    q: beautiful round sphere metric
#
#    L^lm''_lm = inv(Y)_ij'^lm'' |h|^ij' Y^ij'_lm' l'(l'+1) inv(Ψ)_ijb^lm' |h|^ij h^ab^ij Ψ^ija_lm
#
#    |h| = |g| / |q|
#    inv(h) = q inv(g)
#    h^ab = δ^a_c q_cd g^db
#
# where S(f) is the diagonal operators for scaling
# and we've used the fact that ∇.(ϕ^lm Ψ_lm) = l(l+1) ϕ^lm Y_lm 
# See <R G Barrera et al 1985 Eur. J. Phys. 6 287> 
# for reference and introduction to VSH. 
#---------------------------------------------------------------
using LaplaceOnASphere, Test, LinearAlgebra

function gab(θ, ϕ, a, b)
    if a == 1 && b == 1
        return 1
    elseif a == 2 && b == 2
        return 1/sin(θ)^2
    else
        return 0
    end
end

function sqrtdetg(θ, ϕ)
    return sin(θ)
end

function L_M2N_Ylm(S)
    A = zeros(Complex, S.n*2(S.n), (S.l)^2 + 2*(S.l) + 1)
    for index in CartesianIndices(A)
        (i,j) = split(index.I[1], S.n)
        (l,m) = split(index.I[2])
        (θ,ϕ) = grid(i,j,S.n)
        A[index] = (l*(l+1))*Ylm(l, m, θ, ϕ)
    end
    return A
end

#---------------------------------------------------------------
# Construct Operators
#---------------------------------------------------------------

S = SphericalHarmonics(4, 8)

LΨlm  = M2N_Ψlm(S)
# LSgg  = S_Ψlm(S, (θ, ϕ)->sqrtdetg(θ,ϕ)*gab(θ,ϕ,1,1), 
                 # (θ, ϕ)->sqrtdetg(θ,ϕ)*gab(θ,ϕ,2,2))
# LSg   = S_Ylm(S, (θ, ϕ)->1/sqrtdetg(θ,ϕ))
LSgg  = S_Ψlm(S, (θ, ϕ)->1, 
                 (θ, ϕ)->1)

LSg   = S_Ylm(S, (θ, ϕ)->1)

LΨ̄lm  = N2M_Ψlm(S)
LlYlm = L_M2N_Ylm(S)
LỸlm  = N2M_Ylm(S)

Δ = LSg*LlYlm*LΨ̄lm*LSgg*LΨlm
Δlm = LỸlm*LSg*LlYlm*LΨ̄lm*LSgg*LΨlm

#---------------------------------------------------------------
# Construct functions for testing
#---------------------------------------------------------------

l  = 4
m  = 0
f  = map_to_grid(S, (θ, ϕ)->Ylm(l,m,θ,ϕ))
Δf = l*(l+1)*f 

flm  = N2M_Ylm(S)*f
Δflm = N2M_Ylm(S)*Δf

∇u = map_to_grid(S, (x,y)->-(1/2)*sqrt(3/π)*sin(x)^2, (x,y)->0)
∇ulm = N2M_Ψlm(S)*∇u 
∇ū = map_to_grid(S, (x,y)->analyticΨlm(S, ∇ulm, 1, x, y),
                    (x,y)->analyticΨlm(S, ∇ulm, 2, x, y))

#---------------------------------------------------------------
# Test using functions
# TODO: Test ∇u transformations [Done] => Ψ̄lm is working?
#       But why does it disagree with Mathematica?
# TODO: Test with analytical functions for 2 and 3 modes/points
# TODO: Make the operator using only indices
#---------------------------------------------------------------

@test_broken L1(Δ*flm - Δf) < 1e-12
@test_broken L1(Δlm*flm - Δflm) < 1e-12
# @test L1(real.(LSgg*LΨlm*flm) - ∇u) < 1e-12
# @test L1(∇u - ∇ū) < 1e-12

# @test L1(Δlm*flm - (LỸlm*LSg*LlYlm)*∇ulm) < 1e-12
# @show L1(pinv(LỸlm)*Δflm - (LSg*LlYlm)*∇ulm)
# @show L1(pinv(LỸlm*LSg*LlYlm)*Δflm - ∇ulm)

# display(abs.(Δlm*flm))
println("========================")
display(abs.(LỸlm*Δf))
