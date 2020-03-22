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
# where S(f) is the diagonal operators for scaling
# and we've used the fact that ∇.(ϕ^lm Ψ_lm) = l(l+1) ϕ^lm Y_lm 
# See <R G Barrera et al 1985 Eur. J. Phys. 6 287> 
# for reference and introduction to VSH. 
# ASK: Why will it not work? Or will it? 
#---------------------------------------------------------------

using LaplaceOnASphere, Test, LinearAlgebra

S = SphericalHarmonics(2, 2)

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

# Now try a different metric
function g̃ab(θ, ϕ, a, b)
    if a == 1 && b == 1
        return (2θ*cos(θ) + 2*sin(θ))^2
    elseif a == 2 && b == 2
        return 1/sin(θ)^2
    else
        return 0
    end
end
        
function sqrtdetg̃(θ, ϕ)
    return 4*(1 + θ*cot(θ))^2
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


LΨlm = M2N_Ψlm(S)
LSgg = S_Ψlm(S, (θ, ϕ)->sqrtdetg(θ,ϕ)*gab(θ,ϕ,1,1), 
                (θ, ϕ)->sqrtdetg(θ,ϕ)*gab(θ,ϕ,2,2))
LSg  = S_Ylm(S, (θ, ϕ)->1/sqrtdetg(θ,ϕ))
LΨ̄lm = N2M_Ψlm(S)
LYlm = L_M2N_Ylm(S)
LȲlm = pinv(LYlm) 

Δlm = LȲlm*LSg*LYlm*LΨ̄lm*LSgg*LΨlm
F = eigen(Δlm)
@show real.(F.values)
@show imag.(F.values)
@show F.vectors[:, 1]
