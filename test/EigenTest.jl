#---------------------------------------------------------------
# LaplaceOnASphere
# Soham 3/20
#
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
#    Ȳ[^lm'', _kl] S(1/g)[^kl]  Y[^kl, _lm'] S(ll)[^lm'] Ψ̄[^lm'b, _ij] g[^ij] gab[^ij, _ab] Ψ[^ija, _lm] ϕ[^lm]
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

using LinearAlgebra, PyPlot

#---------------------------------------------------------------
# Functions needed for scaling
# Set up new coordinates 
# (θ̃, ϕ̃) = ((1 + θ cos(θ) + sin(θ)^2), 1)
#---------------------------------------------------------------

function ll(l::Int, m::Int)::Int
    return l*(l+1)
end

function invsqrtdetg(θ::T, ϕ::T)::T where {T}
    return 1/sqrtdetg(1, 1, θ, ϕ)
end

function gab(a::Int, b::Int, θ::T, ϕ::T)::T where {T}
    @assert (1 <= a <= 2) && (1 <= b <= 2)
    if a == 1 && b == 1
        # return (1 + (ϕ^2)*cos(ϕ)^2 + sin(ϕ)*(sin(ϕ) + ϕ*cos(θ)*(2 + ϕ*cos(θ)*sin(ϕ))) + ϕ*sin(2ϕ))
        # return 1 + θ*cos(θ) + sin(θ)^2
        return 1
    elseif a == 2 && b == 2
        # return ((1 + θ*cos(ϕ)*sin(θ))^2 + (sin(θ)^2)*((θ*cos(θ) + sin(θ))^2)*(sin(ϕ)^2))
        return 1
        # return 1/sin(θ)^2
    else
        # return ((θ*ϕ)*cos(ϕ)^2 + cos(ϕ)*(ϕ*csc(θ) + θ*sin(ϕ)) + sin(ϕ)*(csc(θ) + sin(θ) + cos(θ)*(θ + ϕ*(θ*cos(θ) + sin(θ))*sin(ϕ))))
        return 0
    end
end

function sqrtdetg(a::Int, b::Int, θ::T, ϕ::T)::T where {T}
    @assert (1 <= a <= 2) && (1 <= b <= 2)
    if a == b
        # return 1/(1 + ϕ*cos(θ)*sin(ϕ) - sin(θ)*(θ*cos(θ) + sin(θ))*sin(ϕ)^2 + cos(ϕ)*sin(θ)*(θ - ϕ*sin(θ)*sin(ϕ))^2)
        # return sqrt(abs(1/(1 + θ*cos(θ) + sin(θ)^2)))
        return 1
        # return abs(sin(θ))
    else
        return 0
    end
end

#---------------------------------------------------------------
# Construct the operator
#---------------------------------------------------------------

SH = SphericalHarmonics(20)
take_me_to_the_vector_modes = nodal_to_modal_vector_op(SH)
take_me_to_the_vector_nodes = modal_to_nodal_vector_op(SH)
take_me_to_the_scalar_nodes = modal_to_nodal_scalar_op(SH)
take_me_to_the_scalar_modes = nodal_to_modal_scalar_op(SH)
scale_with_gab = scaling_vector_op(SH, gab)
scale_with_sqrtdetg = scaling_vector_op(SH, sqrtdetg) 
scale_with_invsqrtdetg = scaling_scalar_op(SH, invsqrtdetg) 
scale_the_modes = modal_scaling_op(SH, ll)

Δ = take_me_to_the_scalar_modes*scale_with_invsqrtdetg*take_me_to_the_scalar_nodes*scale_the_modes*take_me_to_the_vector_modes*scale_with_sqrtdetg*scale_with_gab*take_me_to_the_vector_nodes 

#---------------------------------------------------------------
# Compute eigenvalues and eigenvectors of the operator
#---------------------------------------------------------------
F = eigen(Δ)

for k in 1:9
    contourf(SH, take_me_to_the_scalar_nodes*F.vectors[:, k])
    savefig("./output/eigenvector$k.pdf")
    close()
end

#---------------------------------------------------------------
# Test the operators on Spherical Harmonic functions 
#---------------------------------------------------------------
(l,m) = (6, 0)
u  = map(SH, (θ, ϕ)->ScalarSPH(l, m, θ, ϕ))
Δu = l*(l+1)*u 
@test_broken take_me_to_the_scalar_nodes*(Δ*(take_me_to_the_scalar_modes*u)) ≈ Δu

