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
#---------------------------------------------------------------

function ll(l::Int, m::Int)::Int
    return l*(l+1)
end

function invsqrtdetg(θ::T, ϕ::T)::Complex{T} where {T}
    return 1/sqrtdetg(1, 1, θ, ϕ)
end

function invgab(a::Int, b::Int, θ::T, ϕ::T)::Complex{T} where {T}
    @assert (1 <= a <= 2) && (1 <= b <= 2)
    return invhab(SH, a, b, θ, ϕ)
    # if a == 1 && b == 1
        # return 1
    # elseif a == 2 && b == 2
        # return 1/sin(θ)^2
    # else
        # return 0
    # end
end

function sqrtdetg(a::Int, b::Int, θ::T, ϕ::T)::Complex{T} where {T}
    @assert (1 <= a <= 2) && (1 <= b <= 2)
    if a == b
        return sqrtdeth(SH, θ, ϕ)
        # return abs(sin(θ))
    else
        return 0
    end
end

#---------------------------------------------------------------
# Construct the operator
#---------------------------------------------------------------

SH = SphericalHarmonics{Float64}(4, 10)
take_me_to_the_vector_modes = nodal_to_modal_vector_op(SH)
take_me_to_the_vector_nodes = modal_to_nodal_vector_op(SH)
take_me_to_the_scalar_nodes = modal_to_nodal_scalar_op(SH)
take_me_to_the_scalar_modes = nodal_to_modal_scalar_op(SH)
scale_with_invgab = scaling_vector_op(SH, invgab)
scale_with_sqrtdetg = scaling_vector_op(SH, sqrtdetg) 
scale_with_invsqrtdetg = scaling_scalar_op(SH, invsqrtdetg) 
scale_the_modes = modal_scaling_op(SH, ll)

Δ = (take_me_to_the_scalar_modes*scale_with_invsqrtdetg*take_me_to_the_scalar_nodes*scale_the_modes*
      take_me_to_the_vector_modes*scale_with_sqrtdetg*scale_with_invgab*take_me_to_the_vector_nodes)

#---------------------------------------------------------------
# Compute eigenvalues and eigenvectors of the operator
#---------------------------------------------------------------
F = eigen(Δ)

# @assert maximum(imag.(F.values)) < 1e-10
eigenvalues = F.values
plot(real(eigenvalues[1:15]), "-o")
savefig("./eigenvalues.pdf")
close()

# for k in 1:9
    # contourf(SH, take_me_to_the_scalar_nodes*F.vectors[:, k])
    # savefig("./eigenvector$k.pdf")
    # close()
# end

