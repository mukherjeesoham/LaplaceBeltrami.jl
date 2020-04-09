#---------------------------------------------------------------
# LaplaceOnASphere
# Soham 4/20
# Given a coordinate transformation x -> X comput g(X)
#---------------------------------------------------------------

export invhab, sqrtdeth, invsqrtdeth

function μ(θ::T, ϕ::T) where {T}
    return ScalarSPH(3,1,θ,ϕ)
end

function ν(θ::T, ϕ::T) where {T}
    return ScalarSPH(2,2,θ,ϕ)
end

function computeJacobian(S::SphericalHarmonics{T}, θ::T, ϕ::T)::Array{Complex{T},2} where {T}
    X = map(S, (θ,ϕ)->μ(θ,ϕ)) 
    Y = map(S, (θ,ϕ)->ν(θ,ϕ)) 
    L = nodal_to_modal_scalar_op(S) 
    (Xlm, Ylm) = (L*X, L*Y)
    jacobian = zeros(Complex{T}, (2,2))
    for l in 0:S.lmax, m in -l:l
        Ψ = VectorSPH(l, m, θ, ϕ)
        jacobian[1,1] += Xlm[join(l,m)]*Ψ[1] 
        jacobian[1,2] += Xlm[join(l,m)]*Ψ[2]*sin(θ)
        jacobian[2,1] += Ylm[join(l,m)]*Ψ[1] 
        jacobian[2,2] += Ylm[join(l,m)]*Ψ[2]*sin(θ)
    end
    return jacobian
end

function invqab(S::SphericalHarmonics{T}, a::Int, b::Int, θ::T, ϕ::T)::T where {T}
    if a == b == 1
        return 1
    elseif a == b == 2
        return 1/sin(θ)^2
    else
        return 0
    end
end

function qab(S::SphericalHarmonics{T}, a::Int, b::Int, θ::T, ϕ::T)::T where {T}
    if a == b == 1
        return 1
    elseif a == b == 2
        return sin(θ)^2
    else
        return 0
    end
end

function delta(a::Int, b::Int)::Int
    if a == b 
        return 1
    else
        return 0
    end
end

#---------------------------------------------------------------
# Construct g^ab
#---------------------------------------------------------------
function invgab(S::SphericalHarmonics{T}, a::Int, b::Int, θ::T, ϕ::T)::Complex{T} where {T}
    invgab = T(0)
    J   = computeJacobian(S, θ, ϕ) 
    for m in 1:2, n in 1:2
        invgab += J[a, m]*J[b, n]*invqab(S, m, n, θ, ϕ) 
    end
    return invgab
end

#---------------------------------------------------------------
# Construct h^ab
#---------------------------------------------------------------
function invhab(S::SphericalHarmonics{T}, a::Int, b::Int, θ::T, ϕ::T)::Complex{T} where {T}
    invhab = T(0)
    for c in 1:2, d in 1:2
        invhab += delta(a,c)*qab(S,c,d,θ,ϕ)*invgab(S,d,b,θ,ϕ)
    end
    return invhab
end

function invsqrtdeth(S::SphericalHarmonics, θ::T, ϕ::T)::Complex{T} where {T}
    invdeth = [invhab(S, 1, 1, θ, ϕ) invhab(S, 1, 2, θ, ϕ); 
               invhab(S, 2, 1, θ, ϕ) invhab(S, 2, 2, θ, ϕ)]
    return sqrt(det(invdeth))
end

function sqrtdeth(S::SphericalHarmonics, θ::T, ϕ::T)::Complex{T} where {T}
    return 1/invsqrtdeth(S, θ, ϕ)
end
