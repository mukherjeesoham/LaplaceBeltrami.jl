#-----------------------------------------------------
# Implement the Linear Operators
# Soham 05/2021
#-----------------------------------------------------

using FastSphericalHarmonics, StaticArrays
export grad, laplace, Laplace, gradbar

function sqrtq(μ::T, ν::T) where {T<:Real}
    return sqrt.(q(μ, ν))
end

function grad(C⁰::AbstractMatrix{T}, lmax::Int) where {T <: Real}
    ðC⁰ = spinsph_eth(C⁰, 0)
    F¹  = spinsph_evaluate(ðC⁰, 1)
    M   = map(sqrtq, lmax)
    # FIXME: Including the sinθ term in the gradient messes with the Laplace
    # operator. While we need the sinθ for accurate gradients, piping the
    # output to the divergence operator seems to mess with things. 
    # REMOVED return M .* F¹ 
    return F¹
end

function gradbar(C⁰::AbstractMatrix{T}, lmax::Int) where {T <: Real}
    ðC⁰ = spinsph_eth(C⁰, 0)
    F¹  = spinsph_evaluate(ðC⁰, 1)
    M   = map(sqrtq, lmax)
    return -M .* F¹ 
end

function div(F¹::AbstractMatrix{SVector{2, T}}) where {T<:Real}
    C¹  = spinsph_transform(F¹, 1)  
    ð̄C¹ = spinsph_ethbar(C¹, 1)
    ∇²F = spinsph_evaluate(ð̄C¹, 0) 
    return ∇²F
end

function laplace(C⁰::AbstractMatrix{T}, lmax::Int) where {T<:Real}
    return spinsph_transform(div( grad(C⁰, lmax)), 0)
end

function S1(q::AbstractMatrix{T}, h::AbstractMatrix{T}, F¹::AbstractVector{T}) where {T<:Real}
    return sqrt(det(q) / det(h)) .* lower(q, raise(inv(h), F¹))
end

function S2(q::AbstractMatrix{T}, h::AbstractMatrix{T}, F⁰::AbstractVector{T}) where {T<:Real}
    return sqrt(det(h) / det(q)) .* x
end

function laplace(x::AbstractArray{Float64,1}, lmax::Int)::AbstractArray{Float64,1}
    return vec(laplace(reshape(x, lmax+1, 2lmax+1), lmax))
end

struct Laplace{T} <: AbstractMatrix{T}
    lmax::Int
end

function LinearAlgebra.mul!(y::AbstractVector, A::Laplace, x::AbstractVector)
    y .= laplace(x, A.lmax)
end

function Base.size(A::Laplace)
    size = (A.lmax + 1) * (2 * A.lmax + 1)
    return (size, size)
end

function LinearAlgebra.issymmetric(::Laplace)
    return true
end

Base.eltype(::Laplace{T}) where T = T

function LinearAlgebra.mul!(y::AbstractMatrix, A::Laplace, x::AbstractMatrix)
    @assert size(y,2) == size(x,2)
    for column in 1:size(y,2)
        mul!((@view y[:, column]), A, (@view x[:, column]))
    end
end

function LinearAlgebra.mul!(y::AbstractMatrix, A::Laplace, x::AbstractMatrix, α::Number, β::Number)
    @assert size(y,2) == size(x,2)
    ycopy = copy(y) 
    for column in 1:size(y,2)
        mul!((@view y[:, column]), A, (@view x[:, column]))
    end

    if α == β == 0
        y .= 0 
    elseif α == 0
        y .= β*ycopy
    elseif β == 0
        y .= α*y
    else
        y .= α*y + β*ycopy
    end
end
