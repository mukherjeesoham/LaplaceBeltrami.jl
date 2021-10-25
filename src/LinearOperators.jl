#-----------------------------------------------------
# Implement the Linear Operators using spin-weighted spherical harmonics.
# Soham 05/2021
#-----------------------------------------------------

using FastSphericalHarmonics, StaticArrays, Plots, Dates
export grad, laplace, Laplace, S1, S2
export gradbar, divbar

struct Laplace{T} <: AbstractMatrix{T}
    lmax::Int
    q::Matrix{AbstractArray{T}}
    h::Matrix{AbstractArray{T}}
end

function matrix(μ::T, ν::T) where {T<:Real}
    return SMatrix{2,2}([-1.0 0.0; 0.0 sin(μ)])
end

function gradbar(C⁰::AbstractMatrix{T}, lmax::Int) where {T <: Real}
    ðC⁰ = spinsph_eth(C⁰, 0)
    F¹  = spinsph_evaluate(ðC⁰, 1)
    return F¹
end

function divbar(F¹::AbstractMatrix{SVector{2, T}}) where {T<:Real}
    C¹  = spinsph_transform(F¹, 1)  
    ð̄C¹ = spinsph_ethbar(C¹, 1)
    ∇²F = spinsph_evaluate(ð̄C¹, 0) 
    return ∇²F
end

function grad(C⁰::AbstractMatrix{T}, lmax::Int) where {T <: Real}
    ð̄C⁰ = spinsph_ethbar(C⁰, 0)
    F¹  = spinsph_evaluate(ð̄C⁰, -1)
    M   = map(matrix, lmax)
    return M .* F¹ 
end

function Base. div(F¹::AbstractMatrix{SVector{2, T}}, lmax::Int) where {T<:Real}
    M   = map(matrix, lmax)
    F¹  = SVector{2}.(inv.(M) .* F¹)
    C¹  = spinsph_transform(F¹, -1)  
    ðC¹ = spinsph_eth(C¹, -1)
    ∇²F = spinsph_evaluate(ðC¹, 0) 
    return ∇²F
end

function S1(q::AbstractMatrix{T}, h::AbstractMatrix{T}, F¹::AbstractVector{T}) where {T<:Real}
    return sqrt(det(h) / det(q)) .* lower(q, raise(inv(h), F¹))
end

function S2(q::AbstractMatrix{T}, h::AbstractMatrix{T}, F⁰::T) where {T<:Real}
    return sqrt(det(q) / det(h)) .* F⁰
end

function laplace(C⁰::AbstractMatrix{T}, A::Laplace) where {T<:Real}
    dU      = grad(C⁰, A.lmax) 
    SdU     = map(S1, A.q, A.h, dU) 
    dSdU    = div(SdU, A.lmax) 
    ΔU      = map(S2, A.q, A.h, dSdU) 
    return spinsph_transform(ΔU, 0) 
end

function laplace(x::AbstractArray{Float64,1}, A::Laplace)::AbstractArray{Float64,1}
    return vec(laplace(reshape(x, A.lmax + 1, 2*A.lmax + 1), A))
end

function LinearAlgebra.mul!(y::AbstractVector, A::Laplace, x::AbstractVector)
    y .= laplace(x, A)
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
