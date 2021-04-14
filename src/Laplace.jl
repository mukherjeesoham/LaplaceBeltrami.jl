#---------------------------------------------------------------
# Use spin-weighted spherical harmonics to compute the 
# Laplace operator using FastSphericalHarmonics.jl
# Soham M 03/21
#---------------------------------------------------------------

using FastSphericalHarmonics, LinearAlgebra
export Laplace, laplace, C2Vec

function grad(C⁰::Array{T,2})::Array{T,2} where {T<:Complex}
    ðC¹ = spinsph_eth(C⁰, 0)
    ðF¹ = spinsph_evaluate(ðC¹, 1) 
    return ðF¹ 
end

function div(ðF¹::Array{T,2})::Array{T,2} where {T<:Complex}
    ðC¹  = spinsph_transform(ðF¹, 1) 
    ð̄ðC⁰ = spinsph_ethbar(ðC¹, 1)
    return ð̄ðC⁰ 
end

function laplace(C⁰::Array{T,2})::Array{T,2} where {T <: Complex}
    return (div ∘ grad)(C⁰) 
end

function laplace(x::AbstractArray{Float64,1}, lmax::Int)::AbstractArray{Float64,1}
    return C2Vec(laplace(Vec2C(x, lmax)))
end

function C2Vec(C⁰::AbstractArray{Complex{T},2})::AbstractArray{T,1} where {T}
    return vec(coeff_complex2real(C⁰, 0))
end

function Vec2C(V::AbstractArray{T,1}, lmax::Int)::AbstractArray{Complex{T},2} where {T}
    C⁰ = reshape(V, (lmax+1), (2lmax+1))
    return coeff_real2complex(C⁰, 0)
end

struct Laplace{T} <: AbstractMatrix{T}
    lmax::Int
end

function Base.size(A::Laplace)
    size = (A.lmax + 1) * (2 * A.lmax + 1)
    return (size, size)
end

function LinearAlgebra.issymmetric(::Laplace)
    return true
end

Base.eltype(::Laplace{T}) where T = T

function LinearAlgebra.mul!(y::AbstractVector, A::Laplace, x::AbstractVector)
    y .= laplace(x, A.lmax)
end

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

function coeff_complex2real(C::AbstractArray{Complex{Float64},2}, s::Int)
    @assert s == 0
    C′ = similar(C, Float64)
    C′[:, 1] = real.(C[:, 1])
    for col in 2:2:size(C′, 2)
        for row in 1:size(C′, 1)
            avg = (C[row, col] + conj(C[row, col + 1])) / sqrt(2)
            C′[row, col] = imag(avg)
            C′[row, col + 1] = real(avg)
        end
    end
    return C′
end

function coeff_real2complex(C::AbstractArray{Float64,2}, s::Int)
    @assert s == 0
    C′ = similar(C, Complex{Float64})
    C′[:, 1] = C[:, 1]
    for col in 2:2:size(C, 2)
        for row in 1:size(C, 1)
            val = Complex{Float64}(C[row, col + 1], C[row, col]) / sqrt(2)
            C′[row, col] = val
            C′[row, col + 1] = conj(val)
        end
    end
    return C′
end

