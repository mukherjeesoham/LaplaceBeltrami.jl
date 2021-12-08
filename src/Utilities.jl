#---------------------------------------------------------------
# Use spin-weighted spherical harmonics to compute the 
# Laplace operator using FastSphericalHarmonics.jl
# Soham M 03/21
#---------------------------------------------------------------

using FastSphericalHarmonics, LinearAlgebra, DelimitedFiles, Random
export raise, lower, quad, gramschmidt, rotate

function Base. map(u::Function, lmax::Int)
   N = lmax + 1
   θ, ϕ = sph_points(N) 
   return SMatrix{N,N}([u([θ, ϕ]) for θ in θ, ϕ in ϕ])
end

function raise(qinv::AbstractMatrix{T}, x::AbstractVector{T}) where {T<:Real}
    return qinv * x
end

function lower(q::AbstractMatrix{T}, x::AbstractVector{T}) where {T<:Real}
    return q * x
end

function quad(F⁰::Array{T,2}) where {T}
    C⁰ = spinsph_transform(F⁰,0) 
    return 4π*C⁰[sph_mode(0,0)]
end

function rotate(x::AbstractVector{T})::AbstractVector{T} where {T <: Real} 
    Random.seed!(17)
    A = rand(3,3)
    R = eigen(A + A').vectors
    @assert eltype(R) <: Real
    @assert R' * R ≈ I
    return R * x 
end

function quad(F::Function, kind::Symbol)
    @assert kind == :sphericaldesigns 
    xyz = map(rotate, eachrow(readdlm("src/sf180.16382")))
    N   = first(size(xyz))
    return (4π / N) * sum(map(F, xyz))
end

function one(μ::T, ν::T) where {T<:Real}
    return T(1)
end

function zero(μ::T, ν::T) where {T<:Real}
    return T(0)
end

function LinearAlgebra.dot(u::Array{T,2}, v::Array{T,2}) where {T} 
    return quad(u.*v)
end

function gramschmidt(u1::Array{T,2}, u2::Array{T,2}, u3::Array{T,2}) where {T}
    u1 = u1 ./ sqrt(dot(u1, u1))
    u2 = u2 - dot(u1, u2) .* u1
    u2 = u2 ./ sqrt(dot(u2, u2))
    u3 = u3 - dot(u1, u3) .* u1 - dot(u2, u3) .* u2
    u3 = u3 ./ sqrt(dot(u3, u3))
    return (u1, u2, u3)
end

