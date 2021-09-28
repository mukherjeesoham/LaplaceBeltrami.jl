#---------------------------------------------------------------
# Use spin-weighted spherical harmonics to compute the 
# Laplace operator using FastSphericalHarmonics.jl
# Soham M 03/21
#---------------------------------------------------------------

using FastSphericalHarmonics, LinearAlgebra, ForwardDiff
export quad, gramschmidt, evaluate, jacobian, transform
export cartesian2spherical, spherical2cartesian, normalize
export raise, lower, isdiagonal

function Base. map(u::Function, lmax::Int)
   N = lmax + 1
   θ, ϕ = sph_points(N) 
   return [u(θ, ϕ) for θ in θ, ϕ in ϕ]
end

function quad(F⁰::Array{T,2}) where {T}
    C⁰ = spinsph_transform(F⁰,0) 
    return 4π*C⁰[sph_mode(0,0)]
end

function LinearAlgebra.dot(u::Array{T,2}, v::Array{T,2}) where {T} 
    return quad(u.*v)
end

function evaluate(u::Array{T,2}, lmax::Int) where {T}
    u1 = spinsph_evaluate(reshape(u[:,1], lmax+1, 2lmax+1), 0) 
    u2 = spinsph_evaluate(reshape(u[:,2], lmax+1, 2lmax+1), 0) 
    u3 = spinsph_evaluate(reshape(u[:,3], lmax+1, 2lmax+1), 0) 
    return (u1, u2, u3)
end

function gramschmidt(u1::Array{T,2}, u2::Array{T,2}, u3::Array{T,2}) where {T}
    # Orthonormalize
    u1 = u1 ./ sqrt(dot(u1, u1))
    u2 = u2 - dot(u1, u2) .* u1
    u2 = u2 ./ sqrt(dot(u2, u2))
    u3 = u3 - dot(u1, u3) .* u1 - dot(u2, u3) .* u2
    u3 = u3 ./ sqrt(dot(u3, u3))
    return (u1, u2, u3)
end

function wrap(x)
    (x < 0) ? (return (2π + x)) : (return x)
end

function cartesian2spherical(x::T, y::T, z::T) where {T <: Real} 
    r = sqrt.(x.^2 + y.^2 + z.^2)
    θ = acos.(z ./ r)
    ϕ = map(wrap, atan.(y, x))
    return (r, θ, ϕ)
end

function spherical2cartesian(r::T, θ::T, ϕ::T) where {T <: Real} 
    x = r .* cos.(ϕ) .* sin.(θ)
    y = r .* sin.(ϕ) .* sin.(θ)
    z = r .* cos.(θ)
    return (x, y, z)
end

function spherical2cartesian(θ::T, ϕ::T) where {T <: Real} 
    x = cos.(ϕ) .* sin.(θ)
    y = sin.(ϕ) .* sin.(θ)
    z = cos.(θ)
    return (x, y, z)
end

function normalize(x::T, y::T, z::T) where {T<:Real}
    r = sqrt.(x^2 + y^2 + z^2)
    return (x, y, z) ./ r
end

function cartesian2spherical(x::Vector{T}) where {T<:Real}
    return [cartesian2spherical(x...)...]  
end

function jacobian_cartesian2spherical(x::T, y::T, z::T) where {T<:Real}
    return ForwardDiff.jacobian(cartesian2spherical, [x,y,z])[2:end, :]
end

function jacobian(x::Matrix{T}, y::Matrix{T}, z::Matrix{T}, lmax::Int) where {T<:Real}
    dx = gradbar(spinsph_transform(x, 0), lmax) 
    dy = gradbar(spinsph_transform(y, 0), lmax) 
    dz = gradbar(spinsph_transform(z, 0), lmax) 
    J  = Matrix{SMatrix{2,2,Float64,4}}(undef, size(dx)...) 

    for index in CartesianIndices(dx)
        Jdθ′dx′ = jacobian_cartesian2spherical(x[index], y[index], z[index])
        Jdx′dθ  = [dx[index][1] dx[index][2]; dy[index][1] dy[index][2]; dz[index][1] dz[index][2]]
        J[index] = Jdθ′dx′ * Jdx′dθ
    end

    return J
end

function transform(h::Matrix{T}, J::Matrix{T}) where {T<:SMatrix{2,2}}
    return J .* h .* transpose.(J) 
end

function raise(qinv::AbstractMatrix{T}, x::AbstractVector{T}) where {T<:Real}
    return qinv * x
end

function lower(q::AbstractMatrix{T}, x::AbstractVector{T}) where {T<:Real}
    return q * x
end

function isdiagonal(x::AbstractMatrix{T}, tol::Float64) where {T<:Real}
    for index in CartesianIndices(x)
        i, j = index.I
        if i != j
            if x[index] >= tol
                return false
            end
        end
    end
    return true
end

