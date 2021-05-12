#---------------------------------------------------------------
# Use spin-weighted spherical harmonics to compute the 
# Laplace operator using FastSphericalHarmonics.jl
# Soham M 03/21
#---------------------------------------------------------------

using FastSphericalHarmonics, LinearAlgebra, ForwardDiff
export gramschmidt, evaluate, jacobian, transform
export cartesian2spherical, spherical2cartesian
export raise, lower

function Base. map(u::Function, lmax::Int)
   N = lmax + 1
   θ, ϕ = sph_points(N) 
   return [u(θ, ϕ) for θ in θ, ϕ in ϕ]
end

function quad(F⁰::Array{T,2}, lmax::Int) where {T}
    C⁰ = spinsph_transform(F⁰,0) 
    return 4π*C⁰[sph_mode(0,0)]
end

function LinearAlgebra.dot(u::Array{T,2}, v::Array{T,2}, lmax::Int) where {T} 
    return quad(u.*v, lmax)
end

function evaluate(u::Array{T,2}, lmax::Int) where {T}
    u1 = spinsph_evaluate(reshape(u[:,1], lmax+1, 2lmax+1), 0) 
    u2 = spinsph_evaluate(reshape(u[:,2], lmax+1, 2lmax+1), 0) 
    u3 = spinsph_evaluate(reshape(u[:,3], lmax+1, 2lmax+1), 0) 
    return (u1, u2, u3)
end

function gramschmidt(u1::Array{T,2}, u2::Array{T,2}, u3::Array{T,2}, lmax::Int) where {T}
    # Orthonormalize
    u1 = u1 ./ sqrt(dot(u1, u1, lmax))
    u2 = u2 - dot(u1, u2, lmax) .* u1
    u2 = u2 ./ sqrt(dot(u2, u2, lmax))
    u3 = u3 - dot(u1, u3, lmax) .* u1 - dot(u2, u3, lmax) .* u2
    u3 = u3 ./ sqrt(dot(u3, u3, lmax))
    return (u1, u2, u3)
end

function isdiagonal(x::Matrix{T}) where {T}
    return istril(x) && istriu(x)
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

function cartesian2spherical(x::Vector{T}) where {T<:Real}
    return [cartesian2spherical(x...)...]  
end

function jacobian_cartesian2spherical(μ::T, ν::T) where {T<:Real}
    return ForwardDiff.jacobian(cartesian2spherical, [0.0,μ,ν])
end

function jacobian(x::Matrix{T}, y::Matrix{T}, z::Matrix{T}, lmax::Int) where {T<:Real}
    dx = grad(spinsph_transform(x, 0), lmax) 
    dy = grad(spinsph_transform(y, 0), lmax) 
    dz = grad(spinsph_transform(z, 0), lmax) 
    J1 = map(jacobian_cartesian2spherical, lmax)

    J = Matrix{SMatrix{2,2,Float64,4}}(undef, size(dx)...) 
    for index in CartesianIndices(dx)
        J0 = SMatrix{3,3}([0.0 dx[index][1] dx[index][2]; 
                           0.0 dy[index][1] dy[index][2]; 
                           0.0 dz[index][1] dz[index][2]])
        J2 = J1[index] * J0  
        @show typeof(J2)
        J[index] = J2[2:end, 2:end]
    end
    return J
end

function transform(h::Matrix{SMatrix{2,2,T}}, J::Matrix{SMatrix{2,2,T}}) where {T<:Real}
    return J .* h .* J′ 
end

function raise(qinv::Matrix{T}, x::Vector{T}) where {T<:Real}
    return qinv * x
end

function lower(q::Matrix{T}, x::Vector{T}) where {T<:Real}
    return q * x
end
