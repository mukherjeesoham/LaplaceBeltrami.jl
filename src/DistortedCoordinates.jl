#---------------------------------------------------------------
# LaplaceOnASphere
# Soham 8/2020
# Choose a smooth coodinate transformation and compute the associated metric
#---------------------------------------------------------------

using StaticArrays
export q, h, rθϕ_of_rμν

Random.seed!(42)
A = rand(3,3)
R = eigen(A + A').vectors

# Do a coordinate transformation in Cartesian coordinates to make
# sure it is smooth and you don't introduce coordinate singularities.
# function XYZ_of_xyz(x::T, y::T, z::T) where {T<:Real} 
function XYZ_of_xyz(x, y, z) 
    X, Y, Z = R * [x, y, z]
    X = X + 0.2
    Y = Y - 1
    Z = Z + 1
    # Should we normalize here? Or project out the distances at the end?
    return (X,Y,Z)
end

# (θ, ϕ) / (X, Y, Z) are the *good* coordinates (metric is diagonal)
# (μ, ν) / (x, y, z) are the *bad* coordinates  (metric is not diagonal)
# function rθϕ_of_rμν(r::T, μ::T, ν::T) where {T<:Real}
function rθϕ_of_rμν(r, μ, ν) 
    x, y, z = spherical2cartesian(r, μ, ν)
    X, Y, Z = XYZ_of_xyz(x, y, z)
    r, θ, ϕ = cartesian2spherical(X,Y,Z) 
    return (r,θ,ϕ)
end

# metric in good coordinates (as a function of bad coordinates)
# function p(r::T, μ::T, ν::T) where {T}
function p(r, μ, ν)
    r,θ,ϕ = rθϕ_of_rμν(r, μ, ν) 
    return SMatrix{3,3}([1.0 0.0 0.0; 
                         0.0 r^2 0.0; 
                         0.0 0.0 r^2 * sin(θ)^2])
end

# surface normal in good coordinates (as a function of bad coordinates)
# function m(r::T, μ::T, ν::T) where {T <: Real}
function m(r, μ, ν)
    return [1.0, 0.0, 0.0]
end

# Compute the metric in bad coordinates 
# function h(r::T, μ::T, ν::T) where {T <: Real}
function h(r, μ, ν)
    J  = jacobian(rθϕ_of_rμν, [r, μ, ν]) 
    return SMatrix{3,3}(J * p(r,μ,ν) * J')
end

# Compute the surface normal in bad coordinates 
# function k(r::T, μ::T, ν::T) where {T <: Real}
function k(r, μ, ν) 
    J  = jacobian(rθϕ_of_rμν, [r, μ, ν]) 
    return SVector{3}(J * m(r,μ,ν))
end

# Compute the pullback of the metric on the sphere at r = 1
# FIXME: Ensure this preserves the area of the sphere.
# function h(μ::T, ν::T) where {T<:Real}
function h(μ, ν) 
    g = h(1.0, μ, ν) #test  
    s = k(1.0, μ, ν) #test
    @assert dot(raise(inv(g), s), s) ≈ 1
    return SMatrix{2,2}([g[a,b] - s[a]*s[b] for a in 2:3, b in 2:3]) 
end

