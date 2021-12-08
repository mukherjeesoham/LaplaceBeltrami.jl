#---------------------------------------------------------------
# Spherical to Cartesian Coordinate transformations
# Soham M 11/21
#---------------------------------------------------------------

using StaticArrays, ForwardDiff
export cartesian2spherical, spherical2cartesian
export q, n, jacobian

function wrapphi(x)
    (x < 0) ? (return (2π + x)) : (return x)
end

function cartesian2spherical(xyz::AbstractArray) 
    (x, y, z) = xyz
    r = sqrt.(x.^2 + y.^2 + z.^2)
    θ = acos.(z ./ r)
    ϕ = map(wrapphi, atan.(y, x))
    return SVector{3}([r, θ, ϕ])
end

function spherical2cartesian(rθϕ::AbstractArray) 
    (r, θ, ϕ) = rθϕ
    x = r .* cos.(ϕ) .* sin.(θ)
    y = r .* sin.(ϕ) .* sin.(θ)
    z = r .* cos.(θ)
    return SVector{3}([x, y, z])
end

function normalize(xyz::AbstractArray)
    (x, y, z) = xyz
    r = sqrt.(x^2 + y^2 + z^2)
    return (x, y, z) ./ r
end

function jacobian(u::Function, x::AbstractVector)
    return ForwardDiff.jacobian(u, x)
end

function q(rθϕ::AbstractArray) 
    (r, θ, ϕ) = rθϕ
    return SMatrix{3,3}([1.0 0.0 0.0;
                         0.0 r^2 0.0; 
                         0.0 0.0  r^2 * sin(θ)^2])
end

function n(rθϕ::AbstractArray) 
    return SVector{3}([1.0,0.0,0.0])
end

