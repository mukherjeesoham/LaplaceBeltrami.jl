#---------------------------------------------------------------
# Spherical to Cartesian Coordinate transformations
# Soham M 11/21
#---------------------------------------------------------------

using ForwardDiff
export cartesian2spherical, spherical2cartesian

function wrapphi(x)
    (x < 0) ? (return (2π + x)) : (return x)
end

# function cartesian2spherical(x::T, y::T, z::T) where {T <: Real} 
function cartesian2spherical(x, y, z) 
    r = sqrt.(x.^2 + y.^2 + z.^2)
    θ = acos.(z ./ r)
    ϕ = map(wrapphi, atan.(y, x))
    return (r, θ, ϕ)
end

# function spherical2cartesian(r::T, θ::T, ϕ::T) where {T <: Real} 
function spherical2cartesian(r, θ, ϕ) 
    x = r .* cos.(ϕ) .* sin.(θ)
    y = r .* sin.(ϕ) .* sin.(θ)
    z = r .* cos.(θ)
    return (x, y, z)
end

function normalize(x::T, y::T, z::T) where {T<:Real}
    r = sqrt.(x^2 + y^2 + z^2)
    return (x, y, z) ./ r
end

function jacobian(u::Function, x::Vector) where {T<:Real}
    ux(x::Vector) = SVector{3}(u(x...))
    return ForwardDiff.jacobian(ux, x)
end

# function q(r::T, μ::T, ν::T) where {T <: Real}
function q(r, μ, ν) 
    return SMatrix{3,3}([1.0 0.0 0.0;
                         0.0 r^2 0.0; 
                         0.0 0.0  r^2 * sin(μ)^2])
end

# function n(r::T, μ::T, ν::T) where {T <: Real}
function n(r, μ, ν)
    return SVector{3}([1.0,0.0,0.0])
end

function q(μ::T, ν::T) where {T<:Real}
    g = q(1.0, μ, ν)
    s = n(1.0, μ, ν)
    return SMatrix{2,2}([g[a,b] - s[a]*s[b] for a in 2:3, b in 2:3])
end
