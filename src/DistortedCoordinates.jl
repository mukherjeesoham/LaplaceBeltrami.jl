#---------------------------------------------------------------
# LaplaceOnASphere
# Soham 8/2020
# Choose a smooth coodinate transformation and compute 
# the associated metric
# See associated Mathematica notebooks MetricTransformation.nb
# and AnalyticFunctions.nb
#---------------------------------------------------------------

using ForwardDiff, StaticArrays 
export theta, q, h, Z

function Z(μ::T, ν::T)::T where {T<:Real}
    z = (1 / 80) * (53 * cos(μ) + 90 * cos(μ)^3 - 63 * cos(μ)^5) 
    return z
end

function theta(μ::T, ν::T)::T where {T<:Real}
    # Stretch the coordinates to make them ellipsoids
    (a, b, c) = (1,2,3)
    x = a*sin(μ)*cos(ν) 
    y = b*sin(μ)*sin(ν) 
    z = c*cos(μ) 
    # Now project it back onto a sphere
    r = x^2 + y^2 + z^2
    x = x/r
    y = y/r
    z = z/r
    return acos(z/sqrt(x^2 + y^2 + z^2))
end

function phi(μ::T, ν::T)::T where {T<:Real}
    # Stretch the coordinates to make them ellipsoids
    (a, b, c) = (1,2,3)
    x = a*sin(μ)*cos(ν) 
    y = b*sin(μ)*sin(ν) 
    z = c*cos(μ) 
    # Now project it back onto a sphere
    r = x^2 + y^2 + z^2
    x = x/r
    y = y/r
    z = z/r
    return atan(y, x)
end

function g(μ::T, ν::T) where {T}
    θ = theta(μ, ν) 
    return SMatrix{2,2}([1.0 0; 0 sin(θ)^2])
end

function q(μ::T, ν::T) where {T}
    return SMatrix{2,2}([1.0 0; 0 sin(μ)^2])
end

function θϕ_of_μν(x::Array{T,1}) where {T<:Real} 
    μ, ν = x
    return [theta(μ,ν), phi(μ,ν)] 
end

function jacobian(μ::T, ν::T) where {T<:Real}
    return ForwardDiff.jacobian(θϕ_of_μν, [μ,ν])
end

function h(μ::T, ν::T) where {T <: Real}
    J  = jacobian(μ,ν) 
    h  =  (J * g(μ,ν) * J')
    return SMatrix{2,2}(h)
end
