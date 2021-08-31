#---------------------------------------------------------------
# LaplaceOnASphere
# Soham 8/2020
# Choose a smooth coodinate transformation and compute 
# the associated metric
#---------------------------------------------------------------

using ForwardDiff 
export theta, q, h

function Z(μ::T, ν::T)::T where {T<:Real}
    z = (1 / 80) * (53 * cos(μ) + 90 * cos(μ)^3 - 63 * cos(μ)^5) 
    return z
end

function theta(μ::T, ν::T)::T where {T<:Real}
    x = sin(μ)*cos(ν) 
    y = sin(μ)*sin(ν) 
    z = Z(μ,ν)
    # Renormalize the coordinates
    r = sqrt.(x^2 + y^2 + z^2)
    x, y, z = (x, y, z) ./ r
    return acos(z/sqrt(x^2 + y^2 + z^2))
end

function phi(μ::T, ν::T)::T where {T<:Real}
    x = sin(μ)*cos(ν) 
    y = sin(μ)*sin(ν) 
    z = Z(μ,ν)
    # Renormalize the coordinates
    r = sqrt.(x^2 + y^2 + z^2)
    x, y, z = (x, y, z) ./ r
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
    return [theta(μ,ν), ν] 
end

function jacobian(μ::T, ν::T) where {T<:Real}
    return ForwardDiff.jacobian(θϕ_of_μν, [μ,ν])
end

function h(μ::T, ν::T) where {T <: Real}
    J  = jacobian(μ,ν) 
    h  =  (J * g(μ,ν) * J')
    return SMatrix{2,2}(h)
end

