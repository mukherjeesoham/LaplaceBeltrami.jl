#---------------------------------------------------------------
# LaplaceOnASphere
# Soham 8/2020
# Choose a smooth coodinate transformation and compute 
# the associated metric
#---------------------------------------------------------------

using ForwardDiff, Plots
export q, h, θϕ_of_μν 

function XYZ_of_xyz(x::T, y::T, z::T) where {T<:Real} 
    """
    Since coordinate transformations in x, y and z are smooth, 
    we only do coordinate transformations here. 
    """
    # FIXME: Any changes to x or y coordinates gives an assertion error
    # AssertionError: abs(imag(c)) ≤ sqrt(eps()) in 
    #  [1] coeff_complex2vector(C::Matrix{ComplexF64}, s::Int64)
     # @ FastSphericalHarmonics ~/.julia/packages/FastSphericalHarmonics/Dnkrz/src/spin.jl:67
    X = x 
    Y = y 
    Z = (1 / 80) * (53 * z + 90 * z^3 - 63 * z^5)
    return normalize(X,Y,Z)
end

function θϕ_of_μν(μ::T, ν::T) where {T<:Real}
    x, y, z = spherical2cartesian(μ, ν)
    X, Y, Z = XYZ_of_xyz(x, y, z)
    r, θ, ϕ = cartesian2spherical(X,Y,Z) 
    @assert r ≈ 1.0
    return (θ,ϕ)
end

function g(μ::T, ν::T) where {T}
    θ, ϕ = θϕ_of_μν(μ, ν) 
    return SMatrix{2,2}([1.0 0; 0 sin(θ)^2])
end

function q(μ::T, ν::T) where {T}
    return SMatrix{2,2}([1.0 0; 0 sin(μ)^2])
end

function θϕ_of_μν(μν::Array{T,1}) where {T<:Real} 
    return SVector{2}(θϕ_of_μν(μν...))
end

function jacobian(μ::T, ν::T) where {T<:Real}
    return ForwardDiff.jacobian(θϕ_of_μν, [μ,ν])
end

function h(μ::T, ν::T) where {T <: Real}
    J  = jacobian(μ,ν) 
    h  =  (J * g(μ,ν) * J')
    return SMatrix{2,2}(h)
end

