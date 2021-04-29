#-----------------------------------------------------
# Start with nice coordinates x, y, z. Rotate
# them to get X, Y, Z. Compute g to check all your routines 
# Soham 04/2021
#-----------------------------------------------------

using FastSphericalHarmonics, LinearAlgebra, Test, ForwardDiff

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

function rotate(x::T, y::T, z::T) where {T <: Real} 
    K  = eigen(A + A').vectors
    @assert K' * K ≈ I
    x, y, z = K * [x, y, z]
    return (x, y, z)
end

function grad(F⁰::Array{T,2}, lmax::Int) where {T <: Real}
    C⁰  = spinsph_transform(Complex.(F⁰), 0)
    ðC¹ = spinsph_eth(C⁰, 0)
    ðF¹ = spinsph_evaluate(ðC¹, 1)
    sinθ = map((μ, ν)->sin(μ), lmax)
    return (real.(ðF¹), imag.(ðF¹)) 
end

function qinv(a::Int, b::Int, μ::T, ν::T) where {T <: Real} 
    if a == b == 1
        return 1
    elseif a == b == 2
        return 1 
    elseif a == b == 3
        return csc(μ)^2
    else
        return 0
    end
end

function r′θ′ϕ′_of_rθϕ(coords::Array{T,1}) where {T<:Real}
    r,  θ,  ϕ  = coords 
    r′, θ′, ϕ′ = cartesian2spherical(rotate(spherical2cartesian(r, θ, ϕ)...)...)
    return [r′, θ′, ϕ′]
end

function xyz_of_rθϕ(coords::Array{T,1}) where {T<:Real}
    r,  θ,  ϕ  = coords 
    x, y, z = spherical2cartesian(r, θ, ϕ)
    return [x, y, z]
end

function jacobian(r::T, θ::T, ϕ::T) where {T}
    return ForwardDiff.jacobian(r′θ′ϕ′_of_rθϕ, [r, θ, ϕ])
end

function q′inv(a::Int, b::Int, θ::T, ϕ::T) where {T <: Real}
    hinv  = [qinv(a, b, θ, ϕ) for a in 1:3, b in 1:3]  
    g′inv =  (J * hinv * J')
    return g′inv[a, b]
end

function chop(x::T) where {T}
    x < 1e-12 ? (return 0.0) : (return x)
end

lmax = 13
A = rand(3,3)
θ, ϕ = (π/5, π/7)  

# FIXME: An arbitrary rotation doesn't keep the metric diagonal? What?
hinv = [qinv(a,b,θ,ϕ) for a in 1:3, b in 1:3] 
J    = jacobian(1.0, θ, ϕ)
ginv = [q′inv(a,b,θ,ϕ) for a in 1:3, b in 1:3] 
display(hinv)
display(chop.(ginv))

