#-----------------------------------------------------
# Start with nice coordinates x, y, z. Rotate
# them to get X, Y, Z. Compute g to check all your routines 
# Soham 04/2021
#-----------------------------------------------------

using FastSphericalHarmonics, LinearAlgebra, Test, ForwardDiff, CairoMakie, Random

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
    @assert eltype(K) <: Real
    @assert K' * K ≈ I
    x, y, z = K * [x, y, z]
    return (x, y, z)
end

function qinv(a::Int, b::Int, μ::T, ν::T) where {T <: Real} 
    if a == b == 1
        return 1.0
    elseif a == b == 2
        return csc(μ)^2
    else
        return 0.0
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

function jacobianxyz(r::T, θ::T, ϕ::T) where {T}
    return ForwardDiff.jacobian(xyz_of_rθϕ ∘ r′θ′ϕ′_of_rθϕ, [r, θ, ϕ])
end

function q′inv(a::Int, b::Int, θ::T, ϕ::T) where {T <: Real}
    hinv  = [qinv(a, b, θ, ϕ) for a in 1:2, b in 1:2]  
    g′inv =  (J * hinv * J')
    return g′inv[a, b]
end

function chop(x::T) where {T}
    x < 1e-12 ? (return 0.0) : (return x)
end

function isdiagonal(x::Matrix{T}) where {T}
    return istril(x) && istriu(x)
end

function grad(F⁰::Matrix{T}, lmax::Int) where {T <: Real}
    C⁰  = spinsph_transform(F⁰, 0)
    ðC⁰ = spinsph_eth(C⁰, 0)
    F¹  = spinsph_evaluate(ðC⁰, 1)
    ∂F₁ = map(x->x[1], F¹) 
    ∂F₂ = map(x->x[2], F¹)
    sinθ = map((μ, ν)->sin(μ), lmax)
    return (-∂F₁, - sinθ .* ∂F₂) 
end

# Now compute the Jacobian with spin-weighted spherical harmonics
# and check if we get sane results. First we compute all the fields 
# over the spheres. However, we cannot compute derivatives of r′, θ′ and ϕ′
# using spherical harmoncis since the ϕ function has a jump. Therefore 
# we use the chain rule to compute the Jacobian.
function jacobian(x::Matrix{T}, y::Matrix{T}, z::Matrix{T}) where {T<:Real}
    # Compute Jacobian with spin-weighted spherical harmonics
    dxdθ, dxdϕ = grad(x, lmax) 
    dydθ, dydϕ = grad(y, lmax) 
    dzdθ, dzdϕ = grad(z, lmax) 

    # Compute Jacobian with AD
    dxdθ_ad = map((μ,ν)->jacobianxyz(1.0,μ,ν)[1,2], lmax)
    dxdϕ_ad = map((μ,ν)->jacobianxyz(1.0,μ,ν)[1,3], lmax)
    dydθ_ad = map((μ,ν)->jacobianxyz(1.0,μ,ν)[2,2], lmax)
    dydϕ_ad = map((μ,ν)->jacobianxyz(1.0,μ,ν)[2,3], lmax)
    dzdθ_ad = map((μ,ν)->jacobianxyz(1.0,μ,ν)[3,2], lmax)
    dzdϕ_ad = map((μ,ν)->jacobianxyz(1.0,μ,ν)[3,3], lmax)

    @debug println("Testing derivatives for xyz")
    @debug @show maximum(abs.(dxdθ - dxdθ_ad))
    @debug @show maximum(abs.(dydθ - dydθ_ad))
    @debug @show maximum(abs.(dzdθ - dzdθ_ad))
    @debug @show maximum(abs.(dxdϕ - dxdϕ_ad))
    @debug @show maximum(abs.(dydϕ - dydϕ_ad))
    @debug @show maximum(abs.(dzdϕ - dzdϕ_ad))

    r²    = sqrt.(x .^2 + y .^2 + z .^2)
    denom = sqrt.((r² - z .^2) ./ r²) .* r² .^ (3/2) 
    dθ′dθ = (z .* (x .* dxdθ + y .* dydθ) - dzdθ .* (x .^2 + y .^2)) ./ denom  
    dθ′dϕ = (z .* (x .* dxdϕ + y .* dydϕ) - dzdϕ .* (x .^2 + y .^2)) ./ denom 
    dϕ′dθ = (x .* dydθ - y .* dxdθ) ./ (x .^2 + y .^2)
    dϕ′dϕ = (x .* dydϕ - y .* dxdϕ) ./ (x .^2 + y .^2)

    # AD derivatives
    dθ′dθ_ad = map((μ,ν)->jacobian(1.0,μ,ν)[2,2], lmax)
    dθ′dϕ_ad = map((μ,ν)->jacobian(1.0,μ,ν)[2,3], lmax)
    dϕ′dθ_ad = map((μ,ν)->jacobian(1.0,μ,ν)[3,2], lmax)
    dϕ′dϕ_ad = map((μ,ν)->jacobian(1.0,μ,ν)[3,3], lmax)

    return (dθ′dθ, dθ′dϕ, dϕ′dθ, dϕ′dϕ)
end

function transform(hinv::NTuple{3, Matrix{T}}, J::NTuple{4,Matrix{T}}) where {T<:Real}
    hinv11, hinv12, hinv22 = hinv 
    hinv21 = hinv12
    d1d1, d1d2, d2d1, d2d2 = J
    ginv11 = d1d1 .* d1d1 .* hinv11 + d1d1 .* d1d2 .* hinv12 + d1d2 .* d1d1 .* hinv21 + d1d2 .* d1d2 .* hinv22 
    ginv12 = d1d1 .* d2d1 .* hinv11 + d1d1 .* d2d2 .* hinv12 + d1d2 .* d2d1 .* hinv21 + d1d2 .* d2d2 .* hinv22 
    ginv22 = d2d1 .* d2d1 .* hinv11 + d2d1 .* d2d2 .* hinv12 + d2d2 .* d2d2 .* hinv21 + d2d2 .* d2d2 .* hinv22 
    return (ginv11, ginv12, ginv22)
end

function computemetric(lmax::Int)
    h11 = map((μ,ν)->qinv(1,1,μ,ν), lmax)
    h12 = map((μ,ν)->qinv(1,2,μ,ν), lmax)
    h22 = map((μ,ν)->qinv(2,2,μ,ν), lmax)
    return (h11, h12, h22)
end

function x′y′z′_of_rθϕ(lmax::Int) 
    x = map((μ,ν)->xyz_of_rθϕ(r′θ′ϕ′_of_rθϕ([1.0,μ,ν]))[1], lmax) 
    y = map((μ,ν)->xyz_of_rθϕ(r′θ′ϕ′_of_rθϕ([1.0,μ,ν]))[2], lmax) 
    z = map((μ,ν)->xyz_of_rθϕ(r′θ′ϕ′_of_rθϕ([1.0,μ,ν]))[3], lmax) 
    return (x, y, z)
end

lmax = 13
Random.seed!(lmax)
A = rand(3,3)
ginv = computemetric(lmax)
x′, y′, z′ = x′y′z′_of_rθϕ(lmax) 
jac = jacobian(x′, y′, z′)
h11, h12, h22 = transform(ginv, jac)
@show maximum(abs.(h12))
