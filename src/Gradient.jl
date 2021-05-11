#-----------------------------------------------------
# Implement the gradient and coordinate transofrmation 
# utilities. 
# Soham 05/2021
#-----------------------------------------------------

using FastSphericalHarmonics

export cartesian2spherical, spherical2cartesian
export grad, jacobian, transform

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

function grad(F⁰::Matrix{T}, lmax::Int) where {T <: Real}
    C⁰  = spinsph_transform(F⁰, 0)
    ðC⁰ = spinsph_eth(C⁰, 0)
    F¹  = spinsph_evaluate(ðC⁰, 1)
    ∂F₁ = map(x->x[1], F¹) 
    ∂F₂ = map(x->x[2], F¹)
    sinθ = map((μ, ν)->sin(μ), lmax)
    return (-∂F₁, - sinθ .* ∂F₂) 
end

function jacobian(x::Matrix{T}, y::Matrix{T}, z::Matrix{T}, lmax::Int) where {T<:Real}
    dxdθ, dxdϕ = grad(x, lmax) 
    dydθ, dydϕ = grad(y, lmax) 
    dzdθ, dzdϕ = grad(z, lmax) 

    r²    = sqrt.(x .^2 + y .^2 + z .^2)
    denom = sqrt.((r² - z .^2) ./ r²) .* r² .^ (3/2) 
    dθ′dθ = (z .* (x .* dxdθ + y .* dydθ) - dzdθ .* (x .^2 + y .^2)) ./ denom  
    dθ′dϕ = (z .* (x .* dxdϕ + y .* dydϕ) - dzdϕ .* (x .^2 + y .^2)) ./ denom 
    dϕ′dθ = (x .* dydθ - y .* dxdθ) ./ (x .^2 + y .^2)
    dϕ′dϕ = (x .* dydϕ - y .* dxdϕ) ./ (x .^2 + y .^2)

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
