using FastTransforms
using FastSphericalHarmonics
using ForwardDiff
using LinearAlgebra
using SpecialFunctions
using StaticArrays

export sYlm, chops, ∂θsYlm, ∂ϕsYlm 

bitsign(b::Bool) = b ? -1 : 1
bitsign(n::Integer) = bitsign(isodd(n))

function unit(val::T, ind::CartesianIndex{D}, size::NTuple{D,Int}) where {T,D}
    return setindex!(zeros(T, size), val, ind)
end
function unit(::Type{T}, ind::CartesianIndex{D},
              size::NTuple{D,Int}) where {T,D}
    return unit(one(T), ind, size)
end
function unit(ind::CartesianIndex{D}, size::NTuple{D,Int}) where {D}
    return unit(true, ind, size)
end

# [Generalized binomial coefficient](https://en.wikipedia.org/wiki/Binomial_coefficient#Generalization_and_connection_to_the_binomial_series)
function Base.binomial(α::Number, k::Integer)
    k == 0 && return one(α)
    return prod((α - i) / (k - i) for i in 0:(k - 1))
end

# [Jacobi
# Polynomials](https://en.wikipedia.org/wiki/Jacobi_polynomials)
function JacobiP(α, β, n, z)
    return gamma(α + n + 1) / (factorial(n) * gamma(α + β + n + 1)) *
           sum(binomial(n, m) * gamma(α + β + n + m + 1) / gamma(α + m + 1) *
               ((z - 1) / 2)^m for m in 0:n)
end

# [Associated Legendre
# Polynomials](https://en.wikipedia.org/wiki/Associated_Legendre_polynomials)
function LegendreP(l, m, x)
    return bitsign(m) *
           2^l *
           sqrt(1 - x^2)^m *
           sum(factorial(k) / factorial(k - m) *
               x^(k - m) *
               binomial(l, k) *
               binomial((l + k - 1) / 2, l) for k in m:l)
end

# [Spherical
# harmonics](https://mikaelslevinsky.github.io/FastTransforms/transforms.html)
# (section "sph2fourier")
function Ylm(l, m, θ, ϕ)
    return bitsign(abs(m)) *
           sqrt((l + 1 / 2) * factorial(l - abs(m)) / factorial(l + abs(m))) *
           LegendreP(l, abs(m), cos(θ)) *
           sqrt((2 - (m == 0)) / 2π) *
           (m ≥ 0 ? cos(abs(m) * ϕ) : sin(abs(m) * ϕ))
end

# [Spin-weighted spherical
# harmonics](https://mikaelslevinsky.github.io/FastTransforms/transforms.html)
# (section "spinsph2fourier")
function sYlm(s, l, m, θ, ϕ)
    l0 = max(abs(m), abs(s))
    l1 = min(abs(m), abs(s))
    return cis(m * ϕ) / sqrt(2π) *
           sqrt((l + 1 / 2) * factorial(l + l0) * factorial(l - l0) /
                (factorial(l + l1) * factorial(l - l1))) *
           sin(θ / 2)^abs(m + s) *
           cos(θ / 2)^abs(m - s) *
           JacobiP(abs(m + s), abs(m - s), l - l0, cos(θ))
end

sYlm(::Type{<:Complex}, s, l, m, θ, ϕ) = sYlm(s, l, m, θ, ϕ)

function sYlm(::Type{<:Real}, s, l, m, θ, ϕ)
    @assert s == 0
    if m == 0
        return real(sYlm(s, l, abs(m), θ, ϕ))
    elseif m > 0
        return sqrt(2) * real(sYlm(s, l, abs(m), θ, ϕ))
    else
        return sqrt(2) * imag(sYlm(s, l, abs(m), θ, ϕ))
    end
end

c2a(c) = SVector(real(c), imag(c))
a2c(a) = Complex(a[1], a[2])

∂θYlm(l, m, θ, ϕ) = a2c(ForwardDiff.derivative(θ -> c2a(Ylm(l, m, θ, ϕ)), θ))
∂ϕYlm(l, m, θ, ϕ) = a2c(ForwardDiff.derivative(ϕ -> c2a(Ylm(l, m, θ, ϕ)), ϕ))

function ∂θsYlm(s, l, m, θ, ϕ)
    return a2c(ForwardDiff.derivative(θ -> c2a(sYlm(s, l, m, θ, ϕ)), θ))
end

function ∂ϕsYlm(s, l, m, θ, ϕ)
    return a2c(ForwardDiff.derivative(ϕ -> c2a(sYlm(s, l, m, θ, ϕ)), ϕ))
end

function ∂θsYlm(::Type{T}, s, l, m, θ, ϕ) where {T}
    return a2c(ForwardDiff.derivative(θ -> c2a(sYlm(T, s, l, m, θ, ϕ)), θ))
end

function ∂ϕsYlm(::Type{T}, s, l, m, θ, ϕ) where {T}
    return a2c(ForwardDiff.derivative(ϕ -> c2a(sYlm(T, s, l, m, θ, ϕ)), ϕ))
end
