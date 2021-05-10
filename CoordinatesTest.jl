#-----------------------------------------------------
# Start with nice coordinates x, y, z. Rotate
# them to get X, Y, Z. Compute g to check all your routines 
# Soham 04/2021
#-----------------------------------------------------

using FastSphericalHarmonics, LinearAlgebra, Test, ForwardDiff, CairoMakie

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

function qinv(a::Int, b::Int, μ::T, ν::T) where {T <: Real} 
    if a == b == 1
        return 1 
    elseif a == b == 2
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

function jacobian(θ′::Matrix{T}, ϕ′::Matrix{T}, lmax::Int) where {T}
    dθ′dθ, dθ′dϕ = grad(θ′, lmax) 
    dϕ′dθ, dϕ′dϕ = grad(ϕ′, lmax) 
    return (dθ′dθ, dθ′dϕ, dϕ′dθ, dϕ′dϕ)
end

A = rand(3,3)
θ, ϕ = (π/5, π/7)  

# Test that rotation preserves the diagonal metric.
hinv = [qinv(a,b,θ,ϕ) for a in 1:2, b in 1:2] 
J    = jacobian(1.0, θ, ϕ)[2:end, 2:end]
ginv = [q′inv(a,b,θ,ϕ) for a in 1:2, b in 1:2] 
@debug display(hinv)
@debug display(chop.(ginv))
@test isdiagonal(hinv)
@test isdiagonal(chop.(ginv))

# Now compute the Jacobian with spin-weighted spherical harmonics
# and check if we get sane results. First we compute all the fields 
# over the spheres.
lmax = 13
r′ = map((μ, ν)->r′θ′ϕ′_of_rθϕ([1,μ,ν])[1], lmax) 
θ′ = map((μ, ν)->r′θ′ϕ′_of_rθϕ([1,μ,ν])[2], lmax) 
ϕ′ = map((μ, ν)->r′θ′ϕ′_of_rθϕ([1,μ,ν])[3], lmax) 
hinv11 = map((μ, ν)->qinv(1,1,μ,ν), lmax) 
hinv12 = hinv21 = map((μ, ν)->qinv(1,2,μ,ν), lmax) 
hinv22 = map((μ, ν)->qinv(2,2,μ,ν), lmax) 

# Compute derivatives and the Jacobian
d1d1, d1d2, d2d1, d2d2 = jacobian(θ′, ϕ′, lmax) 
ginv11 = d1d1 .* d1d1 .* hinv11 + d1d1 .* d1d2 .* hinv12 + d1d2 .* d1d1 .* hinv21 + d1d2 .* d1d2 .* hinv22 
ginv12 = d1d1 .* d2d1 .* hinv11 + d1d1 .* d2d2 .* hinv12 + d1d2 .* d2d1 .* hinv21 + d1d2 .* d2d2 .* hinv22 
ginv22 = d2d1 .* d2d1 .* hinv11 + d2d1 .* d2d2 .* hinv12 + d2d2 .* d2d2 .* hinv21 + d2d2 .* d2d2 .* hinv22 

# FIXME: Fix the Jacobian using spin-weighted spherical harmonics. 
@test_broken maximum(abs.(ginv12)) ≈ 1e-12

# Compare the Jacobians from SWSH and AD. The ϕ derivatives seem to divergence,
# and the θ ones converge very, very slowly.
∂θ′∂θ = map((μ, ν)->jacobian(1.0,μ,ν)[2,2], lmax)  
∂θ′∂ϕ = map((μ, ν)->jacobian(1.0,μ,ν)[2,3], lmax)  
∂ϕ′∂θ = map((μ, ν)->jacobian(1.0,μ,ν)[3,2], lmax)  
∂ϕ′∂ϕ = map((μ, ν)->jacobian(1.0,μ,ν)[3,3], lmax)  

@show maximum(abs.(d1d1 - ∂θ′∂θ))
@show maximum(abs.(d1d2 - ∂θ′∂ϕ))
@show maximum(abs.(d2d1 - ∂ϕ′∂θ))
@show maximum(abs.(d2d2 - ∂ϕ′∂ϕ))

# Check that if you use the other Jacobian, it all works out.
d1d1 = ∂θ′∂θ
d1d2 = ∂θ′∂ϕ
d2d1 = ∂ϕ′∂θ
d2d2 = ∂ϕ′∂ϕ
ginv11 = d1d1 .* d1d1 .* hinv11 + d1d1 .* d1d2 .* hinv12 + d1d2 .* d1d1 .* hinv21 + d1d2 .* d1d2 .* hinv22 
ginv12 = d1d1 .* d2d1 .* hinv11 + d1d1 .* d2d2 .* hinv12 + d1d2 .* d2d1 .* hinv21 + d1d2 .* d2d2 .* hinv22 
ginv22 = d2d1 .* d2d1 .* hinv11 + d2d1 .* d2d2 .* hinv12 + d2d2 .* d2d2 .* hinv21 + d2d2 .* d2d2 .* hinv22 
@test maximum(abs.(ginv12)) < 1e-12

# Check if the functions fall off rapidly enough for a spin-weighted spherical
# harmonic expansion with l = 13 to make sense.
function coefficents(F⁰::Matrix{T}, string::String) where {T} 
    C⁰  = spinsph_transform(F⁰, 0)
    display(C⁰)
    fig, ax, hm = heatmap(log.(C⁰))
    Colorbar(fig[1, 2], hm)
    save("output/$string.pdf", fig)
end

coefficents(r′, "theta-expansion")
coefficents(θ′, "theta-expansion")
coefficents(ϕ′, "phi-expansion")
