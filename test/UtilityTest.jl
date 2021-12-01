#-----------------------------------------------------
# Test the quadrature routines based on spherical
# designs
# Soham 11/2021
#-----------------------------------------------------
using Test, LinearAlgebra

function u0(X::AbstractArray) where {T<:Real} 
    return 1.0
end

function u1(X::AbstractArray) where {T<:Real} 
    (x, y, z) = X 
    return 1 + x + y^2 + (x^2 * y) + x^4 + y^5 + (x^2 * y^2 * z^2)
end

function u3(X::AbstractArray) where {T<:Real} 
    α = π
    (x, y, z) = X 
    return (1 + tanh(-α * (x + y  - z))) / α
end

function u4(X::AbstractArray) where {T<:Real} 
    α = π
    (x, y, z) = X 
    return (1 - sign(x + y - z)) / α
end

function u5(X::AbstractArray) where {T<:Real} 
    α = π
    (x, y, z) = X 
    return (1 - sign(π * x + y)) / α
end

@testset "Quadrature" begin
    @test isapprox(quad(u0, :sphericaldesigns), 4π, atol=1e-15)
    @test isapprox(quad(u1, :sphericaldesigns), (216π / 35), atol=1e-6)
    @test isapprox(quad(u3, :sphericaldesigns), 4, atol=1e-3)
    # NOTE: u5 error is larger than those reported in 
    # <https://cbeentjes.github.io/files/Ramblings/QuadratureSphere.pdf>
    @test isapprox(quad(u4, :sphericaldesigns), 4, atol=1e-3)     
    @test_broken isapprox(quad(u5, :sphericaldesigns), 4, atol=1e-3) 
end;
