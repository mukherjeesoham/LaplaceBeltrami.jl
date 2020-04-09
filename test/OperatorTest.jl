#---------------------------------------------------------------
# LaplaceOnASphere
# Soham 3/20
# Test operators
#---------------------------------------------------------------

using LinearAlgebra

lmax = 10
for l in 0:lmax
    S = SphericalHarmonics(l)
    L = nodal_to_modal_scalar_op(S)
    P = modal_to_nodal_scalar_op(S)
    @test isless(maximum(abs.(L*P - I)), 1e-12)
    @test P*L ≈ (P*L)^2 
    L̄ = nodal_to_modal_vector_op(S)
    P̄ = modal_to_nodal_vector_op(S)
    @test isless(maximum(abs.(L*P - I)), 1e-12)
    @test P*L ≈ (P*L)^2 
end

function gab(a::Int, b::Int, θ::T, ϕ::T)::T where {T}
    @assert (1 <= a <= 2) && (1 <= b <= 2)
    if a == 1 && b == 1
        return 1
    elseif a == 2 && b == 2
        return sin(θ)^2
    else
        return cos(θ) + sin(ϕ)
    end
end

function detg(a::Int, b::Int, θ::T, ϕ::T)::T where {T}
    @assert (1 <= a <= 2) && (1 <= b <= 2)
    if a == b
        return sin(θ)
    else
        return 0
    end
end

function g(θ::T, ϕ::T)::T where {T}
    return cos(θ)*sin(ϕ)
end

SH = SphericalHarmonics(3)
ub = map(SH, (θ, ϕ)->sin(θ), (θ, ϕ)->cos(ϕ))
ua = map(SH, (θ, ϕ)->gab(1,1,θ,ϕ)*sin(θ) + gab(1,2,θ,ϕ)*cos(ϕ), 
             (θ, ϕ)->gab(1,2,θ,ϕ)*sin(θ) + gab(2,2,θ,ϕ)*cos(ϕ))
hab = scaling_vector_op(SH, gab)
@test ua ≈ hab*ub

x = map(SH, (θ, ϕ)->sin(θ)*cos(ϕ))
y = map(SH, (θ, ϕ)->cos(θ)*sin(θ)*cos(ϕ)*sin(ϕ))
s = scaling_scalar_op(SH, g)
@test y ≈ s*x

function lscaling(l::Int, m::Int)::Int
    return l*(l+1)
end

function modal_to_nodal_scalar_scaled_op(S::SphericalHarmonics{T})::Array{Complex{T}, 2} where {T}
    lmax, n = S.lmax, S.N
    A = zeros(Complex{T}, n*(2*n), (lmax)^2 + 2*(lmax) + 1)
    for index in CartesianIndices(A)
        (i,j) = split(index.I[1], n)
        (l,m) = split(index.I[2])
        (θ,ϕ) = collocation(S,i,j)
        A[index] = l*(l+1)*ScalarSPH(l, m, θ, ϕ)
    end
    return A
end

L  = modal_scaling_op(SH, lscaling)
@test modal_to_nodal_scalar_op(SH)*L ≈ modal_to_nodal_scalar_scaled_op(SH)
