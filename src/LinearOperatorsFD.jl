#-----------------------------------------------------
# Implement the Linear Operators using FD
# Soham 05/2021
#-----------------------------------------------------

using FastSphericalHarmonics, StaticArrays, Combinatorics
export grad, laplace, Laplace, S1, S2
export gradbar, divbar
using CairoMakie

struct Laplace{T} <: AbstractMatrix{T}
    lmax::Int
    q::Matrix{AbstractArray{T}}
    h::Matrix{AbstractArray{T}}
end

function npoints(lmax::Int)::NTuple{2, Int}
    return (2lmax + 1, 2 * (2lmax + 1))
end

function Base.filter!(C¹::AbstractMatrix{SVector{2,T}}, lmax::Int) where {T<:Real}
    for l in 1:lmax, m in (-l):l 
        if abs(m) >= l
            C¹[spinsph_mode(-1, l, m)] = 0. * C¹[spinsph_mode(-1,l,m)] 
        end
    end
end

function reshapeFD2SH(F::Array{T,1}, ni::Int, nj::Int) where {T}
    return reshape(F, ni, nj) 
end

function reshapeSH2FD(F::Array{T,2}) where {T}
    return vec(F) 
end

function grad(C⁰::AbstractMatrix{T}, lmax::Int) where {T <: Real}
    # FIXME: Fix spin_evaluate
    F⁰     = spinsph_evaluate(C⁰, 0)
    @show size(F⁰) 
    F⁰     = reshapeSH2FD(F⁰)
    nθ, nϕ = npoints(lmax)
    @show nθ, nϕ
    @show size(F⁰)
    Dθ, Dϕ = dscalar(nθ, nϕ, 4) # 4th order FD # FIXME: The sizes are wrong 
    @show size.((nθ, nϕ, Dθ, Dϕ))
    # FIXME: The dimensions seem incorrect for the operators
    sinθ   = map((μ,ν)->sin(μ), nθ, nϕ)
    dF     = map((x,y)->SVector{2}([x,y]), Dθ*F⁰, inv.(sinθ) .* (Dϕ*F⁰))
    @show typeof(dF)
    return dF 
end

function Base. div(F¹::AbstractMatrix{SVector{2, T}}, lmax::Int) where {T<:Real}
    nθ, nϕ = npoints(lmax) 
    Dθ̄, Dϕ̄ = dvector(nθ, nϕ, 4) # 4th order FD
    Fθ = map(x-x[1], F¹) 
    Fϕ = map(x-x[2], F¹) 
    sinθ   = map((μ,ν)->sin(μ), nθ, nϕ)
    # TODO: Introduce regularized variables
    divF = inv(sinθ) * Dθ̄*(sinθ .* Fθ) + inv(sinθ).* (Dϕ̄*Fϕ)
    F⁰   = reshapeFD2SH(F⁰, nθ, nϕ)
    return divF
end

function S1(q::AbstractMatrix{T}, h::AbstractMatrix{T}, F¹::AbstractVector{T}) where {T<:Real}
    return sqrt(det(h) / det(q)) .* lower(q, raise(inv(h), F¹))
end

function S2(q::AbstractMatrix{T}, h::AbstractMatrix{T}, F⁰::T) where {T<:Real}
    return sqrt(det(q) / det(h)) .* F⁰
end

function S3(q::AbstractMatrix{T}, h::AbstractMatrix{T}, F¹::AbstractVector{T}) where {T<:Real}
    return sqrt(det(q) / det(h)) .* F¹
end

function laplace(C⁰::AbstractMatrix{T}, A::Laplace) where {T<:Real}
    ∇F⁰      = grad(C⁰, A.lmax)             # TODO: Test 
    S1∇F⁰    = map(S1, A.q, A.h, ∇F⁰)       # Tested 
    ∇S1∇F⁰   = div(S1∇F⁰, A.lmax)           # TODO: Test 
    S2∇S1∇F⁰ = map(S2, A.q, A.h, ∇S1∇F⁰)    # Tested 
    return spinsph_transform(S2∇S1∇F⁰, 0) 
end

function laplace(x::AbstractArray{Float64,1}, A::Laplace)::AbstractArray{Float64,1}
    return vec(laplace(reshape(x, A.lmax + 1, 2*A.lmax + 1), A))
end

function LinearAlgebra.mul!(y::AbstractVector, A::Laplace, x::AbstractVector)
    y .= laplace(x, A)
end

function Base.size(A::Laplace)
    size = (A.lmax + 1) * (2 * A.lmax + 1)
    return (size, size)
end

function LinearAlgebra.issymmetric(::Laplace)
    return true
end

Base.eltype(::Laplace{T}) where T = T

function LinearAlgebra.mul!(y::AbstractMatrix, A::Laplace, x::AbstractMatrix)
    @assert size(y,2) == size(x,2)
    for column in 1:size(y,2)
        mul!((@view y[:, column]), A, (@view x[:, column]))
    end
end

function LinearAlgebra.mul!(y::AbstractMatrix, A::Laplace, x::AbstractMatrix, α::Number, β::Number)
    @assert size(y,2) == size(x,2)
    ycopy = copy(y) 
    for column in 1:size(y,2)
        mul!((@view y[:, column]), A, (@view x[:, column]))
    end

    if α == β == 0
        y .= 0 
    elseif α == 0
        y .= β*ycopy
    elseif β == 0
        y .= α*y
    else
        y .= α*y + β*ycopy
    end
end
