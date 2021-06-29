#-----------------------------------------------------
# Implement the Linear Operators
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

function matrix(μ::T, ν::T) where {T<:Real}
    return SMatrix{2,2}([-1.0 0.0; 0.0 sin(μ)])
end

function gradbar(C⁰::AbstractMatrix{T}, lmax::Int) where {T <: Real}
    ðC⁰ = spinsph_eth(C⁰, 0)
    F¹  = spinsph_evaluate(ðC⁰, 1)
    return F¹
end

function divbar(F¹::AbstractMatrix{SVector{2, T}}) where {T<:Real}
    C¹  = spinsph_transform(F¹, 1)  
    ð̄C¹ = spinsph_ethbar(C¹, 1)
    ∇²F = spinsph_evaluate(ð̄C¹, 0) 
    return ∇²F
end

function grad(C⁰::AbstractMatrix{T}, lmax::Int) where {T <: Real}
    ð̄C⁰ = spinsph_ethbar(C⁰, 0)
    F¹  = spinsph_evaluate(ð̄C⁰, -1)
    M   = map(matrix, lmax)
    return M .* F¹ 
end

function Base.filter!(C¹::AbstractMatrix{SVector{2,T}}, lmax::Int) where {T<:Real}
    for l in 1:lmax, m in (-l):l 
        if abs(m) >= l
            C¹[spinsph_mode(-1, l, m)] = 0. * C¹[spinsph_mode(-1,l,m)] 
        end
    end
end

function Base. div(F¹::AbstractMatrix{SVector{2, T}}, lmax::Int) where {T<:Real}
    M   = map(matrix, lmax)
    F¹  = SVector{2}.(inv.(M) .* F¹)
    C¹  = spinsph_transform(F¹, -1)  
    # filter!(C¹, lmax)  # Throw away the abs(m) > l modes
    ðC¹ = spinsph_eth(C¹, -1)
    ∇²F = spinsph_evaluate(ðC¹, 0) 
    return ∇²F
end

function curl(F¹::AbstractMatrix{SVector{2, T}}, q::AbstractMatrix, lmax::Int) where {T}
    # FIXME: Curl of a the gradient should be zero. Is this a good way to
    # compute the derivatives? These are vector components with a basis. 
    # Wait--how does this work?
    curlF = similar(F¹) 
    dFθ   = map(x->[0.0,x...], grad(map(x->x[1], F¹), lmax))
    dFϕ   = map(x->[0.0,x...], grad(map(x->x[2], F¹), lmax))
    dFr   = 0 .* dFθ # Introduce a dummy direction
    dF    = [dFr, dFθ, dFϕ]
    for index in CartesianIndices(F¹)
        curlF[index] = [sum(levicivita([a,b,c]) * dF[b][index][c]  for b in 1:3, c in 1:3) for a in 1:3][2:3]
    end
    curlF = (1 ./ sqrt.(det.(q))) .* curlF
    return map(raise, inv.(q), curlF) 
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
    ∇F⁰      = grad(C⁰, A.lmax)             # Tested 

    # TODO: Check if the curl of the gradient is zero. The scaling with the determinant 
    # of h is not necessary here, unless it does good things at the pole. 
    curl∇F⁰    = curl(∇F⁰, A.q, A.lmax) 
    @assert maximum(norm.(curl∇F⁰)) < 1e-12

    S1∇F⁰    = map(S1, A.q, A.h, ∇F⁰)       # Tested 

    # TODO: Modify the gradient such that it's not a gradient anymore. Then compute the curl, 
    # followed by the divergence to check if the identities are satisfied.
    curlS1∇F⁰ = curl(S1∇F⁰, A.q, A.lmax)
    divcurlS1∇F⁰  = div(curlS1∇F⁰, A.lmax)  
    @assert maximum(abs.(divcurlS1∇F⁰)) < 1e-12            # TEST: Divergence of a curl should be zero. 
    @assert quad(divcurlS1∇F⁰) < 1e-12            # TEST: Integrate the divergence on the sphere.
    
    ∇S1∇F⁰   = div(S1∇F⁰, A.lmax)           # FIXME: Compare with AD 

    @assert quad(∇S1∇F⁰) < 1e-12            # TEST: Integrate the divergence on the sphere.

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
