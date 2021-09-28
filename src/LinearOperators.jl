#-----------------------------------------------------
# Implement the Linear Operators using spin-weighted spherical harmonics.
# Soham 05/2021
#-----------------------------------------------------

using FastSphericalHarmonics, StaticArrays, Plots, Dates
export grad, laplace, Laplace, S1, S2
export gradbar, divbar

function visualize(C⁰::AbstractMatrix{T}) where {T<:Real}
    x = 1:size(C⁰)[1]
    plot(x, sum(abs, C⁰, dims=2)) 
    datetime = now()
    savefig("./plots/modes/S-$datetime.png")
end

function visualize(C⁰::AbstractMatrix{T}) where {T<:StaticArrays.SVector{2, Float64}}
    x = 1:size(C⁰)[1]
    C1⁰ = map(x->x[1], C⁰)
    C2⁰ = map(x->x[2], C⁰)
    plot(x, sum(abs, C1⁰, dims=2)) 
    plot!(x, sum(abs, C2⁰, dims=2)) 
    datetime = now()
    savefig("./plots/modes/V-$datetime.png")
end

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
    visualize(C¹)  # <==== Visualize modes
    ðC¹ = spinsph_eth(C¹, -1)
    ∇²F = spinsph_evaluate(ðC¹, 0) 
    return ∇²F
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

function prolongate(C⁰::AbstractMatrix{T}, A::Laplace) where {T}
    lnew = 2 * A.lmax
    PC⁰  = zeros(lnew + 1, 2 * lnew + 1)
    for index in CartesianIndices(C⁰)
        PC⁰[index] = C⁰[index]
    end
    return PC⁰
end

function prolongate(A::Laplace)
    lnew = 2 * A.lmax
    qmetric = map(q, lnew)   
    hmetric = map(h, lnew)   
    Anew    = Laplace{Float64}(lnew, qmetric, qmetric)
    return Anew
end

function restrict(PC⁰::AbstractMatrix{T}, A::Laplace) where {T}
    return PC⁰[1:A.lmax + 1, 1:2*A.lmax + 1]
end

function laplace(C⁰::AbstractMatrix{T}, A::Laplace) where {T<:Real}
    visualize(C⁰)  # <==== Visualize modes
    dU      = grad(C⁰, A.lmax) 
    SdU     = map(S1, A.q, A.h, dU) 
    dSdU    = div(SdU, A.lmax) 
    ΔU      = map(S2, A.q, A.h, dSdU) 
    return spinsph_transform(ΔU, 0) 
end

# function laplace(C⁰::AbstractMatrix{T}, A::Laplace) where {T<:Real}
    # visualize(C⁰)  # <==== Visualize modes
    # # TODO: Should we do this only around the divergence? 
    # # FIXME: Why is this causing issues with even the simple coordinates?
    # PC⁰ = prolongate(C⁰, A) 
    # PA  = prolongate(A)
    # # Do all calculations with twice the number of modes
    # dU   = grad(PC⁰, PA.lmax) 
    # SdU  = map(S1, PA.q, PA.h, dU) 
    # dSdU = div(SdU, PA.lmax) 
    # ΔU   = map(S2, PA.q, PA.h, dSdU) 
    # # Now only keep half the number of modes 
    # RΔU = restrict(ΔU, A) 
    # return spinsph_transform(RΔU, 0) 
# end

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
