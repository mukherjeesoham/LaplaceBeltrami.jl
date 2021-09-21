#-----------------------------------------------------
# Implement the Linear Operators using FD
# Soham 05/2021
#-----------------------------------------------------

using FastSphericalHarmonics, StaticArrays
export grad, div, S1FD, S2FD 

function grad(F::AbstractMatrix{T}, ni::Int, nj::Int) where {T <: Real}
    Dθ, Dϕ = dscalar(ni, nj, 4)
    gradF1 = Dθ*vec(F)
    gradF2 = Dϕ*vec(F)
    dF  = map((x,y)->SVector{2}([x,y]), reshape(gradF1, ni, nj), reshape(gradF2, ni, nj))
    return dF 
end

function Base. div(F::AbstractMatrix{SVector{2, T}}, ni::Int, nj::Int) where {T<:Real}
    Dθ̄, Dϕ̄ = dvector(ni, nj, 4) 
    F1 = map(x->x[1], F) 
    F2 = map(x->x[2], F) 
    divF = Dθ̄*vec(F1) + Dϕ̄*vec(F2) 
    return reshape(divF, ni, nj)
end

function S1FD(h::AbstractMatrix{T}, F¹::AbstractVector{T}) where {T<:Real}
    return sqrt(det(h)) .* raise(inv(h), F¹)
end

function S2FD(h::AbstractMatrix{T}, F⁰::T) where {T<:Real}
    return sqrt(1 / det(h)) .* F⁰
end
