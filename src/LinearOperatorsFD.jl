#-----------------------------------------------------
# Implement the Linear Operators using FD
# Soham 05/2021
#-----------------------------------------------------

using FastSphericalHarmonics, StaticArrays
export grad, div 

function grad(F::AbstractMatrix{T}, ni::Int, nj::Int) where {T <: Real}
    Dθ, Dϕ = dscalar(ni, nj, 4)
    sinθ   = map((μ,ν)->sin(μ), ni, nj)
    gradF1 = Dθ*vec(F)
    gradF2 = Dϕ*vec(F)
    dF  = map((x,y)->SVector{2}([x,y]), reshape(gradF1, ni, nj), reshape(gradF2, ni, nj))
    return dF 
end

function Base. div(F::AbstractMatrix{SVector{2, T}}, ni::Int, nj::Int) where {T<:Real}
    Dθ̄, Dϕ̄ = dvector(ni, nj, 4) 
    sinθ  = map((μ,ν)->sin(μ), ni, nj)
    F1 = map(x->x[1], F) 
    F2 = map(x->x[2], F) 
    # divF1 = vec(inv.(sinθ)) .* Dθ̄*(vec(sinθ) .* vec(F1))
    # divF2 = vec(inv.(sinθ)) .* (Dϕ̄*vec(F2))
    divF1 =  Dθ̄*vec(F1)
    divF2 =  Dϕ̄*vec(F2)
    divF  = divF1 + divF2
    return reshape(divF1 + divF2, ni, nj)
end
