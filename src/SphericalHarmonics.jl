#---------------------------------------------------------------
# LaplaceOnASphere
# Soham 3/20
# Compute scalar and tensor spherical harmonics
# VectorSPH = ∇ScalarSPH
#---------------------------------------------------------------

using GSL
export ScalarSPH, VectorSPH, dYdθ, dYdϕ
abstol = 1e-5

function unpack(x::gsl_sf_result)
    return x.val, x.err
end

function safe_sf_legendre_sphPlm_e(l::Int, m::Int, θ::T)::T where {T}
    if abs(m) > l
        Yl = 0
    else
        Yl, E = unpack(sf_legendre_sphPlm_e(l, abs(m), cos(θ)))
        Yl = m < 0 && isodd(m) ? -Yl : Yl
        try
            @assert isless(abs(E), abstol)
        catch
            @show l, m, θ, E
            @assert isless(abs(E), abstol)
        end
    end
    return Yl
end

function ScalarSPH(l::Int, m::Int, θ::T, ϕ::T)::Complex{T} where {T}
    Yl = safe_sf_legendre_sphPlm_e(l, m, θ)
    return Yl*cis(m*ϕ) 
end

function dYdθ(l::Int, m::Int, θ::T, ϕ::T)::Complex{T} where {T}
    return m*cot(θ)*ScalarSPH(l,m,θ,ϕ) + sqrt((l-m)*(l+m+1))*cis(-ϕ)*ScalarSPH(l,m+1,θ,ϕ)
end

function dYdϕ(l::Int, m::Int, θ::T, ϕ::T)::Complex{T} where {T}
    return im*m*ScalarSPH(l,m,θ,ϕ)
end

function VectorSPH(l::Int, m::Int, θ::T, ϕ::T)::NTuple{2, Complex{T}} where {T}
    # TODO: Generalize for an arbitrary metric. 
    dYdθ = m*cot(θ)*ScalarSPH(l,m,θ,ϕ) + sqrt((l-m)*(l+m+1))*cis(-ϕ)*ScalarSPH(l,m+1,θ,ϕ)
    dYdϕ = im*m*ScalarSPH(l,m,θ,ϕ)
    return (dYdθ, (1/sin(θ))*dYdϕ)
end
