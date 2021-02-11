#---------------------------------------------------------------
# LaplaceOnASphere
# Soham 3/20
# Compute scalar and tensor spherical harmonics
# Ψ = ∇Y
# Φ = e^b_a ∇Y
#---------------------------------------------------------------

using GSL
export Ylm, Glm, Clm  

function unpack(x::gsl_sf_result)
    return x.val, x.err
end

function safe_sf_legendre_sphPlm_e(l::Int, m::Int, θ::T)::T where {T}
    abstol = 1e-8
    if abs(m) > l
        Yl = 0
    else
        Yl, E = unpack(sf_legendre_sphPlm_e(l, abs(m), cos(θ)))
        Yl = m < 0 && isodd(m) ? -Yl : Yl
        try
            @assert isless(abs(E), abstol)
        catch
            println("safe_sf_legendre_sphPlm_e exceeds abstol = $abstol at l = $l, m = $m, θ = $θ with error $E")
            @assert isless(abs(E), abstol)
        end
    end
    return Yl
end

# TODO: Move to real spherical harmonics
function Ylm(l::Int, m::Int, θ::T, ϕ::T)::Complex{T} where {T}
    Yl = safe_sf_legendre_sphPlm_e(l, m, θ)
    return Yl*cis(m*ϕ) 
end

function dYdθ(l::Int, m::Int, θ::T, ϕ::T)::Complex{T} where {T}
    return m*cot(θ)*Ylm(l,m,θ,ϕ) + sqrt((l-m)*(l+m+1))*cis(-ϕ)*Ylm(l,m+1,θ,ϕ)
end

function dYdϕ(l::Int, m::Int, θ::T, ϕ::T)::Complex{T} where {T}
    return im*m*Ylm(l,m,θ,ϕ)
end

function Glm(a::Int, l::Int, m::Int, θ::T, ϕ::T)::Complex{T} where {T}
    if a  == 1
        return dYdθ(l,m,θ,ϕ)
    elseif a == 2
        return dYdϕ(l,m,θ,ϕ)
    end
end

function Clm(a::Int, l::Int, m::Int, θ::T, ϕ::T)::Complex{T} where {T}
    if a  == 1
        return +(1/sin(θ))*dYdϕ(l,m,θ,ϕ)
    elseif  a == 2
        return -sin(θ)*dYdθ(l,m,θ,ϕ)
    end
end
