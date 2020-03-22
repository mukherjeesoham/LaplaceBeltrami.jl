using GSL

export Ylm, dYlmdθ, dYlmdϕ, Ψlm

function unpack(x::gsl_sf_result)
    return x.val, x.err
end

function safe_sf_legendre_sphPlm_e(l::Int, m::Int, θ::Number)::Number
    if abs(m) > l
        Yl = 0
        # @warn "Computing Spherical Harmonics wher |m| > l"
    else
        Yl, E = unpack(sf_legendre_sphPlm_e(l, abs(m), cos(θ)))
        Yl = m < 0 && isodd(m) ? -Yl : Yl
        @assert isless(E, 1e-12)
    end
    return Yl
end
function Ylm(l::Int, m::Int, θ::Number, ϕ::Number)::Number
    Yl = safe_sf_legendre_sphPlm_e(l, m, θ)
    return Yl*cis(m*ϕ) 
end

function dYlmdθ(l::Int, m::Int, θ::Number, ϕ::Number)::Number
    return m*cot(θ)*Ylm(l,m,θ,ϕ) + sqrt((l-m)*(l+m+1))*cis(ϕ)*Ylm(l,m+1,θ,ϕ)
end

function dYlmdϕ(l::Int, m::Int, θ::Number, ϕ::Number)::Number
    return im*m*Ylm(l, m, θ, ϕ)
end

function Ψlm(l::Int, m::Int, θ::Number, ϕ::Number)::NTuple{2, Number}
    return (dYlmdθ(l, m, θ, ϕ), 
            dYlmdϕ(l, m, θ, ϕ))
end
