#---------------------------------------------------------------
# LaplaceOnASphere
# Soham 3/20
# Compute scalar and tensor spherical harmonics
# Ψ = ∇Y
# Φ = e^b_a ∇Y
#---------------------------------------------------------------

using GSL
export ScalarSH, dYdθ, dYdϕ, GradSH, CurlSH, TensorSHE, TensorSHO
abstol = 1e-1

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

function ScalarSH(l::Int, m::Int, θ::T, ϕ::T)::Complex{T} where {T}
    Yl = safe_sf_legendre_sphPlm_e(l, m, θ)
    return Yl*cis(m*ϕ) 
end

function dYdθ(l::Int, m::Int, θ::T, ϕ::T)::Complex{T} where {T}
    return m*cot(θ)*ScalarSH(l,m,θ,ϕ) + sqrt((l-m)*(l+m+1))*cis(-ϕ)*ScalarSH(l,m+1,θ,ϕ)
end

function dYdϕ(l::Int, m::Int, θ::T, ϕ::T)::Complex{T} where {T}
    return im*m*ScalarSH(l,m,θ,ϕ)
end

function GradSH(a::Int, l::Int, m::Int, θ::T, ϕ::T)::Complex{T} where {T}
    if a  == 1
        return dYdθ(l,m,θ,ϕ)
    elseif a == 2
        return dYdϕ(l,m,θ,ϕ)
    end
end

function CurlSH(a::Int, l::Int, m::Int, θ::T, ϕ::T)::Complex{T} where {T}
    if a  == 1
        return +(1/sin(θ))*dYdϕ(l,m,θ,ϕ)
    elseif  a == 2
        return -sin(θ)*dYdθ(l,m,θ,ϕ)
    end
end

# Add tensor spherical harmonics; see D.25 in Baumgarte and Shapiro.
function Xlm(l::Int, m::Int, θ::T, ϕ::T)::Complex{T} where {T}
    return 2*im*m*dYdθ(l,m,θ,ϕ) - 2*cot(θ)*dYdϕ(l,m,θ,ϕ)
end

function Wlm(l::Int, m::Int, θ::T, ϕ::T)::Complex{T} where {T}
    return (-l*(l+1)*ScalarSH(l,m,θ,ϕ) - 2*cot(θ)*dYdθ(l,m,θ,ϕ) 
            + ((2*m^2)/(sin(θ)^2))*ScalarSH(l,m,θ,ϕ))
end

function TensorSHE(a::Int, b::Int, l::Int, m::Int, θ::T, ϕ::T)::Complex{T} where {T}
    if a == b == 1
        return  (1/2)*Wlm(l,m,θ,ϕ) 
    elseif a == b == 2
        return -(1/2)*(sin(θ)^2)*Wlm(l,m,θ,ϕ) 
    else
        return  (1/2)*Xlm(l,m,θ,ϕ) 
    end
end

function TensorSHO(a::Int, b::Int, l::Int, m::Int, θ::T, ϕ::T)::Complex{T} where {T}
    if a == b == 1
        return -(1/(2*sin(θ)))*Xlm(l,m,θ,ϕ) 
    elseif a == b == 2
        return  (1/(2*sin(θ)))*(sin(θ)^2)*Xlm(l,m,θ,ϕ) 
    else
        return  (1/(2*sin(θ)))*(sin(θ)^2)*Wlm(l,m,θ,ϕ) 
    end
end
