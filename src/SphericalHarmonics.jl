#---------------------------------------------------------------
# LaplaceOnASphere
# Soham 3/20
# Compute scalar and tensor spherical harmonics
# Ψ = ∇Y
# Φ = e^b_a ∇Y
#---------------------------------------------------------------

using SHTOOLS
export Ylm, Glm, Clm  
export dYdθ, dYdϕ

function Plm(l::Int, m::Int, x::Real)::Real 
    p, dp = PlmBar_d1(l, x)
    index = PlmIndex(l,m)
    return p[index]
end

function dPlm(l::Int, m::Int, x::Real)::Real
    p, dp = PlmBar_d1(l, x)
    index = PlmIndex(l,m)
    return dp[index]
end

function Ylm(l::Int, m::Int, θ::Real, ϕ::Real)::Real
    if m >= 0
        return Plm(l,m,cos(θ))*cos(m*ϕ) 
    else
        return Plm(l,abs(m),cos(θ))*sin(abs(m)*ϕ) 
    end
end

function dYdθ(l::Int, m::Int, θ::Real, ϕ::Real)::Real
    if m >= 0
        return -sin(θ)*dPlm(l,m,cos(θ))*cos(m*ϕ) 
    else
        return -sin(θ)*dPlm(l,abs(m),cos(θ))*sin(abs(m)*ϕ) 
    end
end

function dYdϕ(l::Int, m::Int, θ::Real, ϕ::Real)::Real
    if m >= 0
        return -m*Plm(l,m,cos(θ))*sin(m*ϕ) 
    else
        return abs(m)*Plm(l,abs(m),cos(θ))*cos(abs(m)*ϕ) 
    end
end

function Glm(a::Int, l::Int, m::Int, θ::Real, ϕ::Real)::Real
    if a  == 1
        return dYdθ(l,m,θ,ϕ)
    elseif a == 2
        return dYdϕ(l,m,θ,ϕ)
    end
end

function Clm(a::Int, l::Int, m::Int, θ::Real, ϕ::Real)::Real
    if a  == 1
        return +(1/sin(θ))*dYdϕ(l,m,θ,ϕ)
    elseif  a == 2
        return -sin(θ)*dYdθ(l,m,θ,ϕ)
    end
end
