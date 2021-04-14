#---------------------------------------------------------------
# LaplaceOnASphere
# Soham 8/2020
# Choose a smooth coodinate transformation and compute 
# the associated metric
#---------------------------------------------------------------

export sqrt_deth_by_detq_q_hinv, sqrt_detq_by_deth
export deth, detq, analyticF, q_hinv, q, hinv

function Z(μ::T, ν::T)::T where {T}
    z = sqrt(4π/3)*ScalarSH(1,0,μ,ν) + (1/10)*(sqrt(4π/7)*ScalarSH(3,0,μ,ν) - sqrt(4π/11)*ScalarSH(5,0,μ,ν)) 
    return z
end

function theta(μ::T, ν::T)::T where {T}
    x = sin(μ)*cos(ν) 
    y = sin(μ)*sin(ν) 
    z = Z(μ,ν)
    return acos(z/sqrt(x^2 + y^2 + z^2))
end

function analyticF(l::Int, m::Int, μ::T, ν::T)::Complex{T} where {T} 
    return ScalarSH(l, m, theta(μ, ν), ν)
end

function qinv(a::Int, b::Int, μ::T, ν::T)::T where {T}
    if a == b == 1
        return 1
    elseif a == b == 2
        return 1/sin(μ)^2 
    else 
        return 0
    end
end

function q(a::Int, b::Int, μ::T, ν::T)::T where {T}
    if a == b == 1
        return 1
    elseif a == b == 2
        return sin(μ)^2 
    else 
        return 0
    end
end

function detq(μ::T, ν::T)::T where {T}
    return sin(μ)^2
end

function g(a::Int, b::Int, μ::T, ν::T)::T where {T}
    if a == b == 1
        return 1
    elseif a == b == 2
        θ = theta(μ, ν) 
        return sin(θ)^2 
    else 
        return 0
    end
end
    
function hinv(a::Int, b::Int, μ::T, ν::T)::T where {T}
    # TODO: Test hinv for accuracy.
    d1d1 = ((5120*(649 + 45*cos(2μ) - 117*cos(4μ) + 63*cos(6μ)))/(3329198 + 157554*cos(2μ) - 46728*cos(4μ) - 161523*cos(6μ) - 5670*cos(8μ) + 3969*cos(10μ))) 
    d1d2 = 0 
    d2d1 = 0 
    d2d2 = 1

    g11  = g(1,1,μ,ν)
    g22  = g(2,2,μ,ν)
    g12  = g21 = g(1,2,μ,ν)
    
    h11  = d1d1*d1d1*g11 + d1d1*d2d1*g12 + d2d1*d1d1*g21 + d2d1*d2d1*g22 
    h22  = d1d2*d1d2*g11 + d1d2*d2d2*g12 + d2d2*d1d2*g21 + d2d2*d2d2*g22 
    h12  = d1d1*d1d2*g11 + d1d1*d2d2*g12 + d2d1*d1d2*g21 + d2d1*d2d2*g22 
    hinv = inv([h11 h12; h12 h22])

    return hinv[a,b]
end

function q_hinv(a::Int, b::Int, μ::T, ν::T)::T where {T}
    return q(a,1,μ,ν)*hinv(1,b,μ,ν) + q(a,2,μ,ν)*hinv(2,b,μ,ν)
end

function deth(μ::T, ν::T)::T where {T}
    hinvmat = [hinv(1,1,μ,ν) hinv(1,2,μ,ν); 
               hinv(2,1,μ,ν) hinv(2,2,μ,ν)]
    return 1/det(hinvmat)
end

function sqrt_detq_by_deth(μ::T, ν::T)::T where {T}
    return sqrt(detq(μ,ν)/deth(μ,ν)) 
end

function sqrt_deth_by_detq_q_hinv(a::Int, b::Int, μ::T, ν::T)::T where {T}
    return (1/sqrt_detq_by_deth(μ, ν))*q_hinv(a,b,μ,ν) 
end

function hinv(a::Int, b::Int, μ::T, ν::T)::T where {T}
    # TODO: Test hinv for accuracy.
    d1d1 = ((5120*(649 + 45*cos(2μ) - 117*cos(4μ) + 63*cos(6μ)))/(3329198 + 157554*cos(2μ) - 46728*cos(4μ) - 161523*cos(6μ) - 5670*cos(8μ) + 3969*cos(10μ))) 
    d1d2 = 0 
    d2d1 = 0 
    d2d2 = 1

    g11  = g(1,1,μ,ν)
    g22  = g(2,2,μ,ν)
    g12  = g21 = g(1,2,μ,ν)
    
    h11  = d1d1*d1d1*g11 + d1d1*d2d1*g12 + d2d1*d1d1*g21 + d2d1*d2d1*g22 
    h22  = d1d2*d1d2*g11 + d1d2*d2d2*g12 + d2d2*d1d2*g21 + d2d2*d2d2*g22 
    h12  = d1d1*d1d2*g11 + d1d1*d2d2*g12 + d2d1*d1d2*g21 + d2d1*d2d2*g22 
    hinv = inv([h11 h12; h12 h22])

    return hinv[a,b]
end
