#---------------------------------------------------------------
# LaplaceOnASphere
# Soham 6/2020
# Compute a conformal coordinate transformation on the
# sphere. See <Bensten et al. 1998>. We use a different
# convention for the coordinates.
#---------------------------------------------------------------

function Z(θ::T, ϕ::T)::Complex{T} where {T<:Real}
    # Project on the complex plane
    return tan(θ/2)*cis(ϕ)
end

function R(z::Complex{T})::NTuple{2,T} where {T<:Real}
    # Project onto the sphere from the complex plane
    return (2*atan(abs(z)), angle(z))
end

function LFT(w::Complex{T})::Complex{T} where {T<:Real}
    # Do the linear fractional transformation on 
    # the complex plane.
    θa, ϕa = ( π/3, 0.0)  # North pole
    θb, ϕb = (2π/3, 0.0)  # South pole

    # Construct the geodesic midpoint
    cx = cos(ϕa)*sin(θa) + cos(ϕb)*sin(θb)
    cy = sin(ϕa)*sin(θa) + sin(ϕb)*sin(θb)
    cz = cos(θa) + cos(θb)
    θc, ϕc = (atan(sqrt(cx^2 + cy^2),cz), atan(cy,cx))  

    a = Z(θa, ϕa) 
    b = Z(θb, ϕb) 
    c = Z(θc, ϕc) 
    
    return (-b*w*(c-a) + a*(c-b))/(-w*(c-a) + (c-b))
end

function θ(μ::T, ν::T)::T where {T<:Real}
    θ̄, ϕ̄ = R(LFT(Z(μ, ν)))
    return θ̄ 
end

function ϕ(μ::T, ν::T)::T where {T<:Real}
    θ̄, ϕ̄ = R(LFT(Z(μ, ν)))
    return ϕ̄ 
end

function hinv(a::Int, b::Int, μ::T, ν::T)::Complex{T} where {T}
    # FIXME: Find a way to test this, or compute the metric analytically in
    # Mathematica.
    θlm = nodal_to_modal_scalar_op(SH)*map(SH, (μ,ν)->θ(μ,ν))
    ϕlm = nodal_to_modal_scalar_op(SH)*map(SH, (μ,ν)->ϕ(μ,ν))
    dθdμ, dθdν = (analyticΨlm(SH, θlm, μ, ν, 1), analyticΨlm(SH, θlm, μ, ν, 2))
    dϕdμ, dϕdν = (analyticΨlm(SH, ϕlm, μ, ν, 1), analyticΨlm(SH, ϕlm, μ, ν, 2))

    gθθ = 1
    gθϕ = gϕθ = 0
    gϕϕ = sin(θ(μ,ν))^2

    hμμ = dθdμ*dθdμ*gθθ + dθdμ*dϕdμ*gθϕ + dϕdμ*dθdμ*gϕθ + dϕdμ*dϕdμ*gϕϕ
    hμν = dθdμ*dθdν*gθθ + dθdμ*dϕdν*gθϕ + dϕdμ*dθdν*gϕθ + dϕdμ*dϕdν*gϕϕ
    hνν = dθdν*dθdν*gθθ + dθdν*dϕdν*gθϕ + dϕdν*dθdν*gϕθ + dϕdν*dϕdν*gϕϕ

    h   = inv([hμμ hμν; hμν hνν])

    return h[a, b]
end
