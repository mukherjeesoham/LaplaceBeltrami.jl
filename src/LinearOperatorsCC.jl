#-----------------------------------------------------
# Compute the gradient and divergence using Cartesian components. We take the
# derivatives using scalar spherical harmonics.
# Soham 08/2021
#-----------------------------------------------------

using StaticArrays
export cartesian, S4, S5, jacobian_xyz_of_rθϕ

function xyz_of_rθϕ(X::Array{T,1}) where {T<:Real} 
    r, μ, ν = X
    x = r*sin(μ)*cos(ν) 
    y = r*sin(μ)*sin(ν) 
    z = r*cos(μ)
    return SVector{3}([x,y,z])
end

function jacobian_xyz_of_rθϕ(μ::T, ν::T) where {T<:Real}
    # FIXME: Test jacobian thoroughly
    return SMatrix{3,3}(ForwardDiff.jacobian(xyz_of_rθϕ, [1,μ,ν]))
end

function jacobian_rθϕ_of_xyz(μ::T, ν::T) where {T<:Real}
    # FIXME: You might need a transpose here.
    return inv(jacobian_xyz_of_rθϕ(μ,ν))
end

function cartesian(metric::Function, μ::T, ν::T) where {T<:Real}
    # Invent the r coordinate.
    M = diagm([1.0,0.0,0.0])
    M[2:end, 2:end] = Array(metric(μ,ν))
    J = jacobian_rθϕ_of_xyz(μ,ν)
    # Transform into Cartesian components
    h  =  (J * M * J')
    return SMatrix{3,3}(h)
end

function grad(F::AbstractMatrix{T}, lmax::Int, symbol::Symbol) where {T <: Real}
    @assert symbol == :Spherical
    ∇F = grad(spinsph_transform(F, 0), lmax) 
    # FIXME: Setting the first component to zero won't work.
    dF = map(x->SVector{3}([0.0, x...]),  ∇F)
    return dF
end

function grad(F::AbstractMatrix{T}, lmax::Int, symbol::Symbol) where {T <: Real}
    @assert symbol == :Cartesian
    ∇F = grad(spinsph_transform(F, 0), lmax) 
    dF = map(x->SVector{3}([0.0, x...]),  ∇F)
    # TODO: Is the Jacobian correct? 
    # Let's expand this out.
    J  = map(jacobian_rθϕ_of_xyz, lmax)

    dFdr = map(x->x[1], dF)
    dFdθ = map(x->x[2], dF)
    dFdϕ = map(x->x[3], dF)

    drdx = map((μ,ν)->cos(ν)*sin(μ), lmax)
    dθdx = map((μ,ν)->cos(ν)*cos(μ), lmax)
    dϕdx = map((μ,ν)->cos(μ), lmax)


    dFdx = dFdr .* drdx + dFdθ .* dθdx  + dFdϕ .* dϕdx 
    dFdy = dFdx
    dFdz = dFdx
    # dFdy = dFdr .* drdy + dFdθ .* dθdy  + dFdϕ .* dϕdy 
    # dFdz = dFdr .* drdz + dFdθ .* dθdz  + dFdϕ .* dϕdz 

    return map((x,y,z)->SVector{3}(x, y, z), dFdx, dFdy, dFdz)  
end


function Base. div(F::AbstractMatrix{SVector{3, T}}, lmax::Int, symbol::Symbol) where {T<:Real}
    @assert symbol == :Cartesian

    # Extract the vector components
    Fx = map(x->x[1], F) 
    Fy = map(x->x[2], F) 
    Fz = map(x->x[3], F)

    # Take gradients of the Cartesian components 
    dFx = grad(Fx, lmax, :Cartesian) 
    dFy = grad(Fy, lmax, :Cartesian) 
    dFz = grad(Fz, lmax, :Cartesian) 

    # Extract the components again 
    dFxdx = map(x->x[1], dFx) 
    dFydy = map(x->x[2], dFy) 
    dFzdz = map(x->x[3], dFz) 

    return dFxdx + dFydy + dFzdz
end

# FIXME: Check scaling with FD operators
function S4(q::AbstractMatrix{T}, h::AbstractMatrix{T}, F¹::AbstractVector{T}) where {T<:Real}
    return sqrt(det(h)) .* raise(inv(h), F¹)
end

function S5(q::AbstractMatrix{T}, h::AbstractMatrix{T}, F⁰::T) where {T<:Real}
    return sqrt(1 / det(h)) .* F⁰
end
