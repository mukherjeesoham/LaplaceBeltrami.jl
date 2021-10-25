#-----------------------------------------------------
# Compute the gradient and divergence using Cartesian components. We take the
# derivatives using scalar spherical harmonics.
# Soham 08/2021
#-----------------------------------------------------

using StaticArrays
export cartesian, S4, S5

function xyz_of_rθϕ(X::Array{T,1}) where {T<:Real} 
    r, μ, ν = X
    x = r*sin(μ)*cos(ν) 
    y = r*sin(μ)*sin(ν) 
    z = r*cos(μ)
    return SVector{3}([x,y,z])
end

function jacobian_xyz_of_rθϕ(μ::T, ν::T) where {T<:Real}
    r = 1
    return SMatrix{3,3}(ForwardDiff.jacobian(xyz_of_rθϕ, [r,μ,ν]))
end

function jacobian_rθϕ_of_xyz(μ::T, ν::T) where {T<:Real}
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
    @assert symbol == :Cartesian
    ∇F  = grad(spinsph_transform(F, 0), lmax) 
    dFdθ = map(x->x[1], ∇F) 
    dFdϕ = map(x->x[2], ∇F) 

    # Unpack the Jacobian
    J    = map(jacobian_rθϕ_of_xyz, lmax)
    dθdx = map(x->x[2,1], J)
    dθdy = map(x->x[2,2], J)
    dθdz = map(x->x[2,3], J)
    dϕdx = map(x->x[3,1], J)
    dϕdy = map(x->x[3,2], J)
    dϕdz = map(x->x[3,3], J)

    # Convert to Cartesian components
    dFdx = dθdx .* dFdθ + dϕdx .* dFdϕ 
    dFdy = dθdy .* dFdθ + dϕdy .* dFdϕ 
    dFdz = dθdz .* dFdθ + dϕdz .* dFdϕ 

    # Pack into a 3 vector
    return map((x,y,z)->SVector{3}(x,y,z), dFdx, dFdy, dFdz)
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
