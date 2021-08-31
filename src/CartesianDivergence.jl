#-----------------------------------------------------
# Compute divergence in Cartesian coordinates
# Soham 08/2021
#-----------------------------------------------------

using StaticArrays

function xyz_of_rθϕ(X::Array{T,1}) where {T<:Real} 
    r, μ, ν = X
    x = r*sin(μ)*cos(ν) 
    y = r*sin(μ)*sin(ν) 
    z = r*cos(μ)
    return SVector{3}([x,y,z])
end

function jacobian_xyz_of_rθϕ(μ::T, ν::T) where {T<:Real}
    return SMatrix{3,3}(ForwardDiff.jacobian(xyz_of_rθϕ, [1,μ,ν]))
end

function jacobian_rθϕ_of_xyz(μ::T, ν::T) where {T<:Real}
    return inv(jacobian_xyz_of_rθϕ(μ,ν))
end

function Base. div(F¹::AbstractMatrix{SVector{2, T}}, lmax::Int, string::String) where {T<:Real}
    @assert string == "Cartesian"
    M  = map(matrix, lmax)
    # F¹ = SVector{2}.(inv.(M) .* F¹)

    # Compute Jacobian
    J  = map(jacobian_rθϕ_of_xyz, lmax)
    
    dθdx = map(x->x[2,1], J)
    dθdy = map(x->x[2,2], J)
    dθdz = map(x->x[2,3], J)
    dϕdx = map(x->x[3,1], J)
    dϕdy = map(x->x[3,2], J)
    dϕdz = map(x->x[3,3], J)

    Fθ = map(x->x[1], F¹) 
    Fϕ = map(x->x[2], F¹)

    # Convert to Cartesian components
    Fx = dθdx .* Fθ + dϕdx .* Fϕ 
    Fy = dθdy .* Fθ + dϕdy .* Fϕ 
    Fz = dθdz .* Fθ + dϕdz .* Fϕ 

    # Take derivatives of Cartesian components.
    dFx = grad(spinsph_transform(Fx, 0), lmax) 
    dFy = grad(spinsph_transform(Fy, 0), lmax) 
    dFz = grad(spinsph_transform(Fz, 0), lmax)

    dFxdθ = map(x->x[1], dFx) 
    dFxdϕ = map(x->x[2], dFx) 
    dFydθ = map(x->x[1], dFy) 
    dFydϕ = map(x->x[2], dFy) 
    dFzdθ = map(x->x[1], dFz) 
    dFzdϕ = map(x->x[2], dFz) 

    # Note that using chain rule here to instead compute derivatives of Fθ and
    # Fϕ is not what we want. However, maybe it's wise to check if it works,
    # since if we want to handle the sinθ terms analytically, then this is
    # necessary.
    # TODO: Should we redefine the dϕ derivatives? Will they be more accurate
    # with a sinθ in there?
    # dFθ = gradbar(spinsph_transform(Fθ, 0), lmax) 
    # dFϕ = gradbar(spinsph_transform(Fϕ, 0), lmax) 
    # dFθdθ = map(x->x[1], dFθ) 
    # dFθdϕ = map(x->x[2], dFθ) 
    # dFϕdθ = map(x->x[1], dFϕ) 
    # dFϕdϕ = map(x->x[2], dFϕ) 

    # dFxdθ = ddθdxdθ .* Fθ + dθdx .* dFθdθ + ddϕdxdθ .* Fϕ + dϕdx .*dFϕdθ 
    # dFxdθ = ddθdxdϕ .* Fθ + dθdx .* dFθdϕ + ddϕdxdϕ .* Fϕ + dϕdx .*dFϕdϕ 
    # dFydθ = ddθdydθ .* Fθ + dθdy .* dFθdθ + ddϕdydθ .* Fϕ + dϕdy .*dFϕdθ 
    # dFydθ = ddθdydϕ .* Fθ + dθdy .* dFθdϕ + ddϕdydϕ .* Fϕ + dϕdy .*dFϕdϕ 
    # dFzdθ = ddθdzdθ .* Fθ + dθdz .* dFθdθ + ddϕdzdθ .* Fϕ + dϕdz .*dFϕdθ 
    # dFzdθ = ddθdzdϕ .* Fθ + dθdz .* dFθdϕ + ddϕdzdϕ .* Fϕ + dϕdz .*dFϕdϕ 

    dFxdx = dFxdθ .* dθdx + dFxdϕ .* dϕdx
    dFydy = dFydθ .* dθdy + dFydϕ .* dϕdy
    dFzdz = dFzdθ .* dθdz + dFzdϕ .* dϕdz

    return dFxdx + dFydy + dFzdz
end
