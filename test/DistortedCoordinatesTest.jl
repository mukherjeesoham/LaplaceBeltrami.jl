#-----------------------------------------------------
# Test whether the the coordinate transformation
# preserves the area of the sphere independent using
# quadraature routines from FastSphericalTransforms
# Soham 11/2021
#-----------------------------------------------------

using Test, StaticArrays, LinearAlgebra

function Ω(X::AbstractVector)
    return X[2:end]
end

function q_θϕ(θϕ::AbstractVector)
    (θ, ϕ) = θϕ 
    return SMatrix{2,2}([1.0 0.0;
                         0.0 sin(θ)^2])
end

function dS(uvw::AbstractVector)
    return (sqrt ∘ det ∘ q_μν ∘ Ω ∘ cartesian2spherical)(uvw) / (sqrt ∘ det ∘ q_θϕ ∘ Ω ∘ cartesian2spherical)(uvw)
end

# Do some very basic tests
x = rand(3) # Cartesian coordinates
r = rand(3) # Spherical (good) coordinates
s = rand(3) # Spherical (bad) coordinates
@test (uvw_of_xyz ∘ xyz_of_uvw)(x) ≈ x 
@test (rθϕ_of_sμν ∘ sμν_of_rθϕ)(r) ≈ r 
@test jacobian(sμν_of_rθϕ,  rθϕ_of_sμν(s)) ≈ inv(jacobian(rθϕ_of_sμν,  s))

@test quad(dS, :sphericaldesigns) ≈ 4π
# FIXME: The general coordinate transformation still doesn't work.
# [1] Compute the Ricci scalar for the 3D and the 2D metric. 
# [2] Only do a 2D transformation. This removes the need for projecting
# the metric with a surface normal.
# [3] Are these two equivalent? Check using Mathematica?
