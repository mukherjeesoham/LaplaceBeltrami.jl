#-----------------------------------------------------
# Test whether the the coordinate transformation
# preserves the area of the sphere independent using
# quadraature routines from FastSphericalTransforms
# Soham 11/2021
#-----------------------------------------------------

using LinearAlgebra, Plots, FastGaussQuadrature, DelimitedFiles, ForwardDiff

function sqrtdetq2D(X::AbstractVector)
    R = cartesian2spherical(X...)
    R = R[2:3]
    return sqrt(det(q(R...)))  /  sqrt(det(q(R...)))
end

function sqrtdeth2D(X::AbstractVector)
    R = cartesian2spherical(X...)
    R = R[2:3]
    return sqrt(det(h(R...))) /  sqrt(det(q(R...)))
end


@show quad(sqrtdetq2D, :sphericaldesigns)
@show quad(sqrtdeth2D, :sphericaldesigns)
