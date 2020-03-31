#---------------------------------------------------------------
# LaplaceOnASphere
# Soham 3/20
# Types for Spherical Harmonics
#---------------------------------------------------------------
export SphericalHarmonics

struct SphericalHarmonics{T}
    lmax::Int
    N::Int
    SphericalHarmonics{T}(lmax, N) where {T} = (N < 2lmax + 1) ? @error("You need more points") : new{T}(lmax, N)
end

SphericalHarmonics(l::Int) = SphericalHarmonics{Float64}(l, 2l+1)
