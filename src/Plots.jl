#---------------------------------------------------------------
# Use spin-weighted spherical harmonics to compute the 
# Laplace operator using FastSphericalHarmonics.jl
# Soham M 03/21
#---------------------------------------------------------------

using CairoMake, FastSphericalHarmonics 

# function plot(u1::Array{T,2}, u2::Array{T,2}, u3::Array{T,2}, lmax::Int, string::String) where {T}
    # N = lmax + 1
    # latglq, longlq = sph_points(N) 
    # @time begin
        # fig = Figure(resolution = (1600, 400))
        # _, co1 = contourf(fig[1, 1][1, 1], longlq, latglq, u1', levels = 10)
        # Colorbar(fig[1, 1][1, 2], co1, width = 20)
        # _, co2 = contourf(fig[1, 2][1, 1], longlq, latglq, u2', levels = 10)
        # Colorbar(fig[1, 2][1, 2], co2, width = 20)
        # _, co3 = contourf(fig[1, 3][1, 1], longlq, latglq, u3', levels = 10)
        # Colorbar(fig[1, 3][1, 2], co3, width = 20)
        # save("output/$string.pdf", fig)
    # end
# end
