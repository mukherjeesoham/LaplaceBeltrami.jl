using FastTransforms, GLMakie, Random, FileIO

Random.seed!(0)

n = 64
θ = [0;(0.5:n-0.5)/n;1]
φ = [(0:2n-2)*2/(2n-1);2]
x = Float32[cospi(φ)*sinpi(θ) for θ in θ, φ in φ]
y = Float32[sinpi(φ)*sinpi(θ) for θ in θ, φ in φ]
z = Float32[cospi(θ) for θ in θ, φ in φ]

# Function on the sphere
u = Float32[cospi(sinpi(θ)) for θ in θ, φ in φ]

camera = 
scene = Scene(resolution = (1200, 1200));
surf  = surface!(scene, x, y, z, color = u, colormap = :viridis, colorrange = (-1.0, 1.0));
rotate_cam!(scene, (0, 0, π/2))
# TODO: Figure out how to save the plot.
# S*&ew this. Do a polar plot instead. 
save("output/sphere.png", surf)
