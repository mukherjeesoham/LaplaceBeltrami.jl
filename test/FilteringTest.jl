#-----------------------------------------------------
# Investigate filtering
# Soham 09/2021
#-----------------------------------------------------

using FastSphericalHarmonics, Plots

# TODO: [1] Test what the coordinate transformation is doing to the modes. Does
#       it introduce mode-mode coupling? 
#       [2] If so, filtering would help. Try a simpler non-linear operation
#       with filtering which you can test.  
#       [3] Modify parts of the Laplace operator and iterate by hand. 

lmax = 12
(l, m) = (4, 3)
reYlm(μ, ν) = sYlm(Real, 0, l, m, μ, ν) + sYlm(Real, 0, 4, 4, μ, ν) 

u0   = map(reYlm, lmax)
u0lm = spinsph_transform(u, 0) 

# Now do a coordinate transformation
u1   = map((μ, ν)->reYlm(θϕ_of_μν(μ,ν)...), lmax)
u1lm = spinsph_transform(u, 0) 

# Plot the modes [FIXME: This is bad an  confusing]
# plot(0:1:lmax, maximum(abs2, u0lm, dims=2), label="0")
# plot!(0:1:lmax, maximum(abs2, u1lm, dims=2), label="1")
heatmap(u0lm)
