#---------------------------------------------------------------
# LaplaceOnASphere
# Soham 3/20
# Test basis transformations and operators
#---------------------------------------------------------------
using PyPlot, LinearAlgebra

if false
    l = 10
    projection_error = zeros(l, 2l+1)
    for index in CartesianIndices(projection_error)
        l, N = index.I
        S = SphericalHarmonics{Float64}(l, N+1) 
        u = map(S, (θ, ϕ)->cos(θ)^3)
        L = nodal_to_modal_scalar_op(S)
        P = modal_to_nodal_scalar_op(S)
        projection_error[index]= log10(L1(u - P*(L*u)))
    end
    imshow(projection_error,  origin="upper")
    colorbar()
    savefig("./output/ScalarSPH_projection_error_cos3.pdf")
    close()
end

lmax = 10
for l in 0:lmax
    @show l
    S = SphericalHarmonics{Float64}(l, 2l+1)
    L = nodal_to_modal_scalar_op(S)
    P = modal_to_nodal_scalar_op(S)
    @test isless(maximum(abs.(L*P - I)), 1e-12)
    @test P*L ≈ (P*L)^2 
    L̄ = nodal_to_modal_vector_op(S)
    P̄ = modal_to_nodal_vector_op(S)
    @test isless(maximum(abs.(L*P - I)), 1e-12)
    @test P*L ≈ (P*L)^2 
end

for l in 4:lmax
    @show l
    S = SphericalHarmonics{Float64}(l, 2l+1) 
    u = map(S, (θ, ϕ)->ScalarSPH(2, 0, θ, ϕ))
    L = nodal_to_modal_scalar_op(S)
    P = modal_to_nodal_scalar_op(S)
    @test L1(u - P*(L*u)) < 1e-9
    ũ = L*u
    @test L1(ũ - L*(P*ũ)) < 1e-12
end
