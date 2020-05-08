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
for l in 4:lmax
    S = SphericalHarmonics(l)
    u = map(S, (θ, ϕ)->ScalarSPH(2, 0, θ, ϕ))
    L = nodal_to_modal_scalar_op(S)
    P = modal_to_nodal_scalar_op(S)
    @test L1(u - P*(L*u)) < 1e-9
    ũ = L*u
    @test L1(ũ - L*(P*ũ)) < 1e-12
end
