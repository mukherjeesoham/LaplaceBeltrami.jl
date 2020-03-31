#---------------------------------------------------------------
# LaplaceOnASphere
# Soham 3/20
# Types for Spherical Harmonics
#---------------------------------------------------------------

using PyPlot
export contourf

function PyPlot. contourf(S1::SphericalHarmonics{T}, u::Array{Complex{T}, 1}) where {T}
    S2  = SphericalHarmonics{Float64}(S1.lmax, 100)
    v   = modal_to_nodal_scalar_op(S2)*(nodal_to_modal_scalar_op(S1)*u)
    θ   = collect(range(0, stop=π,  length=100))
    ϕ   = collect(range(0, stop=2π, length=200))
    contourf(ϕ, θ, reshape(real.(v), (100, 200)))
    colorbar()
end

