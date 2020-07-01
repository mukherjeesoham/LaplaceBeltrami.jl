#---------------------------------------------------------------
# LaplaceOnASphere
# Soham 3/20
# Types for Spherical Harmonics
#---------------------------------------------------------------

using PyPlot
export contourf, plot

function PyPlot. contourf(S1::SphericalHarmonics{T}, u::Array{Complex{T}, 1}, interpolate::Bool) where {T}
    println("We're in the right function")
    if interpolate
        S2  = SphericalHarmonics{Float64}(S1.lmax, 100)
        v   = modal_to_nodal_scalar_op(S2)*(nodal_to_modal_scalar_op(S1)*u)
        θ   = collect(range(0, stop=π,  length=100))
        ϕ   = collect(range(0, stop=2π, length=200))
        contourf(ϕ, θ, reshape(S2, v))
        colorbar()
    else 
        θ   = collect(range(0, stop=π,  length=S1.N))
        ϕ   = collect(range(0, stop=2π, length=2*(S1.N)))
        @show θ
        contourf(ϕ, θ, reshape(S1, u))
        colorbar()
    end
end

function PyPlot.plot(SH::SphericalHarmonics, F::Eigen, l::Int)
    amax = l^2 + 2l + 1
    plot(sort(abs.(F.values))[1:amax], "r-o")
    savefig("./output/eigenvals.pdf")
    close()
end

function PyPlot. contourf(SH::SphericalHarmonics, F::Eigen, l::Int)
    V = query(SH, F, l, :nodal)
    θ = collect(range(0, stop=π,  length=SH.N))
    ϕ = collect(range(0, stop=2π, length=2*(SH.N)))

    figure(figsize = ((2l+1)*5, 5))
    for lind in 1:2l+1 
        subplot(1, 2l+1, lind)   
        contourf(ϕ, θ, reshape(SH, real.(V[:,lind])))
        colorbar()
    end
    savefig("./output/$l-real.pdf")
    tight_layout()
    close()

    figure(figsize = ((2l+1)*5, 5))
    for lind in 1:2l+1 
        subplot(1, 2l+1, lind)   
        contourf(ϕ, θ, reshape(SH, imag.(V[:,lind])))
        colorbar()
    end
    savefig("./output/$l-imag.pdf")
    tight_layout()
    close()
end
