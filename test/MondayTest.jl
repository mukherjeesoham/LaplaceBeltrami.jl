#---------------------------------------------------------------
# LaplaceOnASphere
# Soham 7/20
# Check the accuracy of the spherical harmonics by computing
# [1] Orthogonality relations
# [2] Check for aliasing (use many more points)
#---------------------------------------------------------------

using GSL, FastGaussQuadrature, LinearAlgebra, PyPlot, Distributed
addprocs(4)

function gaussquad(::Type{T}, f::Function, N::Int)::T where {T}
    u = zeros(Complex{T}, N, 2N) 
    nodes, weights = gausslegendre(N)
    theta = -(T(π)/2).*nodes .+ T(π)/2  
    phi   = [(j-1)*(T(π)/N) for j in 1:2N] 
    for index in CartesianIndices(u)    
        i, j = Tuple(index) 
        u[index] = (2/(2N))*weights[i]*f(theta[i], phi[j])*sin(theta[i]) 
    end
    integral = π*(π/2)*sum(u)
    @assert imag(integral) < 1e-12
    return real(integral)
end

f(μ, ν) = ScalarSPH(0,0,μ,ν)
@test isapprox(gaussquad(Float64, f, 100), 2*√π)

function kroneckerdelta(l::Int, l̄::Int)::Int
    if l == l̄
        return 1
    else
        return 0
    end
end

# Test orthogonality of Scalar spherical harmonics
lmax = 40
Threads.@threads for l in 0:lmax
    for l̄ in 0:lmax, m in -l:l, m̄ in -l̄:l̄
        @show l,l̄, m, m̄
        yȳ(μ, ν) = ScalarSPH(l, m, μ, ν)*conj(ScalarSPH(l̄, m̄, μ, ν))
        @test isapprox(gaussquad(Float64, yȳ, 100), kroneckerdelta(l,l̄)*kroneckerdelta(m,m̄), atol=1e-12)
    end
end
