#---------------------------------------------------------------
#e LaplaceOnASphere
# Soham 8/20
# Construct a smooth z-coordinate using
# spherical harmonics
#---------------------------------------------------------------

using LinearAlgebra, PyPlot, FastGaussQuadrature

SH = SphericalHarmonics(12)
@time S, S̄ = scalar_op(SH)
@time V, V̄ = vector_op(SH)
@time C, C̄ = curl_op(SH)

function ScalarSHF(SH, l::Int, m::Int)
    return map(SH, (μ,ν)->ScalarSH(l,m,μ,ν))
end

function GradSHF(SH::SphericalHarmonics{T}, l::Int, m::Int) where {T}
    return map(SH, (μ,ν)->GradSH(1,l,m,μ,ν), (μ,ν)->GradSH(2,l,m,μ,ν))
end

function CurlSHF(SH::SphericalHarmonics{T}, l::Int, m::Int) where {T}
    return map(SH, (μ,ν)->CurlSH(1,l,m,μ,ν), (μ,ν)->CurlSH(2,l,m,μ,ν))
end

# Here's the test we want to do. Start with the superposition of gradient and
# curl vector spherical harmonics. Project onto gradient spherical harmonics;
# subtract from the superposition to get exactly the curl spherical harmonics
# FIXME: We have a problem! While all the orthogonality checks pass for the
# orthogonality between curl and the gradient vector spherical harmonics, we
# see that our pinverse operators are not quite orthogonal. This could be due
# to several reasons
#  -- Incorrect expression for the curl spherical harmonics
#  -- The p-inverse operator doesn't capture the projection correctly. You wouldd
#     need to raise the index and take a conjugate for this to work correctly. 
#  -- Unstable recurrence relations
w = GradSHF(SH,2,2) + CurlSHF(SH,2,2) + CurlSHF(SH,3,2)

# How to fix the operators? They are borked. 
# Let's try raising the index first. Raising the index doesn't help.
# w = map(SH, (μ,ν)->GradSH(1,2,2,μ,ν) + CurlSH(1,2,2,μ,ν) + CurlSH(1,2,2,μ,ν),
        # (μ,ν)->(1/sin(μ)^2)*(GradSH(2,2,2,μ,ν) + CurlSH(2,2,2,μ,ν) + CurlSH(2,2,2,μ,ν)))
# And conjugating the operators
# C̄ = conj.(C̄)
# V̄ = conj.(V̄)

wlm_curl = C̄*w 
wlm_grad = V̄*w 

for index in CartesianIndices(wlm_grad)
    if abs(wlm_grad[index]) > 1e-10
        l, m = split(index.I[1])
        @show l, m, wlm_grad[index]
    end
end

println()

for index in CartesianIndices(wlm_curl)
    if abs(wlm_curl[index]) > 1e-10
        l, m = split(index.I[1])
        @show l, m, wlm_curl[index]
    end
end

# Let's try integration. This should confirm if the problem is with
# the pseudo-inverse operator or the curl vector spherical harmonic.

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
    @assert abs(imag(integral)) < 1e-10
    return real(integral)
end

wlm_curl_gaussquad = zeros(size(wlm_curl))
wlm_grad_gaussquad = zeros(size(wlm_grad))

for index in CartesianIndices(wlm_curl_gaussquad)
    l, m = split(index.I[1])
    g(a,μ,ν) = GradSH(a,2,2,μ,ν) + CurlSH(a,2,2,μ,ν) + CurlSH(a,3,2,μ,ν)
    ff(μ,ν)  = g(1,μ,ν)*conj(CurlSH(1, l, m, μ, ν)) + g(2,μ,ν)*conj(CurlSH(2, l, m, μ, ν))*(1/sin(μ)^2)
    gg(μ,ν)  = g(1,μ,ν)*conj(GradSH(1, l, m, μ, ν)) + g(2,μ,ν)*conj(GradSH(2, l, m, μ, ν))*(1/sin(μ)^2)
    wlm_curl_gaussquad[index] = (1/(l*(l+1)))*gaussquad(Float64, ff, 200)
    wlm_grad_gaussquad[index] = (1/(l*(l+1)))*gaussquad(Float64, gg, 200)
end

for index in CartesianIndices(wlm_curl_gaussquad)
    if abs(wlm_curl_gaussquad[index]) > 1e-10
        l, m = split(index.I[1])
        @show l, m, wlm_curl_gaussquad[index]
    end
end

println()

for index in CartesianIndices(wlm_curl_gaussquad)
    if abs(wlm_grad_gaussquad[index]) > 1e-10
        l, m = split(index.I[1])
        @show l, m, wlm_grad_gaussquad[index]
    end
end

