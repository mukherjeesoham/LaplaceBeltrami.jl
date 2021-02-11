#---------------------------------------------------------------
# LaplaceOnASphere
# Soham 7/20
# Check the accuracy of the spherical harmonics by computing
# [1] Orthogonality relations
# [2] Check for aliasing (use many more points)
#---------------------------------------------------------------

using GSL, FastGaussQuadrature, LinearAlgebra, PyPlot

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

function integral(SH::SphericalHarmonics{T})

f(μ, ν) = ScalarSH(0,0,μ,ν)
@test isapprox(gaussquad(Float64, f, 100), 2*√π)

function kroneckerdelta(l::Int, l̄::Int)::Int
    if l == l̄
        return 1
    else
        return 0
    end
end

# Test orthogonality of Scalar spherical harmonics
if false
    Threads.@threads for l in 21:100
            for l̄ in 0:100
                @show l, l̄
                yȳ(μ, ν) = ScalarSH(l, 0, μ, ν)*conj(ScalarSH(l̄, 0, μ, ν))
                @test isapprox(gaussquad(Float64, yȳ, 200), kroneckerdelta(l,l̄), atol=1e-12)
            end
    end
end

# Test orthogonality of Vector (Grad) spherical harmonics
if false
    Threads.@threads for l in 50:100
            for l̄ in 50:100
                @show l, l̄
                vv̄(μ, ν) = GradSH(1, l, 0, μ, ν)*conj(GradSH(1, l̄, 0, μ, ν)) + GradSH(2, l, 0, μ, ν)*conj(GradSH(2, l̄, 0, μ, ν))*(1/sin(μ)^2)
                @test isapprox(gaussquad(Float64, vv̄, 200), l*(l+1)*kroneckerdelta(l,l̄), atol=1e-10)
            end
    end
end

# Test orthogonality of the scalar, grad and curl spherical harmonics for m ! = 0
if true
@testset "Orthonormality" begin
    Threads.@threads for l in 1:10
        for m in 0:l, l̄ in 1:10, m̄ in 0:l̄
                @show l, l̄, m, m̄
                ss̄(μ, ν) = ScalarSH(l, m, μ, ν)*conj(ScalarSH(l̄, m̄, μ, ν))
                vv̄(μ, ν) = GradSH(1, l, m, μ, ν)*conj(GradSH(1, l̄, m̄, μ, ν)) + GradSH(2, l, m, μ, ν)*conj(GradSH(2, l̄, m̄, μ, ν))*(1/sin(μ)^2)
                cc̄(μ, ν) = CurlSH(1, l, m, μ, ν)*conj(CurlSH(1, l̄, m̄, μ, ν)) + CurlSH(2, l, m, μ, ν)*conj(CurlSH(2, l̄, m̄, μ, ν))*(1/sin(μ)^2)
                vc̄(μ, ν) = GradSH(1, l, m, μ, ν)*conj(CurlSH(1, l̄, m̄, μ, ν)) + GradSH(2, l, m, μ, ν)*conj(CurlSH(2, l̄, m̄, μ, ν))*(1/sin(μ)^2)
                @test isapprox(gaussquad(Float64, ss̄, 200), kroneckerdelta(l,l̄)*kroneckerdelta(m, m̄), atol=1e-10)
                @test isapprox(gaussquad(Float64, vv̄, 200), l*(l+1)*kroneckerdelta(l,l̄)*kroneckerdelta(m, m̄), atol=1e-10)
                @test isapprox(gaussquad(Float64, cc̄, 200), l*(l+1)*kroneckerdelta(l,l̄)*kroneckerdelta(m, m̄), atol=1e-10)
                @test isapprox(gaussquad(Float64, vc̄, 200), 0, atol=1e-10)
            end
    end
end
end

exit()

SH      = SphericalHarmonics(20) 
# Let's also check for aliasing, for both scalar and vector spherical harmonics, and also compare
# pinv vs. gaussian quadrature

if false
    u(μ, ν) = ScalarSH(1,1,μ,ν) + 18*ScalarSH(18,4,μ,ν) + 39*ScalarSH(39,-39,μ,ν) + 71*ScalarSH(71,36,μ,ν)
    umap    = map(SH, u)
    ulm     = nodal_to_modal_scalar_op(SH)*umap
    
    ulm_exact = zeros(length(ulm))
    ulm_integ = zeros(length(ulm))
    
    for index in CartesianIndices(ulm_exact)
        l,m = split(index.I[1])
        if (l,m) == (1,1) || (l,m) == (18,4) || (l,m) == (39, -39) || (l,m) == (71,36) 
            ulm_exact[index] = l  
        end
    end
    
    for index in CartesianIndices(ulm_integ)
        l,m = split(index.I[1])
        @show l, m
        uu(μ, ν) =  u(μ, ν)*conj(ScalarSH(l,m,μ,ν))
        ulm_integ[index] = gaussquad(Float64, uu, 200)
    end
    
    @test isapprox(ulm, ulm_exact, atol=1e-12)
    @test isapprox(ulm, ulm_integ, atol=1e-12)
end

# Do the same exercise for vector spherical harmonics
v(a, μ, ν) = GradSH(a,1,1,μ,ν) + 18*GradSH(a,18,4,μ,ν) + 39*GradSH(a,39,-39,μ,ν) + 71*GradSH(a,71,36,μ,ν)

vmap = map(SH, (μ,ν)->v(1,μ,ν), (μ,ν)->v(2,μ,ν)) 
vlm  = nodal_to_modal_vector_op(SH)*vmap  

vlm_exact = zeros(length(vlm))
vlm_integ = zeros(length(vlm))

for index in CartesianIndices(vlm_exact)
    l,m = split(index.I[1])
    if (l,m) == (1,1) || (l,m) == (18,4) || (l,m) == (39, -39) || (l,m) == (71,36) 
        vlm_exact[index] = l  
    end
end

for index in CartesianIndices(vlm_integ)
    l,m = split(index.I[1])
    vv(μ, ν) = v(1,μ,ν)*conj(GradSH(1,l,m,μ,ν)) + v(2,μ,ν)*conj(GradSH(2,l,m,μ,ν))*(1/sin(μ)^2)
    if l != 0
        vlm_integ[index] = (1/(l*(l+1)))*gaussquad(Float64, vv, 200)
    end
    if l == m == 1
        @show l, m, vlm_integ[index]
    end
end


@show maximum(abs.(vlm - vlm_exact))
@test isapprox(vlm, vlm_exact, atol=1e-10)
@test isapprox(vlm, vlm_integ, atol=1e-10)

