#---------------------------------------------------------------
# LaplaceOnASphere
# Soham 8/20
#---------------------------------------------------------------
using Test, LaplaceOnASphere
using FastGaussQuadrature, Arpack, LinearAlgebra

# TODO:
#       [4] Check Eigenvalues of the Laplace operator for simple coordinate
#       [5] Check Eigenvalues of the Laplace operator for complex coordinates
#       [6] Test convergence of the action of the Laplace operator on functions.


@testset "Basic" begin
    # Have regression tests
    @test Ylm(2,  -2, π/6, π/9) ≈ 0.07397580149502045 - 0.06207306775051537im 
    @test Ylm(12,  0, π/4, π/12) ≈ -0.34799123292479217
    @test Ylm(11, -2, π/1000, π/15) ≈ 0.0001997151969678536 - 0.00008891893458213277im 

    @test Glm(1, 2, -2, π/6, π/9)  ≈ 0.2562596934400104 - 0.2150274142511155im 
    @test Glm(2, 2, -2, π/6, π/9)  ≈ -0.1241461355010307 - 0.14795160299004095im
    @test Glm(1, 12, 0, π/4, π/15) ≈ -1.682083171530092 
    @test Glm(2, 12, 0, π/4, π/15) ≈ 0.0 
    @test Glm(1, 11,-2, π/1000, π/15) ≈ 0.12712904868104322 - 0.05660149920878576im # This breaks if you change abs_tol to 1e-12
    @test Glm(2, 11,-2, π/1000, π/15) ≈ -0.0001778378691642657 - 0.00039943039393570745im
    # TODO: Add regression tests for Curl Vector spherical harmonics
    
    # Test orthogonality
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
    
    function kroneckerdelta(l::Int, l̄::Int)::Int
        if l == l̄
            return 1
        else
            return 0
        end
    end
    
    for l in 2:4, l̄ in 6:9
        m = rand(-l:l)
        m̄ = rand(-l̄:l̄)
        ss̄(μ, ν) = Ylm(l, m, μ, ν)*conj(Ylm(l̄, m̄, μ, ν))
        # NOTE: To test the orthogonality of the Vector spherical harmonics, we *raise* the index of the vectors using
        # the round sphere metric.
        vv̄(μ, ν) = Glm(1, l, m, μ, ν)*conj(Glm(1, l̄, m̄, μ, ν)) + Glm(2, l, m, μ, ν)*conj(Glm(2, l̄, m̄, μ, ν))*(1/sin(μ)^2)
        cc̄(μ, ν) = Clm(1, l, m, μ, ν)*conj(Clm(1, l̄, m̄, μ, ν)) + Clm(2, l, m, μ, ν)*conj(Clm(2, l̄, m̄, μ, ν))*(1/sin(μ)^2)
        vc̄(μ, ν) = Glm(1, l, m, μ, ν)*conj(Clm(1, l̄, m̄, μ, ν)) + Glm(2, l, m, μ, ν)*conj(Clm(2, l̄, m̄, μ, ν))*(1/sin(μ)^2)
        @test isapprox(gaussquad(Float64, ss̄, 200), kroneckerdelta(l,l̄)*kroneckerdelta(m, m̄), atol=1e-10)
        @test isapprox(gaussquad(Float64, vv̄, 200), l*(l+1)*kroneckerdelta(l,l̄)*kroneckerdelta(m, m̄), atol=1e-10)
        @test isapprox(gaussquad(Float64, cc̄, 200), l*(l+1)*kroneckerdelta(l,l̄)*kroneckerdelta(m, m̄), atol=1e-10)
        @test isapprox(gaussquad(Float64, vc̄, 200), 0, atol=1e-10)
    end

    # Test divergence and grad for q and h
    SH    = SphericalHarmonics(12)
    lrand = rand(2:4)
    mrand = rand(-lrand:lrand)
    tYlm   = map(SH, (μ,ν)->Ylm(lrand, mrand, μ,ν))
    tGYlm  = map(SH, (μ,ν)->Glm(1, lrand, mrand, μ,ν), 
                     (μ,ν)->Glm(2, lrand, mrand, μ,ν))
    tdiv   = div(SH)
    tgrad  = grad(SH)
    @test tgrad*tYlm ≈ tGYlm 
    @test tdiv*tGYlm ≈ -lrand*(lrand+1)*tYlm 

    # Test quad
    Y00 = map(SH, (μ,ν)->Ylm(0,0,μ,ν))
    @test quad(SH, Y00) ≈ 2*sqrt(π)

    # Test the Laplace operator for simple coordinates
    lrand = rand(2:4)
    mrand = rand(-lrand:lrand)
    tYlm  = map(SH, (μ,ν)->Ylm(lrand, mrand, μ,ν))
    laplace = tdiv*tgrad   
    @test L2(SH, laplace*tYlm + lrand*(lrand+1)*tYlm) < 1e-8

    # Test the eigenvalues of the laplace operator in simple coordinates
    S, S̄ = scalar_op(SH)
    λ, ϕ = eigs(S̄*laplace*S, nev=14, which=:LR)
    eigvals = [0, -2, -2, -2, -6, -6, -6, -6, -6, -12, -12, -12, -12, -12]
    @test norm(real.(λ) - eigvals) < 1e-12
end;


libraries = []
for file in libraries
    @info "Testing $file"
    include("$(file)Test.jl")
end


