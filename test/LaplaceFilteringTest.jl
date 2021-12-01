#-----------------------------------------------------
# Test what filtering is doing for both scalar
# and vector spherical harmonics
# Soham 09/2021
#-----------------------------------------------------
# TODO: Test filtering for spin 0 harmonics [Done]
# TODO: Test filtering for spin 1 harmonics [Done]
# TODO: Test for real and complex spherical harmonics [Done]
# TODO: Isolate the issue between grad and div in the Laplace operator instead of trying all of them after testing the Laplace operator itself. 
# TODO: Check what happens if you add a little amount of perturbation?  [In Progress]
# NOTE: The order of error seems to be the same magnitude as the perturbation added. 

using FastSphericalHarmonics, LinearAlgebra, Test, Plots

function spinzero(μ::T, ν::T) where {T<:Real}
    u = 0.0
    p = 1e-3
    for l in 0:5, m in -l:l
        u = u + sYlm(Real, 0, l, m,  μ, ν) 
    end
    δu = p * rand(size(u)...) 
    return u + δu
end

function spinone(μ::T, ν::T) where {T<:Real} # Use s = -1
    u = 0.0
    p = 1e-4
    # for l in 0:1, m in -l:l
        # u = u .+ sYlm(Complex, 1, l, m,  μ, ν) 
    # end
    u  = sYlm(-1, 2, 2, μ, ν)
    δu = p * rand(size(u)...) 
    return u + δu
end

if false
    lmax = 6
    s    = map(spinzero, lmax)
    slm  = spinsph_transform(s, 0) 
    @test slm == restrict(prolongate(slm, lmax), 2*lmax) 
    display(slm)
    display(prolongate(slm, lmax))
    display(restrict(slm, 2*lmax))
    slm  = filter(slm, lmax)
    display(slm)
    rs   = spinsph_evaluate(slm, 0)
    @show norm(rs - s) 
    rslm = spinsph_transform(rs, 0)
    rrs  = spinsph_evaluate(rslm, 0)
    @show norm(rslm - slm)
    @show norm(rrs - s) 
    # p = plot(contourf(s, title="original", ylabel="θ", xlabel="ϕ"), 
             # contourf(rs, title="reconstructed", ylabel="θ", xlabel="ϕ"))
    # savefig(p, "plots/scalar-reconstruction.pdf")
end


if false
    lmax = 6
    s    = map(spinone, lmax)
    slm  = spinsph_transform(s, -1) 
    @test slm == restrict(prolongate(slm, lmax), lmax) 
    display(slm)
    slm  = filter(slm, lmax)
    display(slm)
    rs   = spinsph_evaluate(slm, -1)
    @show norm(rs - s) 
    rslm = spinsph_transform(rs, -1)
    rrs  = spinsph_evaluate(rslm, -1)
    @show norm(rslm - slm)
    @show norm(rrs - s) 
    # p = plot(contourf(s, title="original", ylabel="θ", xlabel="ϕ"), 
             # contourf(rs, title="reconstructed", ylabel="θ", xlabel="ϕ"))
    # savefig(p, "plots/scalar-reconstruction.pdf")
end

# Now construct the Laplace operator from scratch and test filtering based on action 
# of the Laplace operator on a vector.
# FIXME: Where should we filter? 
# TODO: Isolate filtering to just the divergence first.
# TODO: Isolate filtering to just the gradient. 
# TODO: Try both. Also, first try killing the spurious modes. 

function laplaceF(C⁰::AbstractMatrix{T}, A::Laplace) where {T<:Real}
    C⁰      = prolongate(C⁰, A.lmax)
    AP      = prolongate(A) # <== Prolongate
    dU      = grad(C⁰, AP.lmax)
    # dU      = spinsph_evaluate(prolongate(spinsph_transform(dU, 1), A.lmax), 1) 
    SdU     = map(S1, AP.q, AP.h, dU) 
    dSdU    = div(SdU, AP.lmax) 
    APR     = restrict(AP)
    dSdU    = spinsph_evaluate(restrict(spinsph_transform(dSdU, 0), AP.lmax), 0) # <== Restrict
    ΔU      = map(S2, APR.q, APR.h, dSdU) 
    return spinsph_transform(ΔU, 0) 
end

function normalize(C⁰::AbstractMatrix{T}) where {T<:Real}
    CC⁰ = deepcopy(C⁰)
    for l in 1:lmax, m in (-l):l 
        CC⁰[spinsph_mode(0, l, m)] = CC⁰[spinsph_mode(0, l, m)] / (l * (l + 1))
    end
    return CC⁰ 
end

# FIXME: The operator is unstable. We can only push uptil n = 6 for lmin = 62. Introduce filering
# FIXME: Removing the abs(m) > l modes at the end of every Laplace operator action doesn't help at all.  
function stability(l::Int, s::Matrix{T}, A::Laplace) where {T}
    error = zeros(5)
    s0 = s
    # FIXME: Check these loops explicitlty.
    for index in CartesianIndices(error) 
        slm  = spinsph_transform(s, 0)
        Δslm = laplace(slm, A) 
        Δslm = filter(Δslm, A.lmax)    # <== Filter the abs(m) > l modes after every laplace solve.
        # TODO: Divide each l mode by l * (l + 1)
        nΔslm  = normalize(Δslm)
        visualize(slm - nΔslm)
        Δs     = spinsph_evaluate(Δslm, 0)
        # visualize(Δs, :contour)
        println(index.I[1], " nδulm ", norm(nΔslm + slm))
        println(index.I[1], " δulm ",  norm(Δslm + l * (l + 1) * slm))
        println(index.I[1], " δu   ",  norm(Δs + l * (l + 1) * s))
        println(index.I[1], " δu0  ",  norm(Δs + (l * (l + 1))^index.I[1] * s0))
        println("--------------------------------------------------")
        s     = Δs 
    end
end

if true
    lmin     = 32
    qmetric  = map(q, lmin)   
    hmetric  = map(h, lmin)   
    A        = Laplace{Float64}(lmin, qmetric, hmetric)
    (l, m)   = (2, 2)
    U        = map((μ,ν)->sYlm(Real,0, l, m, θϕ_of_μν(μ,ν)...), lmin) 
    Ulm      = spinsph_transform(U, 0)
    @show maximum(abs.(laplaceF(Ulm, A) + l * (l + 1) * Ulm ))
    stability(l, U, A)
end

