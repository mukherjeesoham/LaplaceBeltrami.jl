#---------------------------------------------------------------
# LaplaceOnASphere
# Soham 6/20
#
# Construct the Laplace operator in local coordinates using 
# Spherical harmonics
# We have 1/|g| d/dx^i (|g| g^ij dϕ/dx^j) 
# where x^i = {μ, ν} and g = ({1, 0}, {0, sin(θ)^2}}
# We work with a general metric h and we shoe-horn 
# the metric g in there. 
# We have 1/|h| d/dx^i (|h| h^ij dϕ/dx^j) 
# We have |g|/|h| 1/|g| d/dx^i (|g| |h|/|g| δ^i_m h^mj dϕ/dx^j) 
# or  |g|/|h| 1/|g| d/dx^i (|g| |h|/|g| g^in g_nm h^mj dϕ/dx^j) 
# or  |g|/|h| 1/|g| d/dx^i (|g| g^in |h|/|g|  g_nm h^mj dϕ/dx^j) 
# or  |g|/|h| 1/|g| d/dx^i (|g| g^in β_n) 
# or  |g|/|h| Ylm (-l*(l+1)) β^lm_n
#---------------------------------------------------------------

using LinearAlgebra, SparseArrays, PyPlot

#---------------------------------------------------------------
# Functions for constructing the scaling operators
#---------------------------------------------------------------

θ(μ, ν) = μ + cos(μ/2)*(1-cos(μ/2))
ϕ(μ, ν) = ν 

function hinv(a::Int, b::Int, μ::Float64, ν::Float64)::Float64
    if a == b == 1
        p = exp(cos(μ))
        q = -1 + 2*p
        return 4/(2  -sin(μ/2) + sin(μ))^2
    elseif a == b == 2
        p = exp(cos(μ))
        q = -1 + p
        return csc(μ + 2*cos(μ/2)*(sin(μ/4)^2))^2
    else
        return 0
    end
end

function sqrt_detg_by_deth(μ::Float64, ν::Float64)::Float64
    dethinv = -hinv(1,2,μ,ν)*hinv(2,1,μ,ν) + hinv(1,1,μ,ν)*hinv(2,2,μ,ν)
    return sin(μ)*sqrt(dethinv)
end

function sqrt_deth_by_detg_g_hinv(a::Int, b::Int, μ::Float64, ν::Float64)::Float64
   dethinv = -hinv(1,2,μ,ν)*hinv(2,1,μ,ν) + hinv(1,1,μ,ν)*hinv(2,2,μ,ν)
   deth    = 1/dethinv
   if a == 1
       return sqrt(deth)*csc(μ)*hinv(a,b,μ,ν)
   else
       return sqrt(deth)*sin(μ)*hinv(a,b,μ,ν) 
   end
end

function divergence(l::Int, m::Int)::Int
    return -l*(l+1)
end

#---------------------------------------------------------------
# Construct the operator
#---------------------------------------------------------------

function laplace(SH::SphericalHarmonics)
    S = modal_to_nodal_scalar_op(SH) 
    S̄ = nodal_to_modal_scalar_op(SH)
    V = modal_to_nodal_vector_op(SH)
    V̄ = nodal_to_modal_vector_op(SH) 
    
    D = scale_scalar(SH, sqrt_detg_by_deth)
    H = scale_vector(SH, sqrt_deth_by_detg_g_hinv)
    L = scale_lmodes(SH, divergence) 
    
    grad = V
    div  = S*L*V̄
    Δ    = S̄*D*div*(H*grad)

    return Δ    
end

#---------------------------------------------------------------
# Functions for testing the eigenvalues and eigevectors 
# of the operator
#---------------------------------------------------------------

function PyPlot. plot(F::Eigen, lmax::Int) where {T}
    # Plot eigenvalues
    u = F.values
    @assert onlyreal(u)
    absrealF = sort(abs.(real.(u)))
    argmax = (lmax)^2 + 2*(lmax) + 1
    figure(figsize=(10, 5))

    subplot(1,2,1)
    plot(absrealF[1:argmax], "r o")
    plot(round.(absrealF)[1:argmax], "b-o", markersize=2.0, linewidth=1.0)

    subplot(1,2,2)
    error = log10.(abs.(absrealF[1:argmax] - round.(absrealF[1:argmax])))
    plot(error, "k--o")

    savefig("eigenvals-$(SH.lmax)-$(SH.N).pdf")
    close()

    # plot eigenvectors
    V = F.vectors
    for lm in 1:lmax^2 + 2*lmax
        l, m  = split(lm+1)
        Ylm   = reshape(SH, map(SH, (μ,ν)->ScalarSPH(l,-m, θ(μ,ν), ϕ(μ,ν))))
        S     = modal_to_nodal_scalar_op(SH)
        Ȳlm   = reshape(SH, S*V[:, end - lm])

        figure(figsize=(5, 4))
        title("$(real(u[end - lm]))")
        contourf(Ȳlm)
        colorbar()
        savefig("./eigenvectors/eigenvec-$l-$lm-$(SH.lmax)-$(SH.N).pdf")
        close()

        figure(figsize=(5, 4))
        contourf(Ylm)
        colorbar()
        savefig("./eigenvectors/analytic/eigenvec-$l-$lm.pdf")
        close()
    end
end


function coordinates(SH::SphericalHarmonics, u::Eigen)
    S   = modal_to_nodal_scalar_op(SH)
    u10 = S*u.vectors[:, end - 3]
    u11 = S*u.vectors[:, end - 1]

    # Check how large the error is 
    U10 = map(SH, (μ, ν)->ScalarSPH(1,0, θ(μ,ν), ϕ(μ,ν)))
    U11 = map(SH, (μ, ν)->ScalarSPH(1,1, θ(μ,ν), ϕ(μ,ν)))

    contourf(reshape(SH, u10 - U10))
    colorbar()
    savefig("diff-10.pdf")
    close()

    contourf(reshape(SH, u11 - U11))
    colorbar()
    savefig("diff-11.pdf")
    close()

    @show L2(U10 - u10)
    @show L2(U11 - u11)

    # Compute the coordinates
    # TODO: check these transformations
    theta = acos.(u10)
    phi   = -im.*log.(u11./sin.(theta)) 

    # Compare with the coordniate transformation you want
    # to compare with
    theta_analytic = reshape(SH, map(SH, (μ, ν)->θ(μ,ν)))
    phi_analytic   = reshape(SH, map(SH, (μ, ν)->ϕ(μ,ν)))

    theta = reshape(SH, theta)
    phi = reshape(SH, phi)

    contourf(theta)
    colorbar()
    savefig("theta.pdf")
    close()

    contourf(phi)
    colorbar()
    savefig("phi.pdf")
    close()

    # plot the difference
    contourf(theta - theta_analytic)
    colorbar()
    savefig("diff_theta.pdf")
    close()

    contourf(phi - phi_analytic)
    colorbar()
    savefig("diff_phi.pdf")
    close()

    return (θ, ϕ)
end

function eigencheck(SH::SphericalHarmonics)
    Δ = laplace(SH) 
    F = eigen(Δ)
    plot(F, 2)
    coordinates(SH, F)
end

#---------------------------------------------------------------
# test laplace, and if that doesn't work 
# check grad, div and scaling
#---------------------------------------------------------------

SH = SphericalHarmonics(12)

@testset "laplace" begin
    Δ = laplace(SH)
    lmax = 3
    for l in 1:lmax
        for m in -l:l
            Ylm = map(SH, (μ,ν)->ScalarSPH(l,m, θ(μ,ν), ϕ(μ,ν)))
            @test_broken isapprox(S*Δ*S̄*Ylm, -l*(l+1)*Ylm; atol=1e-10)
        end
    end
end;

eigencheck(SH)

# Look at the coefficents
function plot_coeff(SH::SphericalHarmonics, l::Int, m::Int)

    Ylm = map(SH, (μ,ν)->ScalarSPH(l,m, θ(μ,ν), ϕ(μ,ν)))
    S = modal_to_nodal_scalar_op(SH) 
    S̄ = nodal_to_modal_scalar_op(SH)
    V = modal_to_nodal_vector_op(SH)
    V̄ = nodal_to_modal_vector_op(SH) 
    
    D = scale_scalar(SH, sqrt_detg_by_deth)
    H = scale_vector(SH, sqrt_deth_by_detg_g_hinv)
    
    grad = V
    P = V̄*(H*grad)

    vcoeffs = P*S̄*Ylm
    plot(log10.(abs.(real.(vcoeffs))))
    savefig("vcoeffs.pdf")
    close()
end

plot_coeff(SH, 1,1)

