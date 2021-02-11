#---------------------------------------------------------------
# LaplaceOnASphere
# Soham 8/20
# Construct a smooth, purely real z-coordinate using
# spherical harmonics
#---------------------------------------------------------------

using PyPlot, Arpack, LinearMaps, LinearAlgebra, SparseArrays, JLD
using Optim, LsqFit

function laplace(SH::SphericalHarmonics{T}) where {T}
    if isfile("./output/Laplace-round-sphere-$(SH.lmax).jld") == false
        # Compute the operators
        @time S, S̄ = LinearMap.(dropzeros!.(sparse.(scalar_op(SH))))
        @time V, V̄ = LinearMap.(dropzeros!.(sparse.(grad_op(SH))))
        @time H    = LinearMap(dropzeros!(sparse(scale_vector(SH, sqrt_deth_by_detq_q_hinv))))
        @time L    = LinearMap(dropzeros!(sparse(scale_lmodes(SH, (l,m)->-l*(l+1)))))
        @time W    = LinearMap(dropzeros!(sparse(scale_scalar(SH, (μ,ν)->sqrt(detq(μ,ν)/deth(μ,ν))))))
        
        # Compute the Laplace operator using LinearMaps
        grad = V 
        div  = S*L*V̄  

        if false
            println("Computing Laplace eigenvalues for distorted coordinates.")
            # Round sphere in bad coordinates
            @time Δ = S̄*(W*(div*(H*grad)))
        else
            println("Computing Laplace eigenvalues for round sphere coordinates.")
            # Roud sphere in good coordinates
            @time Δ = S̄*(div*grad) 
        end

        @time λ, ϕ = eigs(Δ, nev=14, which=:LR)
        save("./output/Laplace-round-sphere-$(SH.lmax).jld", "λ", λ, "ϕ", ϕ, "Δ", Δ)
    else
        @time S = modal_to_nodal_scalar_op(SH) 
        @time d = load("./output/Laplace-round-sphere-$(SH.lmax).jld") 
        λ = d["λ"]
        ϕ = d["ϕ"]
        Δ = d["Δ"]
    end
    return (λ, ϕ, (S*ϕ)[:,2:4], Δ)
end

function plot(SH::SphericalHarmonics{T}, ϕ::Array{Complex{T},2}, string::String) where {T}
    fig = figure(figsize=(20,7))
    for index in 1:6
        subplot(2,3,index)
        if index <= 3
            contourf(reshape(SH, real.(ϕ[:, mod1(index, 3)]), :scalar))
            title(L"Real")
            colorbar()
        else
            contourf(reshape(SH, imag.(ϕ[:, mod1(index, 3)]), :scalar))
            title(L"Imag")
            colorbar()
        end
    end
    savefig("./output/eigenvalues-$(string).pdf")
    close()
end

function Base.show(SH::SphericalHarmonics{T}, u::Array{T,1}) where {T}
    contourf(reshape(SH, u, :scalar))
    colorbar()
end

function coordinates(u::Array{Complex{T},2})::NTuple{2,Array{T,1}} where {T}
    θ = acos.(sqrt(4π/3)*real.(u[:,1]))
    ϕ = angle.(u[:,2]) 
    return (θ, ϕ)
end

function diagonalize(SH::SphericalHarmonics{T}, θ::Array{T,1}, ϕ::Array{T,1})::NTuple{3,Array{Complex{T},1}} where {T}
    S, S̄ = LinearMap.(dropzeros!.(sparse.(scalar_op(SH))))
    V, V̄ = LinearMap.(dropzeros!.(sparse.(grad_op(SH))))
    dθ = reshape(V*(S̄*θ), (:,2))
    dϕ = reshape(V*(S̄*ϕ), (:,2))
    dθdμ, dθdν = (dθ[:,1], dθ[:,2])
    dϕdμ, dϕdν = (dϕ[:,1], dϕ[:,2])

    hinvμμ = map(SH, (μ,ν)->hinv(1,1,μ,ν)) 
    hinvμν = map(SH, (μ,ν)->hinv(1,2,μ,ν)) 
    hinvνν = map(SH, (μ,ν)->hinv(2,2,μ,ν)) 

    ginvθθ = dθdμ.*dθdμ.*hinvμμ + dθdμ.*dθdν.*hinvμν + dθdν.*dθdμ.*hinvμν + + dθdν.*dθdν.*hinvνν
    ginvθϕ = dθdμ.*dϕdμ.*hinvμμ + dθdμ.*dϕdν.*hinvμν + dθdν.*dϕdμ.*hinvμν + + dθdν.*dϕdν.*hinvνν
    ginvϕϕ = dϕdμ.*dϕdμ.*hinvμμ + dϕdμ.*dϕdν.*hinvμν + dϕdν.*dϕdμ.*hinvμν + + dϕdν.*dϕdν.*hinvνν
    return (ginvθθ, ginvθϕ, ginvϕϕ)
end

function checklaplace(SH, u, Δ)
    S, S̄ = LinearMap.(dropzeros!.(sparse.(scalar_op(SH))))
    for i in 1:3
        @show L2(SH, (S*Δ*S̄)*u[:,i] + 2*u[:,i])
    end
end

function checkorthonormality(SH, u)
    for k in 1:3, l in 1:3
        @show k, l, dot(SH, u[:,k], u[:,l])
    end
end

function conjugate(u::Array{T,2})::Array{T,2} where {T}
    function leastsquares(x)
        w1 = sin(x[1])*cos(x[2])*u[:,1]*cis(x[7]) + sin(x[1])*sin(x[2])*u[:,2]*cis(x[7]) + cos(x[1])*u[:,3]*cis(x[7]) 
        w2 = sin(x[3])*cos(x[4])*u[:,1]*cis(x[7]) + sin(x[3])*sin(x[4])*u[:,2]*cis(x[7]) + cos(x[3])*u[:,3]*cis(x[7]) 
        w3 = sin(x[5])*cos(x[6])*u[:,1]*cis(x[7]) + sin(x[5])*sin(x[6])*u[:,2]*cis(x[7]) + cos(x[5])*u[:,3]*cis(x[7]) 

        # Compute the residuals
        r1 = w1 - conj.(w1)
        r2 = w2 - conj.(w3)
        r3 = dot(w1, w2) 
        r4 = dot(w1, w3)
        r5 = dot(w2, w3)

        # Now merge them to a number. Take the integration
        # weights on the sphere into account. Add the dot products to the residual.
        rsq = quad(SH, conj(r1).*r1 + conj.(r2).*r2) + conj.(r3).*r3 + conj.(r4).*r4 + conj.(r5).*r5
        @assert all(imag.(rsq) .< 1e-12)
        return real(rsq)
    end

    x0 = zeros(9)
    r0 = optimize(leastsquares, x0)
    @show r0
    x = Optim.minimizer(r0)
    w = similar(u)
    w[:,1] = sin(x[1])*cos(x[2])*u[:,1]*cis(x[7]) + sin(x[1])*sin(x[2])*u[:,2]*cis(x[7]) + cos(x[1])*u[:,3]*cis(x[7]) 
    w[:,2] = sin(x[3])*cos(x[4])*u[:,1]*cis(x[7]) + sin(x[3])*sin(x[4])*u[:,2]*cis(x[7]) + cos(x[3])*u[:,3]*cis(x[7]) 
    w[:,3] = sin(x[5])*cos(x[6])*u[:,1]*cis(x[7]) + sin(x[5])*sin(x[6])*u[:,2]*cis(x[7]) + cos(x[5])*u[:,3]*cis(x[7]) 
    return w
end

SH = SphericalHarmonics(12)
λ, ulm, u, Δ = laplace(SH)

# First, check the eigenvalues and the eigenvectors for the round-sphere
# metric. We, know for example, such eigenvectors exist, so this will allow us
# to figure out if our method actually works. Then, formulate the problem as a
# whole least squares problem. This requires some thought, but it should be
# possible.  

q1 = map(SH, (θ, ϕ)->ScalarSH(1,-1, θ, ϕ))
q2 = map(SH, (θ, ϕ)->ScalarSH(1, 1, θ, ϕ))
q3 = map(SH, (θ, ϕ)->ScalarSH(1, 0, θ, ϕ))

@show typeof(q1)
@show typeof(hcat(q1,q2,q3))

plot(SH, hcat(q1,q2,q3), "round-sphere-nice")
plot(SH, u, "round-sphere")
exit()

gramschmidt!(SH, u)
w = conjugate(u)
@show w[:,1][1]
@show w[:,2][1]
@show w[:,3][1]

# What goes wrong? I think the orthogonality
checkorthonormality(SH, w)

plot(SH, w, "conjugate-orthogonal")
