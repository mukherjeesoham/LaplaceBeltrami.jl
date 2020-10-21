#---------------------------------------------------------------
#e LaplaceOnASphere
# Soham 8/20
# Construct a smooth z-coordinate using
# spherical harmonics
#---------------------------------------------------------------

using PyPlot, Arpack, LinearAlgebra

SH = SphericalHarmonics(10)
@test deth(π/4, π/6) ≈ 0.6583073877333249

function only_sqrt_deth_by_detq(a::Int, b::Int, μ::T, ν::T)::T where {T}
    if a == b
        return sqrt(deth(μ,ν)/detq(μ,ν))
    else
        return 0
    end
end

# Compute the operators
@time S, S̄ = scalar_op(SH)
@time V, V̄ = grad_op(SH)
@time H    = scale_vector(SH, sqrt_deth_by_detq_q_hinv)
@time L    = scale_lmodes(SH, (l,m)->-l*(l+1))
@time W    = scale_scalar(SH, (μ,ν)->sqrt(detq(μ,ν)/deth(μ,ν)))  

# Compute the fields
nf  = map(SH, (μ,ν)->analyticF(1,1,μ,ν)) 

# Compute the Laplace operator and look at the residual 
grad = V*S̄ 
div  = S*L*V̄  
println("Computing the Laplace Operator")
@time Δ = S̄*(W*(div*(H*grad)))
@show L2(SH, S*Δ*nf + 2*nf)

# Compute eigenvectors and eigenvalues
if false
    # Use Arpack
    λ, ϕ = eigs(Δ*S, nev=14, which=:SM)
    @show real.(λ)
    
    for column in 2:4
        contourf(reshape(SH, S*ϕ[:,column], :scalar))
        l, m = split(column)
        m    = m+1
        savefig("PDF/$l-$m-eig.pdf")
        close()
    end
end

if true
    # Use eigensolver from LinearAlgebra.
    F = eigen(Δ*S)
    @show sort(abs.(F.values))[2:7]
    
    for column in 1:6
        contourf(reshape(SH, S*F.vectors[:, end-column], :scalar))
        l, m = split(column+1)
        m    = m+1
        savefig("PDF/$l-$m-eig.pdf")
        close()
    end
end


