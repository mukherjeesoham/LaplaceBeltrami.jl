#---------------------------------------------------------------
# LaplaceOnASphere
# Soham 8/20
# Construct a smooth, purely real z-coordinate using
# spherical harmonics
#---------------------------------------------------------------

using LinearMaps, LinearAlgebra, SparseArrays, NLsolve, Arpack, PyPlot, JLD

SH = SphericalHarmonics(8)

if (true)
    # Compute the operators
    S, S̄ = LinearMap.(dropzeros!.(sparse.(scalar_op(SH))))
    V, V̄ = LinearMap.(dropzeros!.(sparse.(grad_op(SH))))
    H    = LinearMap(dropzeros!(sparse(scale_vector(SH, sqrt_deth_by_detq_q_hinv))))
    L    = LinearMap(dropzeros!(sparse(scale_lmodes(SH, (l,m)->-l*(l+1)))))
    
    # Compute the Laplace operator using LinearMaps
    grad = V 
    div  = S*L*V̄  
    
    φ(μ,ν) = ScalarSH(1,0,μ,ν)  
    W = LinearMap(dropzeros!(sparse(scale_scalar(SH, (μ,ν)->exp(-2*φ(μ,ν))*sqrt(detq(μ,ν)/deth(μ,ν))))))
    Δ = S̄*(W*(div*(H*grad)))

    λ, ϕ = eigs(Δ, nev=14, which=:LR)
    print(abs.(ϕ[:,1]))
end


