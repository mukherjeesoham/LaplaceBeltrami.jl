#---------------------------------------------------------------
#e LaplaceOnASphere
# Soham 8/20
# Construct a smooth z-coordinate using
# spherical harmonics
#---------------------------------------------------------------

using PyPlot, Arpack, LinearMaps

SH = SphericalHarmonics(10)

# Compute the operators
@time S, S̄ = scalar_op(SH)
@time V, V̄ = grad_op(SH)
@time H    = scale_vector(SH, sqrt_deth_by_detq_q_hinv)
@time L    = scale_lmodes(SH, (l,m)->-l*(l+1))
@time W    = scale_scalar(SH, (μ,ν)->sqrt(detq(μ,ν)/deth(μ,ν)))

# # Compute the Laplace operator using LinearMaps
@time grad = V*S̄ 
@time div  = S*L*V̄  
println("Computing the full laplace operator")
@time Δ = S̄*(W*(div*(H*grad)))
@time λ, ϕ = eigs(Δ*S; nev=12, which=:LR)
@show real.(λ)

@time S, S̄ = LinearMap.(scalar_op(SH))
@time V, V̄ = LinearMap.(grad_op(SH))
@time H    = LinearMap(scale_vector(SH, sqrt_deth_by_detq_q_hinv))
@time L    = LinearMap(scale_lmodes(SH, (l,m)->-l*(l+1)))
@time W    = LinearMap(scale_scalar(SH, (μ,ν)->sqrt(detq(μ,ν)/deth(μ,ν))))  

println("Compute the liner map Laplace operator and it's eigenvalues")
@time Δ = LinearMap(S̄*W*S*L*V̄*H*V*S̄*S)
@time λ, ϕ = eigs(Δ; nev=12, which=:LR)
@show real.(λ)

