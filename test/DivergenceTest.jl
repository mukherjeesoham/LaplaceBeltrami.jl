#---------------------------------------------------------------
#e LaplaceOnASphere
# Soham 8/20
# Construct a smooth z-coordinate using
# spherical harmonics
#---------------------------------------------------------------

using LinearAlgebra, PyPlot

SH = SphericalHarmonics(30)
@time S, S̄ = scalar_op(SH)
@time V, V̄ = vector_op(SH)
@time L = scale_lmodes(SH, (l,m)->-l*(l+1))
@time H = scale_vector(SH, sqrt_deth_by_detq_q_hinv)
@time J = scale_scalar(SH, sqrt_detq_by_deth)

# Compute the Laplace operator and look at the residual 
grad = V*S̄ 
div  = S*L*V̄  
Δ    = S̄*J*div*H*grad  

(l, m) = (1, 1)
nf  = map(SH, (μ,ν)->analyticF(l,m,μ,ν)) 
δ   = S*Δ*nf + l*(l+1)*nf

# @show deth(π/4,π/19)
# @show hinv(1,1,π/4,π/19)
# @show hinv(1,2,π/4,π/19)
# @show hinv(2,1,π/4,π/19)
# @show hinv(2,2,π/4,π/19)
# @show analyticF(1,1,π/4,π/19)
@show L2(SH, δ)


# Plot the fall-off of the coefficents and the vector components
if false
    semilogy(max_coefficent_for_each_l(SH, V̄*H*grad*nf), "g-^")
    show()
end

if true
    subplot(1,3,1)
    contourf(reshape(SH, δ, :scalar))
    colorbar()
    
    subplot(1,3,2)
    contourf(reshape(SH, S*Δ*nf, :scalar))
    colorbar()
    
    subplot(1,3,3)
    contourf(reshape(SH, -l*(l+1)*nf, :scalar))
    colorbar()
    show()
end

if false
    sinμ = map(SH, (μ,ν)->sin(μ))
    semilogy(max_coefficent_for_each_l(SH, S̄*sinμ), "r-^")
    semilogy(max_coefficent_for_each_l(SH, S̄*δ), "g-^")
    show()
end

