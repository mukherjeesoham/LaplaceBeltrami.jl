#---------------------------------------------------------------
# LaplaceOnASphere
# Soham 9/20
# Test tensor spherical harmonics
#---------------------------------------------------------------

θ = π/5
ϕ = π/19
@show TensorSHE(1,1,2,0,θ,ϕ)
@show TensorSHE(1,2,2,0,θ,ϕ)
@show TensorSHE(2,1,2,0,θ,ϕ)
@show TensorSHE(2,2,2,0,θ,ϕ)

@show TensorSH0(1,1,2,0,θ,ϕ)
@show TensorSH0(1,2,2,0,θ,ϕ)
@show TensorSH0(2,1,2,0,θ,ϕ)
@show TensorSH0(2,2,2,0,θ,ϕ)
