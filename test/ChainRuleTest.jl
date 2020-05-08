#---------------------------------------------------------------
# LaplaceOnASphere
# Soham 4/20
# Test gradient and divergence separately
# under coordinate transformations
#---------------------------------------------------------------
# f (nodes) > ∇f (nodes) [Under a coordinate transformation]

SH = SphericalHarmonics(24)
f  = map(SH, (θ,ϕ)->ScalarSPH(2, 1, θ+ϕ, ϕ))
dfdθ = map(SH, (θ,ϕ)->-(1/2)*cis(ϕ)*sqrt(15/2π)*cos(2θ+2ϕ))
dfdϕ = map(SH, (θ,ϕ)->-(1/2)*im*cis(ϕ)*sqrt(15/2π)*cos(θ)*sin(θ))
 
# Construct dYdθ and dYdϕ operators
function derivative(S::SphericalHarmonics{T}) where {T}
    lmax, n = S.lmax, S.N
    A = zeros(Complex{T}, n*(2*n), (lmax)^2 + 2*(lmax) + 1)
    B = zeros(Complex{T}, n*(2*n), (lmax)^2 + 2*(lmax) + 1)
    C = nodal_to_modal_scalar_op(SH)
    for index in CartesianIndices(A)
        (i,j) = split(index.I[1], n)
        (l,m) = split(index.I[2])
        (θ,ϕ) = collocation(S,i,j)
        dYdθ = m*cot(θ)*ScalarSPH(l,m,θ,ϕ) + sqrt((l-m)*(l+m+1))*cis(-ϕ)*ScalarSPH(l,m+1,θ,ϕ)
        dYdϕ = im*m*ScalarSPH(l,m,θ,ϕ)
        A[index] = dYdθ 
        B[index] = dYdϕ 
    end
    return (A*C,B*C)
end

Dθ, Dϕ = derivative(SH)
@test L1(Dθ*f - dfdθ) < 1e-12
@test L1(Dϕ*f - dfdϕ) < 1e-12

#---------------------------------------------------------------
# Consider the coordinate transformation (θ,ϕ) -> (μ,ν)
# (μ, ν) = (θ + ϕ, ϕ)
#---------------------------------------------------------------
μ(θ, ϕ) = θ + ϕ
ν(θ, ϕ) = ϕ

dθdμ = scaling_scalar_op(SH, (θ,ϕ)-> 1)
dϕdμ = scaling_scalar_op(SH, (θ,ϕ)-> 0)

dθdν = scaling_scalar_op(SH, (θ,ϕ)-> -1)
dϕdν = scaling_scalar_op(SH, (θ,ϕ)-> 1)

fμν = map(SH, (θ,ϕ)->ScalarSPH(2, 1, μ(θ,ϕ), ν(θ,ϕ)))
∇1fμν = map(SH, (θ,ϕ)->-(1/2)*cis(ν(θ,ϕ))*sqrt(15/2π)*cos(2*μ(θ,ϕ)))
∇2fμν = map(SH, (θ,ϕ)->-(1/2)*im*cis(ν(θ,ϕ))*sqrt(15/2π)*cos(μ(θ,ϕ)))
Δfμν  = -6*fμν

# Construct the gradient operator
invsinμ = scaling_scalar_op(SH, (θ,ϕ)->1/sin(μ(θ,ϕ)))
∇1 = dθdμ*Dθ + dϕdμ*Dϕ
∇2 = dθdν*Dθ + dϕdν*Dϕ

# Test the gradient operator
# @show L1(∇1*fμν - ∇1fμν)
# @show L1(∇2*fμν - ∇2fμν)

# Now construct the Laplace operator
cotμ = scaling_scalar_op(SH, (θ,ϕ)->cot(μ(θ,ϕ)))
Δ = dθdμ*Dθ*∇1 + dϕdμ*Dϕ*∇1 + cotμ*∇1 + invsinμ*(dθdν*Dθ*∇2 + dϕdν*Dϕ*∇2)
Δ = 1/(sqrtdetg)*(∇1*sqrtdetg*(ginv11*∇1 + ginv12*∇2) + ∇2*sqrtdetg*(ginv21*∇1 + ginv22*∇2)) # raising the index

(∇f)^i = g^{ij} \partial f / dx^j

@show L1(Δ*fμν - Δfμν)
