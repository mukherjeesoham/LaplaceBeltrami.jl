#---------------------------------------------------------------
# LaplaceOnASphere
# Soham 3/20
# Test metric computations 
#---------------------------------------------------------------
# [1] Test if we can take arbitrary derivatives using Spherical
#     Harmonics expansions
# [2] Then test if we can reproduce the results for contracted 
#     Christoffel symbols
# [3] Test the operator first for standard round sphere metric. 
# [4] Then make a coordinate transformation. 

SH = SphericalHarmonics(44)
f  = map(SH, (θ,ϕ)->ScalarSPH(2, 1, θ, ϕ))
dfdθ = map(SH, (θ,ϕ)->-(1/2)*cis(ϕ)*sqrt(15/2π)*cos(2θ))
dfdϕ = map(SH, (θ,ϕ)->-(1/2)*im*cis(ϕ)*sqrt(15/2π)*cos(θ)*sin(θ))
 
d2fdθ2 = map(SH, (θ,ϕ)->cis(ϕ)*sqrt(30/π)*sin(θ)*cos(θ))

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
@show L1(Dθ*(Dθ*f) - d2fdθ2)

g  = map(SH, (θ,ϕ)->cos(θ)*sin(θ))
dgdθ = map(SH, (θ,ϕ)->cos(2θ))
@show L1(Dθ*g - dgdθ)
