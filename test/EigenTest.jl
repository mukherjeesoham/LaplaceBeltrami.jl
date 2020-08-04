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

using LinearAlgebra, SparseArrays, Test

p = 1.0

θ(μ, ν) = μ
ϕ(μ, ν) = ν + p*sin(μ)

function LaplaceOnASphere.ScalarSPH(l::Int, m::Int, μ::T, ν::T, kind::Symbol)::Complex{T} where {T}
    if kind == :physical
        return ScalarSPH(l, m, θ(μ,ν), ϕ(μ,ν))
    else
        return ScalarSPH(l, m, μ, ν)
    end
end

function LaplaceOnASphere.ScalarSPH(SH::SphericalHarmonics{T}, l::Int, m::Int, kind::Symbol)::Array{Complex{T},1} where {T}
    if kind == :physical
        return map(SH, (μ,ν)->ScalarSPH(l,m,μ,ν, :physical))
    else
        return ScalarSPH(SH, l, m)
    end
end

function hinv(a::Int, b::Int, μ::T, ν::T)::Complex{T} where {T}
    if a == b == 1
        return 1
    elseif a == b == 2
        return (p^2)*cos(μ)^2 + csc(μ)^2
    else
        return -p*cos(μ)
    end
end

function sqrt_detg_by_deth(μ::T, ν::T)::Complex{T} where {T}
    dethinv = -hinv(1,2,μ,ν)*hinv(2,1,μ,ν) + hinv(1,1,μ,ν)*hinv(2,2,μ,ν)
    return sin(μ)*sqrt(dethinv)
end

function sqrt_deth_by_detg(μ::T, ν::T)::Complex{T} where {T}
    return 1/sqrt_detg_by_deth(μ, ν)
end

function sqrt_deth_by_detg_g_hinv(a::Int, b::Int, μ::T, ν::T)::Complex{T} where {T}
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

    F = filter(SH, Int(round(SH.N/2))) 

    grad = V
    div  = S*L*V̄
    Δ    = S̄*D*div*(H*grad)

    return Δ    
end

#---------------------------------------------------------------
# Compute the eigenvalues and the eigenvectors. 
# Now the eigensolver gives us a set of eigenvectors that 
# span the *same* (we hope) space as the original spherical
# harmonic eigenvectors. So we first 
# [1] Need to check if they indeed span the same space.  
# [2] If they do, we check if the eigenvectors are orthogonal
#     under the inner product defined by the metric. If not, 
#     we can make them orthogonal under the metric by a simple
#     scaling.
# [3] We then go a Gram-Schmidt orthogonalization that uses
#     the metric for computing the orthogonalization procedure.
#---------------------------------------------------------------

SH = SphericalHarmonics(12)
Δ  = laplace(SH)
F  = eigen(Δ)

@testset "laplace" begin
    Δ = laplace(SH)
    S = modal_to_nodal_scalar_op(SH) 
    S̄ = nodal_to_modal_scalar_op(SH)
    lmax = 5
    for l in 1:lmax
        for m in -l:l
            # FIXME: Why does it work only for m = 0? This kind
            # of makes sense, since the dependence on the coordinate
            # transformation would only show up for m != 0. 
            Ylm = map(SH, (μ,ν)->ScalarSPH(l,m, μ, ν, :physical))
            @test_skip isapprox(S*Δ*S̄*Ylm, -l*(l+1)*Ylm; atol=1e-10)
        end
    end
end;

#---------------------------------------------------------------
# To check whether they belong to the same vector subspace, 
# each vector in one of the spaces must be a linear combination
# of the other. We want
# V1 = <Y10, V1> Y10 + <Y11, V1> Y11 ...
# Then we need a routine to project onto Y10, Y11.
#---------------------------------------------------------------

function LinearAlgebra.dot(SH::SphericalHarmonics, u::Array{T,1}, l::Int, m::Int)::T where {T}
    # WARNING: Make sure you don't switch on the coordinate transformation
    @assert length(u) == SH.N*(2*SH.N) 
    ulm = nodal_to_modal_scalar_op(SH)*u
    for index in CartesianIndices(ulm)
        l_, m_ = split(index.I...)
        if (l == l_) && (m == m_) 
            return ulm[index]
        end
    end
    return Complex(0) 
end

function project(SH::SphericalHarmonics, V::Array{T,2}, l::Int)::Array{T,2} where {T}
    @assert size(V)[2] == 2l + 1
    W = similar(V)
    for m_ in 1:2l+1
        for m in -l:l
            Ylm = map(SH, (μ,ν)->ScalarSPH(l,m,μ,ν))
            clm = dot(SH, V[:, m_], l, m) 
            W[:, m_] = W[:, m_] + clm*Ylm
        end
    end
    return W
end

# TODO: Check if they belong to the same subspace
V = query(SH, F, 1, :nodal)
W = project(SH, query(SH, F, 1, :nodal), 1)
# @test maximum(abs.(W - V)) < 1e-12

V = query(SH, F, 2, :nodal)
W = project(SH, query(SH, F, 2, :nodal), 2)
# @test maximum(abs.(W - V)) < 1e-12

#---------------------------------------------------------------
# The laplace operator is symmetric. Hence, solver outputs eigenfunctions
# corresponding to different eigenvalues are orthogonal to each other, under
# the Euclidean metric. We need to find orthogonal vectors under the physical
# metric.
#---------------------------------------------------------------

function integrate(SH::SphericalHarmonics{T}, u::Array{Complex{T}})::Complex{T} where {T}
    H = scale_scalar(SH, sqrt_deth_by_detg)
    K = nodal_to_modal_scalar_op(SH)
    ulm = K*(H*u)
    return sqrt(4π)*ulm[1]
end

# int f \sqrt(h) dμ dν
# int f  \sqrt(h) / \sqrt(g)     \sqrt(g) du dν

function LinearAlgebra.dot(SH::SphericalHarmonics, u::Array{T,1}, v::Array{T,1})::T where {T}
    # FIXME: Raise the index 
    return integrate(SH, conj(u).*v)
end

function project(SH::SphericalHarmonics, u::Array{T,1}, a::Array{T,1})::Array{T,1} where {T}
    return (dot(SH,u,a)/dot(SH,u,u))*u
end

function orthogonalize(SH::SphericalHarmonics, V::Array{T,2})::Array{T,2} where {T}
    @assert size(V)[2] == 3
    @assert rank(V) == 3
    A1, A2, A3 = V[:,1], V[:,2], V[:,3]
    U1 = A1
    U2 = A2 - project(SH, U1, A2) 
    U3 = A3 - project(SH, U1, A3) - project(SH, U2, A3) 
    return hcat(U1, U2, U3)
end

K   = query(SH, F, 0, :nodal)
V   = query(SH, F, 1, :nodal)
W   = query(SH, F, 2, :nodal)

# Orthogonalize the eigenvectors
Q = orthogonalize(SH, V)
@show dot(SH, Q[:, 1], Q[:, 1])
@show dot(SH, Q[:, 2], Q[:, 2])
@show dot(SH, Q[:, 3], Q[:, 3])
@show dot(SH, Q[:, 1], Q[:, 2])
@show dot(SH, Q[:, 2], Q[:, 3])
@show dot(SH, Q[:, 3], Q[:, 1])

# Look at the imaginary parts
@show maximum(abs.(imag.(Q[:, 1])))
@show maximum(abs.(imag.(Q[:, 2])))
@show maximum(abs.(imag.(Q[:, 3])))

# Plot the eigenfunctions
plot(SH, F, 3)
contourf(SH, F, 1)
