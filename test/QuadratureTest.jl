#---------------------------------------------------------------
#e LaplaceOnASphere
# Soham 8/20
# Construct a smooth z-coordinate using
# spherical harmonics
# Use more accurate Scalar Spherical Harmonics
#---------------------------------------------------------------

using FastGaussQuadrature, LinearAlgebra

#---------------------------------------------------------------
# Scalar Operators
#---------------------------------------------------------------
function nodal_to_modal_scalar_op_quadrature(SH::SphericalHarmonics{T})::Array{Complex{T},2} where {T}
    lmax, N = SH.lmax, SH.N
    nodes, weights = gausslegendre(N)
    theta = -(T(π)/2).*nodes .+ T(π)/2  
    phi   = [(j-1)*(T(π)/N) for j in 1:2N] 
    A     = zeros(Complex{T}, lmax^2 + 2lmax + 1, 2N^2)
    for index in CartesianIndices(A)
        lm, ij = Tuple(index) 
        l, m   = split(lm) 
        i, j   = split(ij, N) 
        A[index] =  (π^2/2N)*weights[i]*conj(ScalarSH(l, m, theta[i], phi[j]))*sin(theta[i])
    end
    return A
end

function modal_to_nodal_scalar_op_quadrature(SH::SphericalHarmonics{T})::Array{Complex{T},2} where {T}
    lmax, N = SH.lmax, SH.N
    nodes, weights = gausslegendre(N)
    theta = -(T(π)/2).*nodes .+ T(π)/2  
    phi   = [(j-1)*(T(π)/N) for j in 1:2N] 
    A     = zeros(Complex{T}, 2N^2, lmax^2 + 2lmax + 1)
    for index in CartesianIndices(A)
        ij, lm = Tuple(index) 
        l, m   = split(lm) 
        i, j   = split(ij, N) 
        A[index] =  ScalarSH(l, m, theta[i], phi[j])
    end
    return A
end

#---------------------------------------------------------------
# Maps 
#---------------------------------------------------------------
function map_quadrature(SH::SphericalHarmonics{T}, u::Function)::Array{Complex{T},1} where {T}
    lmax, N = SH.lmax, SH.N
    nodes, weights = gausslegendre(N)
    theta = -(T(π)/2).*nodes .+ T(π)/2  
    phi   = [(j-1)*(T(π)/N) for j in 1:2N] 
    U     = zeros(Complex{T}, 2N^2)
    for index in CartesianIndices(U)
        ij = Tuple(index)[1]
        i ,j = split(ij, N) 
        U[index] = u(theta[i], phi[j])
    end
    return U
end

function map_quadrature(SH::SphericalHarmonics{T}, u1::Function, u2::Function)::Array{Complex{T},1} where {T}
    lmax, N = SH.lmax, SH.N
    nodes, weights = gausslegendre(N)
    theta = -(T(π)/2).*nodes .+ T(π)/2  
    phi   = [(j-1)*(T(π)/N) for j in 1:2N] 
    U     = zeros(Complex{T}, 4N^2)
    for index in CartesianIndices(U)
        ija = Tuple(index)[1]
        i ,j, a  = split3(ija, N) 
        U[index] = -(a - 2)*u1(theta[i], phi[j]) + (a - 1)*u2(theta[i], phi[j]) 
    end
    return U
end

#---------------------------------------------------------------
# Grad Operators
#---------------------------------------------------------------
function nodal_to_modal_grad_op_quadrature(SH::SphericalHarmonics{T})::Array{Complex{T},2} where {T}
    lmax, N = SH.lmax, SH.N
    nodes, weights = gausslegendre(N)
    theta = -(T(π)/2).*nodes .+ T(π)/2  
    phi   = [(j-1)*(T(π)/N) for j in 1:2N] 
    A     = zeros(Complex{T}, lmax^2 + 2lmax + 1, 4N^2)
    for index in CartesianIndices(A)
        lm, ija  = Tuple(index) 
        l, m     = split(lm) 
        i, j, a  = split3(ija, N) 
        if l == 0
            A[index ] = 0
        else
            A[index] = (1/(l*(l+1)))*(π^2/2N)*weights[i]*conj(GradSH(a, l, m, theta[i], phi[j]))*(1 + (a-1)*cot(theta[i])^2)*sin(theta[i])
        end
    end
    return A
end

function modal_to_nodal_grad_op_quadrature(SH::SphericalHarmonics{T})::Array{Complex{T},2} where {T}
    lmax, N = SH.lmax, SH.N
    nodes, weights = gausslegendre(N)
    theta = -(T(π)/2).*nodes .+ T(π)/2  
    phi   = [(j-1)*(T(π)/N) for j in 1:2N] 
    A     = zeros(Complex{T}, 4N^2, lmax^2 + 2lmax + 1)
    for index in CartesianIndices(A)
        ija, lm  = Tuple(index) 
        l, m     = split(lm) 
        i, j, a  = split3(ija, N) 
        A[index] = GradSH(a, l, m, theta[i], phi[j])
    end
    return A
end

#---------------------------------------------------------------
# Curl Operators
#---------------------------------------------------------------
function nodal_to_modal_curl_op_quadrature(SH::SphericalHarmonics{T})::Array{Complex{T},2} where {T}
    lmax, N = SH.lmax, SH.N
    nodes, weights = gausslegendre(N)
    theta = -(T(π)/2).*nodes .+ T(π)/2  
    phi   = [(j-1)*(T(π)/N) for j in 1:2N] 
    A     = zeros(Complex{T}, lmax^2 + 2lmax + 1, 4N^2)
    for index in CartesianIndices(A)
        lm, ija  = Tuple(index) 
        l, m     = split(lm) 
        i, j, a  = split3(ija, N) 
        if l == 0
            A[index] = 0
        else
            A[index] = (1/(l*(l+1)))*(π^2/2N)*weights[i]*conj(CurlSH(a, l, m, theta[i], phi[j]))*(1 + (a-1)*cot(theta[i])^2)*sin(theta[i])
        end
    end
    return A
end

function modal_to_nodal_curl_op_quadrature(SH::SphericalHarmonics{T})::Array{Complex{T},2} where {T}
    lmax, N = SH.lmax, SH.N
    nodes, weights = gausslegendre(N)
    theta = -(T(π)/2).*nodes .+ T(π)/2  
    phi   = [(j-1)*(T(π)/N) for j in 1:2N] 
    A     = zeros(Complex{T}, 4N^2, lmax^2 + 2lmax + 1)
    for index in CartesianIndices(A)
        ija, lm  = Tuple(index) 
        l, m     = split(lm) 
        i, j, a  = split3(ija, N) 
        A[index] = CurlSH(a, l, m, theta[i], phi[j])
    end
    return A
end

#---------------------------------------------------------------
# Test on functions 
#---------------------------------------------------------------
SH = SphericalHarmonics{Float64}(13, 204)
u  = map(SH, (μ,ν)->ScalarSH(3,2,μ,ν) + ScalarSH(3,-1,μ,ν)) 
S, S̄ = scalar_op(SH)

ū  = S̄*u 
for index in CartesianIndices(ū)
    if abs(ū[index]) > 1e-12
        l, m = split(index.I[1])
        @show l, m, ū[index] 
    end
end

ũ  = S*ū
@show L2(SH, u - ũ)

w = map(SH, (μ,ν)->GradSH(1,2,2,μ,ν) + CurlSH(1,2,-1,μ,ν) + CurlSH(1,2,2,μ,ν),
            (μ,ν)->GradSH(2,2,2,μ,ν) + CurlSH(2,2,-1,μ,ν) + CurlSH(2,2,2,μ,ν))
G, Ḡ = grad_op(SH)
C, C̄ = curl_op(SH)

w̄g = Ḡ*w
w̄c = C̄*w

for index in CartesianIndices(w̄g)
    if abs(w̄g[index]) > 1e-12
        l, m = split(index.I[1])
        @show l, m, w̄g[index] 
    end
end

for index in CartesianIndices(w̄c)
    if abs(w̄c[index]) > 1e-12
        l, m = split(index.I[1])
        @show l, m, w̄c[index] 
    end
end

w̃ = G*w̄g + C*w̄c
g = map(SH, (μ,ν)->1, (μ,ν)->(1/sin(μ)^2))
e = w - w̃
@show norm(e)
