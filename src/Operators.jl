#---------------------------------------------------------------
# LaplaceOnASphere
# Soham 3/20
# Compute operators for transforming between spaces 
#---------------------------------------------------------------

using LinearAlgebra, FastGaussQuadrature
export modal_to_nodal_scalar_op, nodal_to_modal_scalar_op
export modal_to_nodal_grad_op, nodal_to_modal_grad_op
export modal_to_nodal_curl_op, nodal_to_modal_curl_op
export scalar_op, grad_op, curl_op
export scale_scalar, scale_vector, scale_lmodes

function nodal_to_modal_scalar_op(SH::SphericalHarmonics{T})::Array{Complex{T},2} where {T}
    lmax, N = SH.lmax, SH.N
    nodes, weights = gausslegendre(N)
    theta = -(T(π)/2).*nodes .+ T(π)/2  
    phi   = [(j-1)*(T(π)/N) for j in 1:2N] 
    A     = zeros(Complex{T}, lmax^2 + 2lmax + 1, 2N^2)
    @inbounds for index in CartesianIndices(A)
        lm, ij = Tuple(index) 
        l, m   = split(lm) 
        i, j   = split(ij, N) 
        A[index] =  (π^2/2N)*weights[i]*conj(ScalarSH(l, m, theta[i], phi[j]))*sin(theta[i])
    end
    return A
end

function modal_to_nodal_scalar_op(SH::SphericalHarmonics{T})::Array{Complex{T},2} where {T}
    lmax, N = SH.lmax, SH.N
    nodes, weights = gausslegendre(N)
    theta = -(T(π)/2).*nodes .+ T(π)/2  
    phi   = [(j-1)*(T(π)/N) for j in 1:2N] 
    A     = zeros(Complex{T}, 2N^2, lmax^2 + 2lmax + 1)
    @inbounds for index in CartesianIndices(A)
        ij, lm = Tuple(index) 
        l, m   = split(lm) 
        i, j   = split(ij, N) 
        A[index] =  ScalarSH(l, m, theta[i], phi[j])
    end
    return A
end

function nodal_to_modal_grad_op(SH::SphericalHarmonics{T})::Array{Complex{T},2} where {T}
    lmax, N = SH.lmax, SH.N
    nodes, weights = gausslegendre(N)
    theta = -(T(π)/2).*nodes .+ T(π)/2  
    phi   = [(j-1)*(T(π)/N) for j in 1:2N] 
    A     = zeros(Complex{T}, lmax^2 + 2lmax + 1, 4N^2)
    @inbounds for index in CartesianIndices(A)
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

function modal_to_nodal_grad_op(SH::SphericalHarmonics{T})::Array{Complex{T},2} where {T}
    lmax, N = SH.lmax, SH.N
    nodes, weights = gausslegendre(N)
    theta = -(T(π)/2).*nodes .+ T(π)/2  
    phi   = [(j-1)*(T(π)/N) for j in 1:2N] 
    A     = zeros(Complex{T}, 4N^2, lmax^2 + 2lmax + 1)
    @inbounds for index in CartesianIndices(A)
        ija, lm  = Tuple(index) 
        l, m     = split(lm) 
        i, j, a  = split3(ija, N) 
        A[index] = GradSH(a, l, m, theta[i], phi[j])
    end
    return A
end

function nodal_to_modal_curl_op(SH::SphericalHarmonics{T})::Array{Complex{T},2} where {T}
    lmax, N = SH.lmax, SH.N
    nodes, weights = gausslegendre(N)
    theta = -(T(π)/2).*nodes .+ T(π)/2  
    phi   = [(j-1)*(T(π)/N) for j in 1:2N] 
    A     = zeros(Complex{T}, lmax^2 + 2lmax + 1, 4N^2)
    @inbounds for index in CartesianIndices(A)
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

function modal_to_nodal_curl_op(SH::SphericalHarmonics{T})::Array{Complex{T},2} where {T}
    lmax, N = SH.lmax, SH.N
    nodes, weights = gausslegendre(N)
    theta = -(T(π)/2).*nodes .+ T(π)/2  
    phi   = [(j-1)*(T(π)/N) for j in 1:2N] 
    A     = zeros(Complex{T}, 4N^2, lmax^2 + 2lmax + 1)
    @inbounds for index in CartesianIndices(A)
        ija, lm  = Tuple(index) 
        l, m     = split(lm) 
        i, j, a  = split3(ija, N) 
        A[index] = CurlSH(a, l, m, theta[i], phi[j])
    end
    return A
end

function scalar_op(SH::SphericalHarmonics{T})::NTuple{2, Array{Complex{T}}} where {T}
    return (modal_to_nodal_scalar_op(SH), nodal_to_modal_scalar_op(SH))
end

function grad_op(SH::SphericalHarmonics{T})::NTuple{2, Array{Complex{T}}} where {T}
    return (modal_to_nodal_grad_op(SH), nodal_to_modal_grad_op(SH))
end

function curl_op(SH::SphericalHarmonics{T})::NTuple{2, Array{Complex{T}}} where {T}
    return (modal_to_nodal_curl_op(SH), nodal_to_modal_curl_op(SH))
end

function scale_lmodes(S::SphericalHarmonics{T}, g::Function)::Array{T,2} where {T}
    lmax = S.lmax
    A = zeros(T, (lmax)^2 + 2*(lmax)+1, (lmax)^2 + 2*(lmax)+1)
    @inbounds for index in CartesianIndices(A)
        P, Q = index.I
        l,m = split(P)
        p,q = split(Q)
        if (l, m) == (p,q)
           A[index] = g(l,m) 
        end
    end
    return A

end

function scale_scalar(S::SphericalHarmonics{T}, g::Function)::Array{Complex{T}, 2} where {T}
    N = S.N
    A = zeros(Complex{T}, N*2N, N*2N)
    @inbounds for index in CartesianIndices(A)
        P, Q = index.I
        m,n = split(P, N)
        k,l = split(Q, N)
        if (m, n) == (k,l)
            θ, ϕ  = collocation(S, k, l)
            A[index] = g(θ, ϕ) 
        end
    end
    return A
end

function scale_vector(S::SphericalHarmonics{T}, g::Function)::Array{Complex{T}, 2} where {T}
    N = S.N
    A = zeros(Complex{T}, 2N*2N, 2N*2N)
    @inbounds for index in CartesianIndices(A)
        P, Q = index.I
        m,n,a = split3(P, N)
        k,l,b = split3(Q, N)
        if (m,n) == (k,l)
            θ, ϕ  = collocation(S, k, l)
            A[index] = g(a, b, θ,ϕ)
        end
    end
    return A 
end

