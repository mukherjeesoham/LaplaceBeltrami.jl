#---------------------------------------------------------------
# LaplaceOnASphere
# Soham 3/20
# Compute operators for transforming between spaces 
# UPDATE: Moved to gauss quadrature for going from
# nodal to modal basis.
#---------------------------------------------------------------

using LinearAlgebra, FastGaussQuadrature
export grad, scalar_op

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
        A[index] =  (π^2/2N)*weights[i]*conj(Ylm(l, m, theta[i], phi[j]))*sin(theta[i])
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
        A[index] =  Ylm(l, m, theta[i], phi[j])
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
            A[index] = (1/(l*(l+1)))*(π^2/2N)*weights[i]*conj(Glm(a, l, m, theta[i], phi[j]))*(1 + (a-1)*cot(theta[i])^2)*sin(theta[i])
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
        A[index] = Glm(a, l, m, theta[i], phi[j])
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
            A[index] = (1/(l*(l+1)))*(π^2/2N)*weights[i]*conj(Clm(a, l, m, theta[i], phi[j]))*(1 + (a-1)*cot(theta[i])^2)*sin(theta[i])
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
        A[index] = Clm(a, l, m, theta[i], phi[j])
    end
    return A
end

function scale_lmodes(S::SphericalHarmonics{T})::Array{T,2} where {T}
    lmax = S.lmax
    A = zeros(T, (lmax)^2 + 2*(lmax)+1, (lmax)^2 + 2*(lmax)+1)
    @inbounds for index in CartesianIndices(A)
        P, Q = index.I
        l,m = split(P)
        p,q = split(Q)
        if (l, m) == (p,q)
            A[index] = -l*(l+1) 
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

function Base. div(S::SphericalHarmonics{T}) where {T} 
    return modal_to_nodal_scalar_op(S)*scale_lmodes(S)*nodal_to_modal_grad_op(S)
end

function grad(S::SphericalHarmonics{T}) where {T}
    return modal_to_nodal_grad_op(S)*nodal_to_modal_scalar_op(S) 
end

function scalar_op(S::SphericalHarmonics{T}) where {T}
    return (modal_to_nodal_scalar_op(S), nodal_to_modal_scalar_op(S))
end
