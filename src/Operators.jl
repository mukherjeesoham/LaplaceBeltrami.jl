#---------------------------------------------------------------
# LaplaceOnASphere
# Soham 3/20
# Compute operators for transforming between spaces 
#---------------------------------------------------------------

using LinearAlgebra
export modal_to_nodal_scalar_op, nodal_to_modal_scalar_op
export modal_to_nodal_vector_op, nodal_to_modal_vector_op
export scale_scalar, scale_vector, scale_lmodes

function modal_to_nodal_scalar_op(S::SphericalHarmonics{T})::Array{Complex{T}, 2} where {T}
    lmax, n = S.lmax, S.N
    A = zeros(Complex{T}, n*(2*n), (lmax)^2 + 2*(lmax)+1)
    for index in CartesianIndices(A)
        (i,j) = split(index.I[1], n)
        (l,m) = split(index.I[2])
        (θ,ϕ) = collocation(S,i,j)
        A[index] = ScalarSPH(l, m, θ, ϕ)
    end
    return A
end

function nodal_to_modal_scalar_op(S::SphericalHarmonics{T})::Array{Complex{T}, 2} where {T}
    return pinv(modal_to_nodal_scalar_op(S))
end

function modal_to_nodal_vector_op(S::SphericalHarmonics{T})::Array{Complex{T}, 2} where {T}
    lmax, n = S.lmax, S.N
    A = zeros(Complex{T}, 2*(n*(2*n)), (lmax)^2 + 2*(lmax)+1)
    for index in CartesianIndices(A)
        (i,j,a) = split3(index.I[1], n)
        (l,m)   = split(index.I[2])
        (θ,ϕ)   = collocation(S,i,j)
        A[index] = GradSH(a, l, m, θ, ϕ)
    end
    return A
end

function nodal_to_modal_vector_op(S::SphericalHarmonics{T})::Array{Complex{T}, 2} where {T}
    return pinv(modal_to_nodal_vector_op(S))
end

function scale_scalar(S::SphericalHarmonics{T}, g::Function)::Array{Complex{T}, 2} where {T}
    N = S.N
    A = zeros(Complex{T}, N*2N, N*2N)
    for index in CartesianIndices(A)
        P, Q = index.I
        m,n = split(P, N)
        k,l = split(Q, N)
        θ, ϕ  = collocation(S, k, l)
        if (m, n) == (k,l)
            A[index] = g(θ, ϕ) 
        end
    end
    return A
end

function scale_lmodes(S::SphericalHarmonics{T}, g::Function)::Array{T,2} where {T}
    lmax = S.lmax
    A = zeros(T, (lmax)^2 + 2*(lmax)+1, (lmax)^2 + 2*(lmax)+1)
    for index in CartesianIndices(A)
        P, Q = index.I
        l,m = split(P)
        p,q = split(Q)
        if (l, m) == (p,q)
           A[index] = g(l,m) 
        end
    end
    return A

end

function scale_vector(S::SphericalHarmonics{T}, g::Function)::Array{T, 2} where {T}
    N = S.N
    A = zeros(Complex{T}, 2N*2N, 2N*2N)
    for index in CartesianIndices(A)
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
