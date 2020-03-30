#---------------------------------------------------------------
# LaplaceOnASphere
# Soham 3/20
# Compute operators for transforming between spaces 
#---------------------------------------------------------------

using LinearAlgebra
export modal_to_nodal_scalar_op, nodal_to_modal_scalar_op
export modal_to_nodal_vector_op, nodal_to_modal_vector_op
export scaling_scalar_op, scaling_vector_op 

function modal_to_nodal_scalar_op(S::SphericalHarmonics{T})::Array{Complex{T}, 2} where {T}
    lmax, n = S.lmax, S.N
    A = zeros(Complex{T}, n*(2*n), (lmax)^2 + 2*(lmax) + 1)
    for index in CartesianIndices(A)
        (i,j) = split(index.I[1], n)
        (l,m) = split(index.I[2])
        (θ,ϕ) = grid(S,i,j)
        A[index] = ScalarSPH(l, m, θ, ϕ)
    end
    return A
end

function nodal_to_modal_scalar_op(S::SphericalHarmonics{T})::Array{Complex{T}, 2} where {T}
    return pinv(modal_to_nodal_scalar_op(S))
end

function modal_to_nodal_vector_op(S::SphericalHarmonics{T})::Array{Complex{T}, 2} where {T}
    lmax, n = S.lmax, S.N
    A = zeros(Complex, 2*(n*(2*n)), (lmax)^2 + 2*(lmax) + 1)
    for index in CartesianIndices(A)
        (i,j,a) = split3(index.I[1], n)
        (l,m)   = split(index.I[2])
        (θ,ϕ)   = grid(S,i,j)
        A[index] = VectorSPH(l, m, θ, ϕ)[a]
    end
    return A
end

function nodal_to_modal_vector_op(S::SphericalHarmonics{T})::Array{Complex{T}, 2} where {T}
    return pinv(modal_to_nodal_vector_op(S))
end

function scaling_scalar_op(S::SphericalHarmonics{T}, u::Function)::Array{Complex{T}, 2} where {T}
    n = S.N
    A = Diagonal(zeros(N*2N)) 
    for index in 1:N*2N
        i, j = split(index, N) 
        A[index, index] = u(grid(S,i,j)...)
    end
    return A
end

function scaling_vector_op(S::SphericalHarmonics{T}, u1::Function, u2::Function)::Array{Complex{T}, 2} where {T}
    N = S.n
    A = zeros(2N*2N, 2N*2N)
    for index in CartesianIndices(A)
        if index.I[1] == index.I[2]
            i,j,a = split3(index.I[1], N)
            A[index] = (a == 1 ? u1(grid(S,i,j)...) : u2(grid(S,i,j)...))
        end
    end
    return A
end
