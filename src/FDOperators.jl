#---------------------------------------------------------------
# LaplaceOnASphere
# Soham 5/20
# Construct FD operators
#---------------------------------------------------------------

using LinearAlgebra

export D1D1, D1D2, D2D1, D2D2, S2D, X1, X2

function D1D1(::Type{T}, N::Int)::Array{T,2} where {T}
    A = diagm(-1=>-ones(N-1), 1=>ones(N-1))
    # Use a different 1st order accurate stencil at the boundary
    A[1,1] = -1
    A[end, end] = 1
    A[1,:] = A[1,:]/(π/N)
    A[end,:] = A[end,:]/(π/N)
    A[2:end-1,:] = A[2:end-1,:]/(2π/N)
    return A
end

function D1D2(::Type{T}, N::Int)::Array{T,2} where {T}
    A = diagm(-1=>-ones(T, N-1), 1=>ones(T, N-1)) 
    A[1, end] = -1
    A[end, 1] =  1
    return A/(4π/N)
end

function D2D1(::Type{T}, N1::Int, N2::Int)::Array{T,2} where {T}
    I2 = diagm(0=>ones(T, N2))
    D1 = D1D1(T, N1) 
    return kron(I2, D1)
end

function D2D2(::Type{T}, N1::Int, N2::Int)::Array{T,2} where {T}
    I1 = diagm(0=>ones(T, N1))
    D2 = D1D2(T, N2) 
    return kron(D2, I1)
end

function S2D(::Type{T}, N1::Int, N2::Int, map::Function)::Array{T,2} where {T}
    A = [map(X1(T,i,N1), X2(T,j,N2)) for i in 1:N1, j in 1:N2] 
    return diagm(0=>vec(A))
end

function X1(::Type{T}, i::Int, N::Int)::T where {T}
    return (π/N)*(i + 1/2)
end

function X2(::Type{T}, i::Int, N::Int)::T where {T}
    return (2π/N)*i
end

