#---------------------------------------------------------------
# LaplaceOnASphere
# Soham 5/20
# Construct new sparse FD operators that respect
# the symmetries around the poles for scalars and vectors
#---------------------------------------------------------------

using FastSphericalHarmonics
using SparseArrays, LinearAlgebra 
export npoints, map, Dθ, Dϕ, Dθ̄, Dϕ̄
export collocation, dscalar, dvector

function npoints(lmax::Int)::NTuple{2, Int}
    N = lmax + 1
    M = 2N - 1 
    return (N, 2M)
end

function collocation(i::Int, j::Int, ni::Int, nj::Int)
    # TODO: Add ability to compute for arbitrary i and j to test
    # the wrap function
    N = ni 
    M = nj
    θ, ϕ = (π / N * (0.5:(N - 0.5)), 2π / M * (0:(M - 1)))
    return θ[i], ϕ[j]
end

function Base.map(u::Function, ni::Int, nj::Int)
    scalar = zeros(ni, nj)
    return [u(collocation(index.I..., ni, nj)...) for index in CartesianIndices(scalar)]
end

function Base.join(i::Int, j::Int, ni::Int, nj::Int)::Int
    return LinearIndices((ni, nj))[i,j]
end

function Base.split(ij::Int, ni::Int, nj::Int)
    return CartesianIndices((ni, nj))[ij].I
end

function wrap(i::Int, j::Int, ni::Int, nj::Int)
    @assert iseven(nj)
    if i < 1 # north pole
        i = 1 - i 
        j = mod1(j + nj÷2, nj)
    elseif i > ni # south pole
        i = 2ni + 1 - i 
        j = mod1(j + nj÷2, nj)
    elseif j < 1 ||  j > nj  # crossing 0 or 2π
        j = mod1(j, nj)
    end
    return (i,j)
end

function parity(i::Int, ni::Int)
    if i < 1 || i > ni # across the poles
        return -1
    else
        return +1
    end
end

# FIXME: We get 2nd order convergence for the scalar
# derivative operators for a 4th order method.
function stencil(order::Int, index::Int)
    if order == 2
        return (-1/2, 0, 1/2)[index+2]
    elseif order == 4
        return (1/12, -2/3, 0, 2/3, -1/12)[index+3]
    elseif order == 6
        return (-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60)[index+4]
    elseif order == 8
        return (1/280, -4/105, 1/5, -4/5, 0, 4/5, -1/5, 4/105, -1/280)[index+5]	
    else
        errorexit("Invalid order")
    end
end

function Dθ(ni::Int, nj::Int, order::Int)
    I, J, V = (Int[], Int[], Float64[])
    for i in 1:ni, j in  1:nj
        for Δi in -order÷2:order÷2
            push!(I, join(i, j, ni, nj)) 
            push!(J, join(wrap(i+Δi, j, ni, nj)..., ni, nj)) 
            push!(V, stencil(order, Δi))
        end
    end 
    return dropzeros(sparse(I, J, V/(π/ni)))
end

function Dϕ(ni::Int, nj::Int, order::Int)
    I, J, V = (Int[], Int[], Float64[])
    for i in 1:ni, j in  1:nj
        for Δj in -order÷2:order÷2
            push!(I, join(i, j, ni, nj)) 
            push!(J, join(wrap(i, j+Δj, ni, nj)..., ni, nj)) 
            push!(V, stencil(order, Δj))
        end
    end 
    return dropzeros(sparse(I, J, V/(2π/nj)))
end

function Dθ̄(ni::Int, nj::Int, order::Int)
    I, J, V = (Int[], Int[], Float64[])
    for i in 1:ni, j in  1:nj
        for Δi in -order÷2:order÷2
            push!(I, join(i, j, ni, nj)) 
            push!(J, join(wrap(i+Δi, j, ni, nj)..., ni, nj)) 
            push!(V, parity(i+Δi, ni)*stencil(order, Δi))
        end
    end 
    return dropzeros(sparse(I, J, V/(π/ni)))
end

function Dϕ̄(ni::Int, nj::Int, order::Int)
    return Dϕ(ni, nj, order)
end

function dscalar(ni::Int, nj::Int, order::Int)
    return (Dθ(ni, nj, order), Dϕ(ni, nj, order))
end

function dvector(ni::Int, nj::Int, order::Int)
    return (Dθ̄(ni, nj, order), Dϕ̄(ni, nj, order))
end
