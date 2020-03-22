export grid, split3, join3, map_to_grid

function Base. join(i::Int, j::Int, N::Int)::Int
    @assert i <= N
    @assert j <= 2N
    return LinearIndices((N, 2N))[i,j]
end

function Base. split(ij::Int, N::Int)
    return CartesianIndices((N, 2N))[ij].I
end

function Base. join(l::Int, m::Int)::Int
    @assert abs(m) <= l 
    return l^2 + l + m + 1 
end

function Base. split(lm::Int)::NTuple{2, Int}
    if lm == 1
        return (0,0)
    else
        for l in 1:lm, m in -l:l
            if l^2 + l + m + 1 == lm
                @assert abs(m) <= l
                return (l,m)
            end
        end
    end
end

function grid(i::Int, j::Int, N::Int)::NTuple{2, Number}
    θ = i*π/(N+1) # [0,  π]
    ϕ = (j-1)*π/N # [0, 2π]
    return (θ, ϕ)
end

function join3(i::Int, j::Int, a::Int, N::Int)::Int
    return (a-1)*N*(2N) + join(i,j,N)
end

function split3(ija::Int, N::Int)::NTuple{3, Int}
    q, r = divrem(ija, N*2N)
    if r == 0
        return (split(N*2N, N)..., q)
    else
        return (split(r, N)..., q+1) 
    end
end

function map_to_grid(S::SphericalHarmonics, u::Function)::Array{Number, 1}
    uvec = Array{Number, 1}(undef, S.n*2(S.n))
    for index in CartesianIndices(uvec)
        i, j = split(index.I[1], S.n)
        θ, ϕ = grid(i, j, S.n)
        uvec[index] = u(θ, ϕ)
    end
    return uvec
end

function map_to_grid(S::SphericalHarmonics, uθ::Function, uϕ::Function)::Array{Number, 1}
    uvec = Array{Number, 1}(undef, 2*S.n*2(S.n))
    for index in CartesianIndices(uvec)
        i, j, a = split3(index.I[1], S.n)
        θ, ϕ = grid(i, j, S.n)
        uvec[index] = (a==1 ? uθ(θ, ϕ) : uϕ(θ, ϕ))
    end
    return uvec
end

