#---------------------------------------------------------------
# LaplaceOnASphere
# Soham 3/20
# Utitilies for collocation
#---------------------------------------------------------------

export LInf, L2, L1, onlyreal

function collocation(S::SphericalHarmonics{T}, i::Int, j::Int)::NTuple{2, T} where {T}
    # NOTE: Driscoll and Healy points have a collocation point at the poles. 
    # This won't work for us at the moment since Ψ breaks down at the poles.
    # Currently using ECP collocation points. 
    θ = (i-1/2)*(π/S.N)  # [0,  π]
    ϕ = (j-1)*(2π/S.N)   # [0, 2π]
    return (θ, ϕ)
end

function Base. map(S::SphericalHarmonics{T}, u::Function)::Array{Complex{T}, 1} where {T}
    N = S.N
    uvec = Array{Complex{T}, 1}(undef, N*2N)
    for index in CartesianIndices(uvec)
        i, j = split(index.I[1], N)
        θ, ϕ = collocation(S, i, j)
        uvec[index] = u(θ, ϕ)
    end
    return uvec
end
        
function Base. map(S::SphericalHarmonics{T}, uθ::Function, uϕ::Function)::Array{Complex{T}, 1} where {T}
    N = S.N
    uvec = Array{Complex{T}, 1}(undef, 2*(N*2N))
    for index in CartesianIndices(uvec)
        i, j, a = split3(index.I[1], N)
        θ, ϕ = collocation(S, i, j)
        uvec[index] = (a==1 ? uθ(θ, ϕ) : uϕ(θ, ϕ))
    end
    return uvec
end

function analyticYlm(S::SphericalHarmonics{T}, ulm::Array{Complex{T}, 1}, θ::T, ϕ::T)::Complex{T} where {T}
    u = Complex(T(0))
    for l in 0:S.lmax, m in -l:l
        u += ulm[join(l,m)]*ScalarSPH(l, m, θ, ϕ)
    end 
    return u
end

function analyticΨlm(S::SphericalHarmonics{T}, ulm::Array{Complex{T}, 1}, θ::T, ϕ::T)::Complex{T} where {T}
    u = Complex(T(0))
    for l in 0:S.lmax, m in -l:l
        u += ulm[join(l,m)]*VectorSPH(l, m, θ, ϕ)[a]
    end 
    return u
end

function LInf(x::Array{T,1}) where {T}
    return maximum(abs.(x))
end

function L1(x::Array{T,1}) where {T}
    return sum(abs.(x))
end

function L2(x::Array{T,1}) where {T}
    return sqrt(sum(abs.(x).^2))
end

function onlyreal(u::Array)::Bool
    maximum(abs.(imag.(u))) < 1e-10
end

function Base. reshape(SH::SphericalHarmonics, u::Array{T,1})::Array{T,2} where {T}
    return reshape(u, (SH.N, 2*SH.N))
end

