#---------------------------------------------------------------
# LaplaceOnASphere
# Soham 3/20
# Utitilies for collocation
#---------------------------------------------------------------

using FastGaussQuadrature, LinearAlgebra
export L2, quad, dot, gramschmidt!, onlyreal

function collocation(S::SphericalHarmonics{T}, i::Int, j::Int)::NTuple{2, T} where {T}
    lmax, N = S.lmax, S.N
    nodes, weights = gausslegendre(N)
    theta = -(T(π)/2).*nodes .+ T(π)/2  
    phi   = (j-1)*(T(π)/N)  
    return (theta[i], phi)
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

function integral(SH::SphericalHarmonics{T})::Array{Complex{T},2} where {T}
    lmax, N = SH.lmax, SH.N
    nodes, weights = gausslegendre(N)
    theta = -(T(π)/2).*nodes .+ T(π)/2  
    phi   = [(j-1)*(T(π)/N) for j in 1:2N] 
    A     = zeros(Complex{T}, lmax^2 + 2lmax + 1, 2N^2)
    @inbounds for index in CartesianIndices(A)
        lm, ij = Tuple(index) 
        l, m   = split(lm) 
        i, j   = split(ij, N) 
        A[index] =  (π^2/2N)*weights[i]*sin(theta[i])
    end
    return A
end

function quad(SH::SphericalHarmonics{T}, u::Array{Complex{T},1})::Complex{T} where {T}
    SHF = SphericalHarmonics{Float64}(SH.lmax, 2*SH.N)
    nodes, weights = gausslegendre(SHF.N)
    S̄ = nodal_to_modal_scalar_op(SH)
    S = modal_to_nodal_scalar_op(SHF)
    H = integral(SHF)
    integ = H*(S*(S̄*u))
    return integ[1]
end

function L2(SH::SphericalHarmonics{T}, x::Array{Complex{T},1})::T where {T}
    return real(sqrt(quad(SH, conj(x).*x)))
end

function onlyreal(u::Array)::Bool
    maximum(abs.(imag.(u))) < 1e-10
end

function Base. reshape(SH::SphericalHarmonics, u::Array{T,1}, L::Symbol)::Array{T,2} where {T}
    if L == :scalar
        return reshape(u, (SH.N, 2*SH.N))
    elseif L == :vector 
        return reshape(u, (SH.N, 2*SH.N, 2))
    end
end

function LinearAlgebra.dot(SH::SphericalHarmonics{T}, u::Array{Complex{T},1}, v::Array{Complex{T},1})::Complex{T} where {T} 
    H = scale_scalar(SH, (μ,ν)->sqrt(deth(μ,ν)/detq(μ,ν)))
    return quad(SH, H*(conj(u).*v))
end

function gramschmidt!(SH::SphericalHarmonics{T}, u::Array{Complex{T},2})::Array{Complex{T},2} where {T}
    u[:,1] = u[:,1]/sqrt(dot(SH, u[:,1], u[:,1]))
    u[:,2] = u[:,2] - dot(SH, u[:,1], u[:,2]).*u[:,1]
    u[:,2] = u[:,2]/sqrt(dot(SH, u[:,2], u[:,2]))
    u[:,3] = u[:,3] - dot(SH, u[:,1], u[:,3]).*u[:,1] - dot(SH, u[:,2], u[:,3]).*u[:,2]
    u[:,3] = u[:,3]/sqrt(dot(SH, u[:,3], u[:,3]))
    return u
end

