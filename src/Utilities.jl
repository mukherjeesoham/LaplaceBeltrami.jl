#---------------------------------------------------------------
# LaplaceOnASphere
# Soham 3/20
# Utitilies for collocation
#---------------------------------------------------------------

export max_coefficent_for_each_l, L2

function collocation(S::SphericalHarmonics{T}, i::Int, j::Int)::NTuple{2, T} where {T}
    # NOTE: Driscoll and Healy points have a collocation point at the poles. 
    #       Currently using ECP collocation points. 
    θ = (i-1/2)*(π/S.N)  # [0,  π]
    ϕ = (j-1)*(π/S.N)    # [0, 2π]
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

function LInf(x::Array{T,1}) where {T}
    return maximum(abs.(x))
end

function quad(SH::SphericalHarmonics{T}, s::Array{Complex{T},1})::T where {T}
    integral = (nodal_to_modal_scalar_op(SH)*s)[1] 
    @assert imag(integral) < 1e-12
    return sqrt(4π)*real(integral)
end

function L1(SH::SphericalHarmonics{T}, x::Array{Complex{T},1})::T where {T}
    return quad(SH, abs.(x))
end

function L2(SH::SphericalHarmonics{T}, x::Array{Complex{T},1})::T where {T}
    return sqrt(quad(SH, conj(x).*x))
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

function query(SH::SphericalHarmonics, F::Eigen, l::Int, kind::Symbol)
    vectors   = Complex{Float64}[]
    S         = modal_to_nodal_scalar_op(SH)
    roundFval = real.(round.(F.values))
    for index in CartesianIndices(roundFval)
        if roundFval[index] == -l*(l+1)
            if kind == :nodal 
                append!(vectors, S*F.vectors[:, index])
            elseif kind == :modal
                append!(vectors, F.vectors[:, index])
            end
        end
    end
    return reshape(vectors, (:, 2l+1))
end

function max_coefficent_for_each_l(SH::SphericalHarmonics{T}, umodal::Array{Complex{T},1}) where {T}
    labs = zeros(T, SH.lmax+1)
    for l in 0:SH.lmax
        lm = zeros(T, 2*l + 1)
        for m in -l:l
            lm[m+l+1] = abs(umodal[join(l,m)])
        end
        labs[l+1] = maximum(lm)
    end
    return labs 
end

function Base. filter(S::SphericalHarmonics{T}, lcutoff::Int)::Array{T,2} where {T}
    lmax = S.lmax
    A = zeros(T, (lmax)^2 + 2*(lmax)+1, (lmax)^2 + 2*(lmax)+1)
    @inbounds for index in CartesianIndices(A)
        P, Q = index.I
        l,m = split(P)
        p,q = split(Q)
        if (l, m) == (p,q)
            if l > lcutoff
                A[index] = 0.0
            else
                A[index] = 1.0
            end
        end
    end
    return A
end
