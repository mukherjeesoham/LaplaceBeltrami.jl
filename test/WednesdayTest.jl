#---------------------------------------------------------------
# LaplaceOnASphere
# Soham 7/20
# Test the pseudo-inverse with numerical quadrature
#---------------------------------------------------------------

using FastGaussQuadrature, LinearAlgebra, PyPlot

#---------------------------------------------------------------
# Use Gaussian quadrature to evaluate integrals on the sphere
# and compare with Mathematica
#---------------------------------------------------------------
function gaussquad(::Type{T}, f::Function, N::Int)::T where {T}
    # FIXME: There might be a bug in the integration code. Is the
    # error really supposed to be this large? 
    # FIXME: We might need to investigate a more accurate SPH routine.
    u = zeros(Complex{T}, N, 2N) 
    nodes, weights = gausslegendre(N)
    theta = -(T(π)/2).*nodes .+ T(π)/2  
    # FIXME: 1:2N
    phi   = [(j-1)*(T(π)/N) for j in 1:2N] 
    for index in CartesianIndices(u)    
        i, j = Tuple(index) 
        u[index] = (2/(2N))*weights[i]*f(theta[i], phi[j])*sin(theta[i]) 
    end
    integral = π*(π/2)*sum(u)
    @assert imag(integral) < 1e-5
    return real(integral)
end

f(μ, ν) = ScalarSPH(0,0,μ,ν)
@show gaussquad(Float64, f, 100) - 2*√π

#---------------------------------------------------------------
# Compute coefficents using gaussian quadrature
#---------------------------------------------------------------
function projectscalar(l::Int, m::Int, u::Function)
    f(μ, ν) = u(μ, ν)*conj(ScalarSPH(l,m,μ,ν))
    return gaussquad(Float64, f, 100)
end

function projectvector(l::Int, m::Int, U1::Function, U2::Function)
    f(μ, ν) = U1(μ, ν)*GradSH(1,l,m,μ,ν) + U2(μ, ν)*GradSH(2,l,m,μ,ν)
    return (1/(l*(l+1)))*gaussquad(Float64, f, 100)
end

function find_coefficents(u::Function, lmax::Int)
    lm = zeros((lmax)^2 + 2*(lmax)+1) 
    for index in CartesianIndices(lm) 
        l, m = split(Tuple(index)[1])
        lm[index] = projectscalar(l,m,u)  
    end
    return lm
end

function find_coefficents(U1::Function, U2::Function, lmax::Int)
    # NOTE: This accepts a vector (and not a co-vector) as input
    lm = zeros((lmax)^2 + 2*(lmax)+1) 
    for index in CartesianIndices(lm) 
        l, m = split(Tuple(index)[1])
        lm[index] = projectscalar(l,m,u)  
    end
    return lm
end

#---------------------------------------------------------------
# Test these functions, first on a vector, and then on a scalar, 
# and quantify the errors with pinv and Mathematica
#---------------------------------------------------------------
u(μ,ν) = ScalarSPH(19,17,μ,ν) + ScalarSPH(20,-18,μ,ν) + ScalarSPH(14,0,μ,ν)
ulm_integration = find_coefficents(u,20)

SH = SphericalHarmonics(20)
umap = map(SH, u)
ulm_leastsquares  = nodal_to_modal_scalar_op(SH)*umap

ulm_exact = zeros(20^2 + 40 + 1)
for index in CartesianIndices(ulm_exact)
    l, m = split(Tuple(index)[1])
    if (l,m) == (19,17) || (l,m) == (20,-18) || (l,m) == (14,0)
        ulm_exact[index] = 1
    end
end

# As expected, the error in the gaussian quadrature is much higher. 
@show norm(ulm_exact - ulm_integration)
@show norm(ulm_exact - ulm_leastsquares)

#---------------------------------------------------------------
# Let's try projecting a complicated function
# FIXME: Check for aliasing.
# Have enough points to remove aliasing.
#---------------------------------------------------------------
function analyticF(l::Int, m::Int, μ::T, ν::T)::Complex{T} where {T} 
    A1 = real(GradSH(1,1,1,μ,ν))  
    A2 = real(GradSH(2,1,1,μ,ν))  
    B1 = real(GradSH(1,3,1,μ,ν))  
    B2 = real(GradSH(2,3,1,μ,ν))  
    T1 = (√14*A1 - B1)/100 
    T2 = (√14*A2 - B2)/100 
    return ScalarSPH(l, m, μ + T1, ν + T2)
end

SH = SphericalHarmonics(30)
k(μ, ν) = analyticF(12,4,μ,ν)
kmap = map(SH, k)
klm  = nodal_to_modal_scalar_op(SH)*kmap 
# for index in CartesianIndices(klm)
    # l, m = split(Tuple(index)[1])
    # @show l,m, klm[index]
# end

function max_coefficent_for_each_l(SH::SphericalHarmonics{T}, umodal::Array{Complex{T},1}) where {T}
    labs = zeros(Complex{T}, SH.lmax+1)
    for l in 0:SH.lmax
        lm = zeros(T, 2*l + 1)
        for m in -l:l
            lm[m+l+1] = abs(umodal[join(l,m)])
        end
        labs[l+1] = maximum(lm)
    end
    return labs 
end

# semilogy(max_coefficent_for_each_l(SH, klm), "r-o")
# show()
@show projectscalar(12,4,k) - 0.9946229930286894
@show klm[join(12,4)] -  0.9946229930286894

#---------------------------------------------------------------
# Now try the vector spherical harmonic
# We will start with the gradient which is a co-vector
#---------------------------------------------------------------
function analytic∇F(l::Int, m::Int, μ::T, ν::T)::NTuple{2,Complex{T}} where {T} 
    A1 = real(GradSH(1,1,1,μ,ν))  
    A2 = real(GradSH(2,1,1,μ,ν))  
    B1 = real(GradSH(1,3,1,μ,ν))  
    B2 = real(GradSH(2,3,1,μ,ν))  
    T1 = (√14*A1 - B1)/100 
    T2 = (√14*A2 - B2)/100 
    
    dθdμ = 1 + (3/640)*sqrt(21/π)*cos(ν)*(sin(μ) - 3*sin(3μ)) 
    dθdν = (3/160)*sqrt(21/π)*cos(μ)*sin(ν)*sin(μ)^2 
    dϕdμ = (3/160)*sqrt(21/π)*cos(μ)*sin(ν)*sin(μ)^2 
    dϕdν = 1 + (1/160)*sqrt(21/π)*cos(ν)*sin(μ)^3 

    dFdθ = GradSH(1, l, m, μ + T1, ν + T2)   
    dFdϕ = GradSH(2, l, m, μ + T1, ν + T2)   

    ∇F1 = dFdθ*dθdμ + dFdϕ*dϕdμ 
    ∇F2 = dFdθ*dθdν + dFdϕ*dϕdν    

    return (∇F1, ∇F2) 
end

umap = map(SH, (μ,ν)->analytic∇F(12,4,μ,ν)[1], (μ,ν)->analytic∇F(12,4,μ,ν)[2])
# Raise index using the round-metric
Umap = map(SH, (μ,ν)->analytic∇F(12,4,μ,ν)[1], (μ,ν)->(1/sin(μ)^2)*analytic∇F(12,4,μ,ν)[2])

ulm  = nodal_to_modal_vector_op(SH)*umap
Ulm  = nodal_to_modal_vector_op(SH)*Umap

# semilogy(max_coefficent_for_each_l(SH, ulm), "r-o")
# semilogy(max_coefficent_for_each_l(SH, Ulm), "b-o")
# show()

#---------------------------------------------------------------
# Now try projecting a vector in a vector basis. i.e., 
# raise the index of GradSH
#---------------------------------------------------------------
function qinv(a::Int, b::Int, μ::T, ν::T)::T where {T}
    if a == b == 1
        return 1
    elseif a == b == 2
        return 1/sin(μ)^2 
    else 
        return 0
    end
end

function modal_to_nodal_vector_op_raised_index(S::SphericalHarmonics{T})::Array{Complex{T}, 2} where {T}
    lmax, n = S.lmax, S.N
    A = zeros(Complex{T}, 2*(n*(2*n)), (lmax)^2 + 2*(lmax)+1)
    for index in CartesianIndices(A)
        (i,j,a) = split3(index.I[1], n)
        (l,m)   = split(index.I[2])
        (θ,ϕ)   = collocation(S,i,j)
        A[index] = qinv(a,1,θ,ϕ)*GradSH(1, l, m, θ, ϕ) + qinv(a,2,θ,ϕ)*GradSH(2, l, m, θ, ϕ)
    end
    return A
end

function nodal_to_modal_vector_op_raised_index(S::SphericalHarmonics{T})::Array{Complex{T}, 2} where {T}
    return pinv(modal_to_nodal_vector_op_raised_index(S))
end

Ulm_raised_index  = nodal_to_modal_vector_op_raised_index(SH)*Umap

semilogy(max_coefficent_for_each_l(SH, ulm), "r-o")
semilogy(max_coefficent_for_each_l(SH, Ulm), "b-o")
semilogy(max_coefficent_for_each_l(SH, Ulm_raised_index), "g-o")
show()
