#---------------------------------------------------------------
# LaplaceOnASphere
# Soham 7/20
# Debugging the code and the tests. Broadly, we can 
# introduce a bug if one of the three places  
#   A Construction of the Laplace operator
#   B Construction of coordinates from the eigenvalues
#   C Comparison with the good coordinates
# Tests we can do
#   A Test the operator on eigenfunctions and their linear
#     combinations
#   B Test without any coordinate transformation and slowly
#     tune up the coordinate transformation
#   C Use tests from B.  
#---------------------------------------------------------------

using LinearAlgebra

p = 0.01
θ(μ, ν) = μ
ϕ(μ, ν) = ν + p*sin(μ)

function hinv(a::Int, b::Int, μ::T, ν::T)::Complex{T} where {T}
    # TODO: Test
    if a == b == 1
        return 1
    elseif a == b == 2
        return (p^2)*cos(μ)^2 + csc(μ)^2
    else
        return -p*cos(μ)
    end
end

function sqrt_detg_by_deth(μ::T, ν::T)::Complex{T} where {T}
    # TODO: Test
    dethinv = -hinv(1,2,μ,ν)*hinv(2,1,μ,ν) + hinv(1,1,μ,ν)*hinv(2,2,μ,ν)
    return sin(μ)*sqrt(dethinv)
end

function sqrt_deth_by_detg(μ::T, ν::T)::Complex{T} where {T}
    return 1/sqrt_detg_by_deth(μ, ν)
end

function sqrt_deth_by_detg_g_hinv(a::Int, b::Int, μ::T, ν::T)::Complex{T} where {T}
   # TODO: Test
   dethinv = -hinv(1,2,μ,ν)*hinv(2,1,μ,ν) + hinv(1,1,μ,ν)*hinv(2,2,μ,ν)
   deth    = 1/dethinv
   if a == 1
       return sqrt(deth)*csc(μ)*hinv(a,b,μ,ν)
   else
       return sqrt(deth)*sin(μ)*hinv(a,b,μ,ν) 
   end
end

function divergence(l::Int, m::Int)::Int
    return -l*(l+1)
end

function laplace(SH::SphericalHarmonics)
    S = modal_to_nodal_scalar_op(SH) 
    S̄ = nodal_to_modal_scalar_op(SH)
    V = modal_to_nodal_vector_op(SH)
    V̄ = nodal_to_modal_vector_op(SH) 
    
    D = scale_scalar(SH, sqrt_detg_by_deth)
    H = scale_vector(SH, sqrt_deth_by_detg_g_hinv)
    L = scale_lmodes(SH, divergence) 

    grad = V
    div  = S*L*V̄
    Δ    = S̄*D*div*(H*grad)

    return Δ    
end

SH = SphericalHarmonics(12)
Δ  = laplace(SH)
S  = modal_to_nodal_scalar_op(SH) 
S̄  = nodal_to_modal_scalar_op(SH)
lmax = 1

for l in 0:lmax
    for m in -l:l
        @show l, m
        Ylm = map(SH, (μ, ν)->ScalarSPH(l, m, θ(μ,ν), ϕ(μ, ν)))
        @show norm(S*Δ*S̄*Ylm  + l*(l+1)*Ylm)
    end
end

# Test grad for m != 0 without coordinate transformation
Y11  = map(SH, (μ, ν)->ScalarSPH(1,1, μ, ν)) 
∇Y11 = map(SH, (μ,ν)->GradSH(1,1,1,μ,ν), (μ,ν)->GradSH(2,1,1,μ,ν))
S̄    = nodal_to_modal_scalar_op(SH)
V    = modal_to_nodal_vector_op(SH)
grad = V*S̄
@show norm(grad*Y11 - ∇Y11)

# Test grad for m != 0 with coordinate transformation
Y11_ct  = map(SH, (μ,ν)->ScalarSPH(1,1, θ(μ,ν), ϕ(μ,ν))) 
∇Y11_ct = map(SH, (μ,ν)-> (1/2)*cis(ν + p*sin(μ))*sqrt(3/2π)*cos(μ)*(-1 - im*p*sin(μ)), 
                  (μ,ν)->-(1/2)*im*cis(ν + p*sin(μ))*sqrt(3/2π)*sin(μ))
# D_μ, D_ν

# This is not as small as it should be! Check the coefficents first?
error = grad*Y11_ct - ∇Y11_ct
# @show dot(SH, error, error)
# @show norm(grad*Y11_ct - ∇Y11_ct)

μ = map(SH, (μ, ν)->μ) 
ν = map(SH, (μ, ν)->ν) 
F = V*S̄*Y11_ct
δ = (grad*Y11_ct) - ∇Y11_ct

# Plot the error
l = size(δ)[1]
δ1 = δ[1:l ÷ 2]
δ2 = δ[l ÷ 2 + 1: end]

@show typeof(δ1)
@show typeof(δ2)

using PyPlot
fig = figure(figsize=(12, 5))
subplot(1,2,1)
contourf(reshape(δ1, (SH.N, 2*SH.N)))
colorbar()
subplot(1,2,2)
contourf(reshape(δ2, (SH.N, 2*SH.N)))
colorbar()
savefig("error-vec.pdf")
close()


imax, jmax, a = split3(argmax(abs.(δ)), SH.N)
@show imax, jmax, a

for index in CartesianIndices(μ)
    i, j = split(index.I[1], SH.N)
    if all((i, j) .== (imax, jmax))
        @show μ[index], ν[index]
        @show F[join3(i, j, 1, SH.N)] 
        @show F[join3(i, j, 2, SH.N)] 
        @show δ[join3(i, j, 1, SH.N)]
        @show δ[join3(i, j, 2, SH.N)]
    end
end

# We incur aleast 10% error after the projection to grad. However, 
# this still doesn't explain why the difference is so large compared 
# to the analytical expressions. Are the analytic expressions wrong?
# Where is the largest error? 


