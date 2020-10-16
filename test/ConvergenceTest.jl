#---------------------------------------------------------------
#e LaplaceOnASphere
# Soham 8/20
# Construct a smooth z-coordinate using
# spherical harmonics
#---------------------------------------------------------------

using PyPlot

SH = SphericalHarmonics(30)
# @test deth(π/4, π/6) ≈ 0.6583073877333249

function only_sqrt_deth_by_detq(a::Int, b::Int, μ::T, ν::T)::T where {T}
    if a == b
        return sqrt(deth(μ,ν)/detq(μ,ν))
    else
        return 0
    end
end


# Compute the operators
@time S, S̄ = scalar_op(SH)
@time V, V̄ = vector_op(SH)
@time H = scale_vector(SH, sqrt_deth_by_detq_q_hinv)
@time H1 = scale_vector(SH, q_hinv)
@time H2 = scale_vector(SH, only_sqrt_deth_by_detq)

# Compute the fields
nf  = map(SH, (μ,ν)->analyticF(2,1,μ,ν)) 
∇nf = V*(S̄*nf) 

# Plot the fall-off of the coefficents and the vector components
if false
    semilogy(max_coefficent_for_each_l(SH, S̄*nf),  "m v")
    semilogy(max_coefficent_for_each_l(SH, V̄*∇nf), "g ^")
    semilogy(max_coefficent_for_each_l(SH, V̄*(H*∇nf)), "r-o")
    show()
end

function filter_low_modes(ulm::Array{T,1})::Array{T,1} where {T}
    ul = max_coefficent_for_each_l(SH, ulm)
    ul[abs.(ul) .< 1e-10] .= NaN
    return ul
end

# See which operation messes with the smoothness
if true
    semilogy(filter_low_modes(V̄*(H2*(H1*∇nf))), "r-^", linewidth=0.4, label="H1*H2")
    semilogy(filter_low_modes(V̄*(H1*∇nf)), "b-v", linewidth=0.4, label="H1")
    semilogy(filter_low_modes(V̄*(H2*∇nf)), "k-o", linewidth=0.4, label="H2")
    semilogy(filter_low_modes(V̄*∇nf), "g-s", linewidth=0.8, label="{}")
    legend(frameon=false)
    show()
end

exit()

# Compute the Laplace operator and look at the residual 
L = scale_lmodes(SH, (l,m)->-l*(l+1))
W = scale_scalar(SH, (μ,ν)->sqrt(detq(μ,ν)/deth(μ,ν)))  
grad = V*S̄ 
div  = S*L*V̄  
Δ = S̄*(W*(div*(H*grad)))
@time S*Δ*nf

using LinearAlgebra
@show norm(S*Δ*nf + 2*nf)

