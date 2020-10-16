#---------------------------------------------------------------
#e LaplaceOnASphere
# Soham 8/20
# Construct a smooth z-coordinate using
# spherical harmonics
#---------------------------------------------------------------

using PyPlot

SH = SphericalHarmonics(40)
@test deth(π/4, π/6) ≈ 0.6583073877333249

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
nf  = map(SH, (μ,ν)->analyticF(1,1,μ,ν)) 
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


# # Project the ratio of the determinants in Scalar spherical harmonics
# # and the q_hniv in terms of tensor spherical harmonics. Concerning the 
# # non-zero coefficents we'll get for projecting on Ylm, these should
# # vanish due to the dot product.
# sqrtdethbydetq = map(SH, (μ,ν)->sqrt(deth(μ,ν)/detq(μ,ν)))

# if false
    # semilogy(filter_low_modes(V̄*∇nf), "g-s", linewidth=0.8, label="{}")
    # semilogy(filter_low_modes(S̄*sqrtdethbydetq), "b-s", linewidth=0.4, label="(deth/detq)")
    # legend(frameon=false)
    # show()
# end

# # Use the raise index operator on a smooth vector field and see how
# # things change.
# H3 = scale_vector(SH, q)
# H4 = scale_vector(SH, hinv)
# @test H3*H4 ≈ H1 

# usmooth = map(SH, (μ,ν)->GradSH(1,1,0,μ,ν), (μ,ν)->GradSH(2,1,0,μ,ν))

# if true
    # semilogy(max_coefficent_for_each_l(SH, V̄*usmooth), "b-o", linewidth=0.4, label="{}")
    # semilogy(max_coefficent_for_each_l(SH, V̄*(H3*usmooth)), "g-^", linewidth=0.4, label="H3")
    # semilogy(max_coefficent_for_each_l(SH, V̄*(H4*usmooth)), "r-^", linewidth=0.4, label="H4")
    # legend(frameon=false)
    # show()
# end
