#---------------------------------------------------------------
# LaplaceOnASphere
# Soham 7/20
# Construct analytic functions on the sphere and 
# construct a coordinate transformation out of it. 
#---------------------------------------------------------------

using LinearAlgebra, PyPlot

function analyticF(l::Int, m::Int, μ::T, ν::T)::Complex{T} where {T} 
    A1 = real(GradSH(1,1,1,μ,ν))  
    A2 = real(GradSH(2,1,1,μ,ν))  
    B1 = real(GradSH(1,3,1,μ,ν))  
    B2 = real(GradSH(2,3,1,μ,ν))  
    T1 = (√14*A1 - B1)/100 
    T2 = (√14*A2 - B2)/100 
    return ScalarSPH(l, m, μ + T1, ν + T2)
end

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

function qinv(a::Int, b::Int, μ::T, ν::T)::T where {T}
    if a == b == 1
        return 1
    elseif a == b == 2
        return 1/sin(μ)^2 
    else 
        return 0
    end
end

function detq(μ::T, ν::T)::T where {T}
    return sin(μ)^2
end


function g(a::Int, b::Int, μ::T, ν::T)::T where {T}
    if a == b == 1
        return 1
    elseif a == b == 2
        θ = μ - (3/160)*sqrt(21/π)*cos(μ)*cos(ν)*sin(μ)^2
        return sin(θ)^2 
    else 
        return 0
    end
end
    
function hinv(a::Int, b::Int, μ::T, ν::T)::T where {T}
    d1d1 = 1 + (3/640)*sqrt(21/π)*cos(ν)*(sin(μ) - 3*sin(3μ)) 
    d1d2 = (3/160)*sqrt(21/π)*cos(μ)*sin(ν)*sin(μ)^2 
    d2d1 = (3/160)*sqrt(21/π)*cos(μ)*sin(ν)*sin(μ)^2 
    d2d2 = 1 + (1/160)*sqrt(21/π)*cos(ν)*sin(μ)^3 

    g11  = g(1,1,μ,ν)
    g22  = g(2,2,μ,ν)
    g12  = g21 = g(1,2,μ,ν)
    
    h11  = d1d1*d1d1*g11 + d1d1*d2d1*g12 + d2d1*d1d1*g21 + d2d1*d2d1*g22 
    h22  = d1d2*d1d2*g11 + d1d2*d2d2*g12 + d2d2*d1d2*g21 + d2d2*d2d2*g22 
    h12  = d1d1*d1d2*g11 + d1d1*d2d2*g12 + d2d1*d1d2*g21 + d2d1*d2d2*g22 
    hinv = inv([h11 h12; h12 h22])

    if a == b == 1
        return hinv[1,1]
    elseif a == b == 2
        return hinv[2,2]
    else
        return hinv[1,2]
    end
end

function deth(μ::T, ν::T)::T where {T}
    hinvmat = [hinv(1,1,μ,ν) hinv(1,2,μ,ν); 
               hinv(2,1,μ,ν) hinv(2,2,μ,ν)]
    return 1/det(hinvmat)
end

function quad(SH::SphericalHarmonics{T}, s::Array{Complex{T},1})::T where {T}
    q = map(SH, (μ,ν)->sqrt(detq(μ,ν)))
    integrand = s
    integral  = (S̄*integrand)[1] 
    @assert abs(imag(integral)) < 1e-10
    return sqrt(4π)*real(integral)
end

function LinearAlgebra.norm(SH::SphericalHarmonics{T}, u::Array{Complex{T},1})::T where {T}
    U = scale_vector(SH, qinv)*u
    s = vec(sum(reshape(U.*u, (:, 2)), dims=2))
    return sqrt(abs(quad(SH, s)))
end

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

SH = SphericalHarmonics(42)

# Test the function and it's gradient (analytic expressions)
@test isapprox(real(analyticF(1,1,π/4,π/6)), -0.20806299407080006, atol=1e-15)
@test isapprox(imag(analyticF(1,1,π/4,π/6)), -0.12091898953255495, atol=1e-15)
@test isapprox(real(analytic∇F(1,1,π/4,π/6)[1]), -0.21011550466219212, atol=1e-15) 
@test isapprox(imag(analytic∇F(1,1,π/4,π/6)[1]), -0.12449707588262553, atol=1e-15) 
@test isapprox(real(analytic∇F(1,1,π/4,π/6)[2]), 0.11968050567258394, atol=1e-15) 
@test isapprox(imag(analytic∇F(1,1,π/4,π/6)[2]), -0.2101598759559276, atol=1e-15) 

# Test coefficents
S̄  = nodal_to_modal_scalar_op(SH)
nf = map(SH, (μ,ν)->analyticF(1,1,μ,ν)) 
mf = S̄*nf 

@test isapprox(abs(mf[join(2,-2)]), 0.0, atol=1e-7)
@test isapprox(abs(mf[join(2,-1)]), 0.0, atol=1e-15)
@test isapprox(abs(mf[join(2, 0)]), 0.002107167467887221, atol=1e-10)
@test isapprox(abs(mf[join(2, 1)]), 0.0, atol=1e-15)
@test isapprox(abs(mf[join(2, 2)]), 0.003096959366727792, atol=1e-10)

# Test gradient
V   = modal_to_nodal_vector_op(SH)
∇f  = map(SH, (μ,ν)->analytic∇F(1,1,μ,ν)[1], (μ,ν)->analytic∇F(1,1,μ,ν)[2])
∇nf = V*mf 

u = map(SH, (μ,ν)->ScalarSPH(0,0,μ,ν))
w = map(SH, (μ,ν)->1, (μ,ν)->sin(μ))
@test isapprox(quad(SH, u), 3.544907701871067, atol=1e-10) 
@test isapprox(norm(SH, w), 5.013256549304452, atol=1e-10) 

# Compute the norm of the error vector over the sphere and make
# sure it's small.
@test isapprox(norm(SH, ∇f), 0.011037702906455915, atol=1e-9) 
@test isapprox(norm(SH, ∇nf - ∇f), 0.0, atol=1e-9) 

# Compute metric and metric det for the transformed metric and 
# check the Laplace operator
@test isapprox(deth(π/4,π/6), 0.47546396953798614, atol=1e-15)

# Test scaling of the gradient and it's projection onto
# vector spherical harmonics
sqrt_detq_by_deth(μ, ν) = sqrt(detq(μ,ν)/deth(μ,ν)) 
sqrt_deth_by_detq_hinv(a, b, μ, ν) = (1/sqrt_detq_by_deth(μ, ν))*hinv(a,b,μ,ν) 
V̄ = nodal_to_modal_vector_op(SH) 
H = scale_vector(SH, sqrt_deth_by_detq_hinv)

# Let's test the H operator.
w = map(SH, (μ,ν)->sqrt(deth(μ,ν)/detq(μ,ν))*hinv(1,1,μ,ν)*analytic∇F(1,1,μ,ν)[1] + sqrt(deth(μ,ν)/detq(μ,ν))*hinv(1,2,μ,ν)*analytic∇F(1,1,μ,ν)[2],
            (μ,ν)->sqrt(deth(μ,ν)/detq(μ,ν))*hinv(2,1,μ,ν)*analytic∇F(1,1,μ,ν)[1] + sqrt(deth(μ,ν)/detq(μ,ν))*hinv(2,2,μ,ν)*analytic∇F(1,1,μ,ν)[2])
@test isapprox(norm(SH, w- H*∇nf), 0.0, atol=1e-8)

# This is difficult to test since Mathematica doesn't finish running
# Let's test the raise index operator and the determinant.
# FIXME: This integral is not converging, due to the fact that H*∇f doesn't have a converging expansion 
# @show norm(SH, H*∇nf)

# Plot the fall-off of the coefficents and the vector components
if true
    semilogy(max_coefficent_for_each_l(SH,V̄*(H*∇nf)), "m-o")
    show()

    w1 = map(SH, (μ,ν)->sqrt(deth(μ,ν)/detq(μ,ν))*hinv(1,1,μ,ν)*analytic∇F(1,1,μ,ν)[1] + sqrt(deth(μ,ν)/detq(μ,ν))*hinv(1,2,μ,ν)*analytic∇F(1,1,μ,ν)[2])
    w2 = map(SH, (μ,ν)->sqrt(deth(μ,ν)/detq(μ,ν))*hinv(2,1,μ,ν)*analytic∇F(1,1,μ,ν)[1] + sqrt(deth(μ,ν)/detq(μ,ν))*hinv(2,2,μ,ν)*analytic∇F(1,1,μ,ν)[2])
    subplot(2,2,1)
    contourf(real.(reshape(w1, (SH.N, 2*SH.N))))
    title("real(w1)")
    colorbar()
    subplot(2,2,2)
    contourf(imag.(reshape(w1, (SH.N, 2*SH.N))))
    title("imag(w1)")
    colorbar()
    subplot(2,2,3)
    contourf(real.(reshape(w2, (SH.N, 2*SH.N))))
    title("real(w2)")
    colorbar()
    subplot(2,2,4)
    contourf(imag.(reshape(w2, (SH.N, 2*SH.N))))
    title("imag(w2)")
    colorbar()
    show()
end

# Now test divergence of vector fields
divergence(l, m) = -l*(l+1)
S = modal_to_nodal_scalar_op(SH) 
L = scale_lmodes(SH, divergence) 
div = S*L*V̄

u  = map(SH, (μ,ν)->ScalarSPH(1,1,μ,ν))
∇u = map(SH, (μ,ν)->GradSH(1,1,1,μ,ν), (μ,ν)->GradSH(2,1,1,μ,ν)) 
@test isapprox(norm(div*∇u + 2 .*u), 0.0, atol=1e-11)
