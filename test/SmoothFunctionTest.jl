#---------------------------------------------------------------
#e LaplaceOnASphere
# Soham 8/20
# Construct a smooth z-coordinate using
# spherical harmonics
#---------------------------------------------------------------

using LinearAlgebra, PyPlot, ProgressMeter

function Z(μ::T, ν::T)::T where {T}
    z = sqrt(4π/3)*ScalarSPH(1,0,μ,ν) + (1/10)*(sqrt(4π/7)*ScalarSPH(3,0,μ,ν) - sqrt(4π/11)*ScalarSPH(5,0,μ,ν)) 
    return z
end

function theta(μ::T, ν::T)::T where {T}
    x = sin(μ)*cos(ν) 
    y = sin(μ)*sin(ν) 
    z = Z(μ,ν)
    return acos(z/sqrt(x^2 + y^2 + z^2))
end

function analyticF(l::Int, m::Int, μ::T, ν::T)::Complex{T} where {T} 
    return ScalarSPH(l, m, theta(μ, ν), ν)
end

function g(a::Int, b::Int, μ::T, ν::T)::T where {T}
    if a == b == 1
        return 1
    elseif a == b == 2
        θ = theta(μ, ν) 
        return sin(θ)^2 
    else 
        return 0
    end
end
    
function hinv(a::Int, b::Int, μ::T, ν::T)::T where {T}
    d1d1 = ((5120*(649 + 45*cos(2μ) - 117*cos(4μ) + 63*cos(6μ)))/(3329198 + 157554*cos(2μ) - 46728*cos(4μ) - 161523*cos(6μ) - 5670*cos(8μ) + 3969*cos(10μ))) 
    d1d2 = 0 
    d2d1 = 1 
    d2d2 = 1

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

function q_hinv(a::Int, b::Int, μ::T, ν::T)::T where {T}
    return q(a,1,μ,ν)*hinv(1,b,μ,ν) + q(a,2,μ,ν)*hinv(2,b,μ,ν)
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

function scale_inv_detq(a::Int, b::Int, μ::T, ν::T)::T where {T} 
    if a == b 
        return 1/sqrt(detq(μ,ν))
    else
        return 0
    end
end

function scale_deth(a::Int, b::Int, μ::T, ν::T)::T where {T} 
    if a == b 
        return sqrt(deth(μ,ν))
    else
        return 0
    end
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


function plotvectorfield(SH::SphericalHarmonics{T}, u::Array{Complex{T},1}) where {T}
    u = reshape(u, (:,2))

    subplot(2,2,1)
    contourf(real.(reshape(u[:,1], (SH.N, 2*SH.N))))
    title("real(u_θ)")
    colorbar()

    subplot(2,2,2)
    contourf(real.(reshape(u[:,2], (SH.N, 2*SH.N))))
    title("real(u_ϕ)")
    colorbar()

    subplot(2,2,3)
    contourf(imag.(reshape(u[:,1], (SH.N, 2*SH.N))))
    title("imag(u_θ)")
    colorbar()

    subplot(2,2,4)
    contourf(imag.(reshape(u[:,2], (SH.N, 2*SH.N))))
    title("imag(u_ϕ)")
    colorbar()

    show()
end

#====================================================================#
# Experiments
#====================================================================#

SH = SphericalHarmonics{Float64}(15, 50)

# Test scaling of the gradient and it's projection onto vector spherical
# harmonics
sqrt_detq_by_deth(μ, ν) = sqrt(detq(μ,ν)/deth(μ,ν)) 
sqrt_deth_by_detq_q_hinv(a, b, μ, ν) = (1/sqrt_detq_by_deth(μ, ν))*q_hinv(a,b,μ,ν) 
@test deth(π/4, π/6) ≈ 0.6583073877333249

@time V   = modal_to_nodal_vector_op(SH)
@time V̄   = nodal_to_modal_vector_op(SH) 
@time C   = modal_to_nodal_curl_op(SH) 
@time C̄   = nodal_to_modal_curl_op(SH) 
@time V̄q  = nodal_to_modal_vector_op_q(SH) 
@time S̄   = nodal_to_modal_scalar_op(SH)
@time H   = scale_vector(SH, sqrt_deth_by_detq_q_hinv)
@time nf  = map(SH, (μ,ν)->analyticF(1,1,μ,ν)) 
@time ∇nf = V*(S̄*nf) 

# Plot the fall-off of the coefficents and the vector components
if false
    semilogy(max_coefficent_for_each_l(SH, V̄*(H*∇nf)),  "m-o")
    semilogy(max_coefficent_for_each_l(SH, V̄q*(H*∇nf)), "r-o")
    semilogy(max_coefficent_for_each_l(SH, C̄*(H*∇nf)),  "b-o")
    show()
end

if false
    RI = scale_vector(SH, q_hinv)
    DH = scale_vector(SH, scale_deth)
    DQ = scale_vector(SH, scale_inv_detq)
    @test H ≈ DQ*DH*RI
    
    u = reshape(∇nf, (:,2))
    v = reshape(RI*∇nf, (:,2))
    w = reshape(DH*RI*∇nf, (:,2))
    x = reshape(DQ*DH*RI*∇nf, (:,2))
    
    fig = figure(figsize=(25, 5))
    
    subplot(1,4,1)
        contourf(imag.(reshape(u[:,2], (SH.N, 2*SH.N))))
        title("imag((grad f)_ϕ)")
        colorbar()
    
    subplot(1,4,2)
        contourf(imag.(reshape(v[:,2], (SH.N, 2*SH.N))))
        title("imag(raise_index * (grad f)_ϕ)")
        colorbar()
    
    subplot(1,4,3)
        contourf(imag.(reshape(w[:,2], (SH.N, 2*SH.N))))
        title("imag(sqrt(deth) * raise_index * (grad f)_ϕ)")
        colorbar()
    
    subplot(1,4,4)
        contourf(imag.(reshape(x[:,2], (SH.N, 2*SH.N))))
        title("imag((1/sqrt(detq)) * sqrt(deth) * raise_index * (grad f)_ϕ)")
        colorbar()
    show()
end

# Plot the difference due to truncated expansion
regradH∇nf = V*V̄*(H*∇nf)
recurlH∇nf = C*C̄*(H*∇nf)
H∇nf = H*∇nf
plotvectorfield(SH, regradH∇nf + recurlH∇nf - H∇nf)

# Mystery: Why is the expansion of our "smooth" function not falling off-exponentially? 
#   
