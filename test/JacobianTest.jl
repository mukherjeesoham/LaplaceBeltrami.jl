#---------------------------------------------------------------
# LaplaceOnASphere
# Soham 3/20
# Test metric functions
#---------------------------------------------------------------

S = SphericalHarmonics(4)

function μ(θ::T, ϕ::T) where {T}
    return ScalarSPH(2, 0, θ, ϕ)
end

function ν(θ::T, ϕ::T) where {T}
    return ScalarSPH(1, 1, θ, ϕ)
end

function computeJacobian(S::SphericalHarmonics{T}, θ::T, ϕ::T)::Array{Complex{T},2} where {T}
    X = map(S, (θ,ϕ)->μ(θ,ϕ)) 
    Y = map(S, (θ,ϕ)->ν(θ,ϕ)) 
    L = nodal_to_modal_scalar_op(S) 
    (Xlm, Ylm) = (L*X, L*Y)
    jacobian = zeros(Complex{T}, (2,2))
    for l in 0:S.lmax, m in -l:l
        Ψ = VectorSPH(l, m, θ, ϕ)
        jacobian[1,1] += Xlm[join(l,m)]*Ψ[1] 
        jacobian[1,2] += Xlm[join(l,m)]*Ψ[2]*sin(θ)
        jacobian[2,1] += Ylm[join(l,m)]*Ψ[1]
        jacobian[2,2] += Ylm[join(l,m)]*Ψ[2]*sin(θ)
    end
    return jacobian
end

function computeGradient(S::SphericalHarmonics{T})::Array{Complex{T},1} where {T}
    X = map(S, (θ,ϕ)->ν(θ,ϕ)) 
    L = nodal_to_modal_scalar_op(S) 
    P = modal_to_nodal_vector_op(S)
    ∇X = P*(L*X)
    return ∇X
end

∇μ = map(S, (θ, ϕ)->(-3/2)*sqrt(5/π)*cos(θ)*sin(θ), (θ, ϕ)->0)
∇ν = map(S, (θ, ϕ)->-(1/2)*cis(ϕ)*sqrt(3/(2π))*cos(θ), 
            (θ, ϕ)->-(1/2)*im*cis(ϕ)*sqrt(3/(2π)))
@test ∇ν ≈ computeGradient(S)

function testJacobian(θ::T, ϕ::T)::Array{Complex{T},2} where {T}
    J11 = (-3/2)*sqrt(5/π)*cos(θ)*sin(θ)
    J12 = T(0)
    J21 = -(1/2)*cis(ϕ)*sqrt(3/(2π))*cos(θ)
    J22 = -(1/2)*im*cis(ϕ)*sqrt(3/(2π))*sin(θ)
    return [J11 J12; 
            J21 J22]
end

@test computeJacobian(S, π/3, π/6) ≈ testJacobian(π/3, π/6)
