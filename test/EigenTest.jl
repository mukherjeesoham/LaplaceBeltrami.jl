#---------------------------------------------------------------
# LaplaceOnASphere
# Soham 6/20
#
# Construct the Laplace operator in local coordinates using 
# Spherical harmonics
# We have 1/|g| d/dx^i (|g| g^ij dϕ/dx^j) 
# where x^i = {μ, ν} and g = ({1, 0}, {0, sin(θ)^2}}
# We work with a general metric h and we shoe-horn 
# the metric g in there. 
# We have 1/|h| d/dx^i (|h| h^ij dϕ/dx^j) 
# We have |g|/|h| 1/|g| d/dx^i (|g| |h|/|g| δ^i_m h^mj dϕ/dx^j) 
# or  |g|/|h| 1/|g| d/dx^i (|g| |h|/|g| g^in g_nm h^mj dϕ/dx^j) 
# or  |g|/|h| 1/|g| d/dx^i (|g| g^in |h|/|g|  g_nm h^mj dϕ/dx^j) 
# or  |g|/|h| 1/|g| d/dx^i (|g| g^in β_n) 
# or  |g|/|h| Ylm (-l*(l+1)) β^lm_n
#---------------------------------------------------------------

using LinearAlgebra

#---------------------------------------------------------------
# Functions for constructing the scaling operators
#---------------------------------------------------------------

function hinv(a::Int, b::Int, μ::Float64, ν::Float64)::Float64
    if a == b == 1
        return 1
    elseif a == b == 2
        return 1/sin(μ)^2
    else
        return 0
    end
end

function sqrt_detg_by_deth(μ::Float64, ν::Float64)::Float64
    dethinv = -hinv(1,2,μ,ν)*hinv(2,1,μ,ν) + hinv(1,1,μ,ν)*hinv(2,2,μ,ν)
    return sin(μ)*sqrt(dethinv)
end

function sqrt_deth_by_detg_g_hinv(a::Int, b::Int, μ::Float64, ν::Float64)::Float64
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

#---------------------------------------------------------------
# Construct the operators
#---------------------------------------------------------------

SH = SphericalHarmonics(12)
lmax = 3

S = modal_to_nodal_scalar_op(SH) 
S̄ = nodal_to_modal_scalar_op(SH)
V = modal_to_nodal_vector_op(SH)
V̄ = nodal_to_modal_vector_op(SH) 

D = scale_scalar(SH, sqrt_detg_by_deth)
H = scale_vector(SH, sqrt_deth_by_detg_g_hinv)
L = scale_lmodes(SH, divergence) 

grad    = V*S̄
div     = S*L*V̄
laplace = D*div*(H*grad)

#---------------------------------------------------------------
# test laplace, and if that doesn't work 
# check grad, div and scaling
#---------------------------------------------------------------
@testset "all" begin

    @test H*grad ≈ grad
    @test laplace ≈ div*grad


    # @testset "projection" begin
        # for l in 0:lmax
            # for m in -l:l
                # ylm = map(sh, (μ,ν)->scalarsph(l,m,μ,ν))
                # ψlm = map(sh, (μ,ν)->gradsh(1,l,m,μ,ν), (μ,ν)->gradsh(2,l,m,μ,ν))
                # e1  = l1(s*(s̄*ylm) - ylm)
                # e2  = l1(v*(v̄*ψlm) - ψlm)
                # @test e1 < 1e-10
                # @test e2 < 1e-10
            # end
        # end
    # end;
    
    @testset "grad" begin
        for l in 0:lmax
            for m in -l:l
                Ylm = map(SH, (μ,ν)->ScalarSPH(l,m,μ,ν))
                Ψlm = map(SH, (μ,ν)->GradSH(1,l,m,μ,ν), (μ,ν)->GradSH(2,l,m,μ,ν))
                @test isapprox(grad*Ylm, Ψlm; atol=1e-10)
            end
        end
    end;

    @testset "divergence" begin
        for l in 0:lmax
            for m in -l:l
                Ylm = map(SH, (μ,ν)->ScalarSPH(l,m,μ,ν))
                Ψlm = map(SH, (μ,ν)->GradSH(1,l,m,μ,ν), (μ,ν)->GradSH(2,l,m,μ,ν))
                E1  = L1(div*Ψlm + l*(l+1)*Ylm)
                @test isapprox(div*Ψlm, -l*(l+1)*Ylm, atol=1e-10)
            end
        end
    end;

    @testset "laplace" begin
        for l in 0:lmax
            for m in -l:l
                Ylm = map(SH, (μ,ν)->ScalarSPH(l,m,μ,ν))
                Ψlm = map(SH, (μ,ν)->GradSH(1,l,m,μ,ν), (μ,ν)->GradSH(2,l,m,μ,ν))
                @test isapprox(div*(grad*Ylm), -l*(l+l)*Ylm, atol=1e-10)
            end
        end
    end;
    
end;

