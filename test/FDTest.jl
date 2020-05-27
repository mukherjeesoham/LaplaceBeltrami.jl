#---------------------------------------------------------------
# LaplaceOnASphere
# Soham 5/20
# Construct higher order FD operators
# Convergence is too slow for 2nd order FD
#---------------------------------------------------------------

using LinearAlgebra, PyPlot, Preconditioners

N1, N2 = (400, 10)

# 1D operators
# θ  = map1D(Float64, :θ, N1, x->x) 
# ϕ  = map1D(Float64, :ϕ, N2, x->x) 
# D1 = Dθ(Float64, N1)
# f  = sin.(θ)
# g  = cos.(θ)
# h  = (1/2)*sqrt(3/π)*g
# dh = -(1/2)*sqrt(3/π)*f
# # u  = abs.(f).*dh
# # du = -sqrt(3/π)*g.*abs.(f) 
# u  = abs.(f).*dh
# du = -sqrt(3/π)*g.*abs.(f) 

# # plot(θ, du)
# # plot(θ, D1*u, "r-o")
# # show()
# # exit()

# # 2D operators
# θ  = map2D(Float64, N1, N2, (x, y)->x) 
# D1 = Dθ(Float64, N1, N2)
# S1 = Scale(Float64, N1, N2, (x, y)->abs(sin(x)))
# f  = sin.(θ)
# g  = cos.(θ)
# h  = (1/2)*sqrt(3/π)*g
# dh = -(1/2)*sqrt(3/π)*f

# u  = abs.(f).*dh
# du = -sqrt(3/π)*g.*abs.(f) 

# @show LInf(D1*h - dh) 
# @show LInf(S1*D1*h - abs.(f).*dh)
# @show LInf(D1*u - du)
# exit()

#---------------------------------------------------------------
# Construct gradient, divergence and the laplace operator 
#---------------------------------------------------------------
(T, N1, N2) = (Float64, N1, N2)
D1 = Dθ(T, N1, N2)
D2 = Dϕ(T, N1, N2)
ginv11 = Scale(Float64, N1, N2, (θ,ϕ)->1) 
ginv12 = Scale(Float64, N1, N2, (θ,ϕ)->0) 
ginv21 = Scale(Float64, N1, N2, (θ,ϕ)->0) 
ginv22 = Scale(Float64, N1, N2, (θ,ϕ)->1/sin(θ)^2) 
sqrtdetg = Scale(Float64, N1, N2, (θ,ϕ)->abs(sin(θ)))
invsqrtdetg = Scale(Float64, N1, N2, (θ,ϕ)->1/abs(sin(θ)))

grad    = [ginv11*D1 + ginv12*D2, ginv21*D1 + ginv22*D2] 
div     = [invsqrtdetg*(D1*sqrtdetg), invsqrtdetg*(D2*sqrtdetg)] 
laplace = div[1]*grad[1] + div[2]*grad[2]

#---------------------------------------------------------------
# Test convergence 
#---------------------------------------------------------------
for l in 0:4, m in 0:l
    if l == 1 && m == 0
        Ylm =  map2D(Float64, N1, N2, (θ,ϕ)->ScalarSPH(l,m,θ,ϕ))
        @show l, m, LInf(laplace*Ylm + l*(l+1)*Ylm)
    end
end

#---------------------------------------------------------------
# Check and compute eigenvalues 
#---------------------------------------------------------------
# @show cond(grad[1])
# @show cond(grad[2])
# @show cond(div[1])
# @show cond(div[2])
@show cond(laplace)
# @show maximum(abs.(laplace))
# @show minimum(abs.(laplace))

F = eigen(laplace)
plot(F.values[1:15])
show()



