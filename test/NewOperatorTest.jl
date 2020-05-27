#---------------------------------------------------------------
# LaplaceOnASphere
# Soham 5/20
# Construct new sparse FD operators that respect
# the symmetries around the poles
#---------------------------------------------------------------

using SparseArrays, LinearAlgebra

function map(u::Function, ni::Int, nj::Int)
    scalar = zeros(Complex{Float64}, ni, nj)
    for index in CartesianIndices(scalar)
        scalar[index] = u(collocation(index.I..., ni, nj)...) 
    end
    return vec(scalar)
end

function Base.join(i::Int, j::Int, ni::Int, nj::Int)::Int
    return LinearIndices((ni, nj))[i,j]
end

function Base.split(ij::Int, ni::Int, nj::Int)
    return CartesianIndices((ni, nj))[ij].I
end

function collocation(i::Int, j::Int, ni::Int, nj::Int)
    @assert 1 <= i <= ni
    @assert 1 <= j <= nj
    theta = (i-1/2)*(π/ni)
    phi   = (j-1)*(2π/nj)
    return (theta, phi)
end

function wrap(i::Int, j::Int, ni::Int, nj::Int)
    @assert iseven(nj)
    if i == 0 # north pole
        i = 1 - i 
        j = mod1(j + nj÷2, nj)
    elseif i == ni+1 # south pole
        i = 2ni + 1 - i 
        j = mod1(j + nj÷2, nj)
    elseif j == 0 ||  j == nj + 1 # crossing 0 or 2π
        j = mod1(j, nj)
    end
    return (i,j)
end

function Dθ(ni::Int, nj::Int)
    A = spzeros(ni*nj, ni*nj)
    for i in 1:ni, j in  1:nj
        A[join(i, j, ni, nj), join(wrap(i-1, j, ni, nj)..., ni, nj)] = -1/2
        A[join(i, j, ni, nj), join(wrap(i+1, j, ni, nj)..., ni, nj)] = +1/2
    end
    return A/(π/ni)
end

function Dϕ(ni::Int, nj::Int)
    A = spzeros(ni*nj, ni*nj)
    for j in 1:nj, i in  1:ni
        A[join(i, j, ni, nj), join(wrap(i, j-1, ni, nj)..., ni, nj)] = -1/2
        A[join(i, j, ni, nj), join(wrap(i, j+1, ni, nj)..., ni, nj)] = +1/2
    end
    return A/(2π/nj)
end

function Dθvec(ni::Int, nj::Int)
    A = spzeros(ni*nj, ni*nj)
    for i in 1:ni, j in  1:nj
        if i == 1 # northpole 
            A[join(i, j, ni, nj), join(wrap(i-1, j, ni, nj)..., ni, nj)] = +1/2
            A[join(i, j, ni, nj), join(wrap(i+1, j, ni, nj)..., ni, nj)] = +1/2
        elseif i == ni # southpole
            A[join(i, j, ni, nj), join(wrap(i-1, j, ni, nj)..., ni, nj)] = -1/2
            A[join(i, j, ni, nj), join(wrap(i+1, j, ni, nj)..., ni, nj)] = -1/2
        else    # somewhere in the middle
            A[join(i, j, ni, nj), join(wrap(i-1, j, ni, nj)..., ni, nj)] = -1/2
            A[join(i, j, ni, nj), join(wrap(i+1, j, ni, nj)..., ni, nj)] = +1/2
        end
    end
    return A/(π/ni)
end

function Dϕvec(ni::Int, nj::Int)
    A = spzeros(ni*nj, ni*nj)
    for j in 1:nj, i in  1:ni
        A[join(i, j, ni, nj), join(wrap(i, j-1, ni, nj)..., ni, nj)] = -1/2
        A[join(i, j, ni, nj), join(wrap(i, j+1, ni, nj)..., ni, nj)] = +1/2
    end
    return A/(2π/nj)
end

function S0(ni::Int, nj::Int, u::Function)
    A = spzeros(ni*nj, ni*nj)
    for j in 1:nj, i in  1:ni
        A[join(i, j, ni, nj), join(i, j, ni, nj)] = u(collocation(i, j, ni, nj)...)
    end
    return A
end

#---------------------------------------------------------------
# Test for high l,m [Done]
# Test convergence for derivative operators [Done]
# Test scaling [Done]
# Test grad and div
# Test Laplacee
# Test Eigenvals
#---------------------------------------------------------------

N1, N2      = (50, 100)
D1          = Dθ(N1, N2)
D2          = Dϕ(N1, N2)
D1vec       = Dθvec(N1, N2)
D2vec       = Dϕvec(N1, N2)
# ginv11      = S0(N1, N2, (θ,ϕ)->1) 
# ginv12      = S0(N1, N2, (θ,ϕ)->0) 
# ginv21      = S0(N1, N2, (θ,ϕ)->0) 
# ginv22      = S0(N1, N2, (θ,ϕ)->1/sin(θ)^2) 
# sqrtdetg    = S0(N1, N2, (θ,ϕ)->sin(θ))
# invsqrtdetg = S0(N1, N2, (θ,ϕ)->1/sin(θ))

# grad    = [ginv11*D1 + ginv12*D2, ginv21*D1 + ginv22*D2] 
# div     = [invsqrtdetg*(D1*sqrtdetg), invsqrtdetg*(D2*sqrtdetg)] 
# laplace = div[1]*grad[1] + div[2]*grad[2]

# grad    = [D1, (1/sinθ^2)*D2] 
# div     = [1/sinθ*D1*sinθ, (1/sinθ)*D2*sinθ]

S2 = S0(N1, N2, (x,y)->1/sin(x)^2)
S1 = S0(N1, N2, (x,y)->cos(x)/sin(x))
laplace = S1*D1 + D1vec*D1 + S2*D2vec*D2  

l, m = (3,2)
Ylm =  map((θ,ϕ)->ScalarSPH(l,m,θ,ϕ), N1, N2)
@show l, m, LInf(laplace*Ylm + l*(l+1)*Ylm)

N1, N2      = (100, 200)
D1          = Dθ(N1, N2)
D2          = Dϕ(N1, N2)
D1vec       = Dθvec(N1, N2)
D2vec       = Dϕvec(N1, N2)
# ginv11      = S0(N1, N2, (θ,ϕ)->1) 
# ginv12      = S0(N1, N2, (θ,ϕ)->0) 
# ginv21      = S0(N1, N2, (θ,ϕ)->0) 
# ginv22      = S0(N1, N2, (θ,ϕ)->1/sin(θ)^2) 
# sqrtdetg    = S0(N1, N2, (θ,ϕ)->sin(θ))
# invsqrtdetg = S0(N1, N2, (θ,ϕ)->1/sin(θ))

# grad    = [ginv11*D1 + ginv12*D2, ginv21*D1 + ginv22*D2] 
# div     = [invsqrtdetg*(D1*sqrtdetg), invsqrtdetg*(D2*sqrtdetg)] 
# laplace = div[1]*grad[1] + div[2]*grad[2]

# grad    = [D1, (1/sinθ^2)*D2] 
# div     = [1/sinθ*D1*sinθ, (1/sinθ)*D2*sinθ]

S2 = S0(N1, N2, (x,y)->1/sin(x)^2)
S1 = S0(N1, N2, (x,y)->cos(x)/sin(x))
laplace = S1*D1 + D1vec*D1 + S2*D2vec*D2  

l, m = (3,2)
Ylm =  map((θ,ϕ)->ScalarSPH(l,m,θ,ϕ), N1, N2)
@show l, m, LInf(laplace*Ylm + l*(l+1)*Ylm)

# using PyPlot
# imshow(reshape(abs.(laplace*Ylm + l*(l+1)*Ylm), N1, N2))
# savefig("error.pdf")
# close()

# N1, N2      = (200, 400)
# D1          = Dθ(N1, N2)
# D2          = Dϕ(N1, N2)
# D1vec       = Dθvec(N1, N2)
# D2vec       = Dϕvec(N1, N2)
# ginv11      = S0(N1, N2, (θ,ϕ)->1) 
# ginv12      = S0(N1, N2, (θ,ϕ)->0) 
# ginv21      = S0(N1, N2, (θ,ϕ)->0) 
# ginv22      = S0(N1, N2, (θ,ϕ)->1/sin(θ)^2) 
# sqrtdetg    = S0(N1, N2, (θ,ϕ)->sin(θ))
# invsqrtdetg = S0(N1, N2, (θ,ϕ)->1/sin(θ))

# grad    = [ginv11*D1 + ginv12*D2, ginv21*D1 + ginv22*D2] 
# div     = [invsqrtdetg*(D1*sqrtdetg), invsqrtdetg*(D2*sqrtdetg)] 
# laplace = div[1]*grad[1] + div[2]*grad[2]

# l, m = (2,1)
# Ylm =  map((θ,ϕ)->ScalarSPH(l,m,θ,ϕ), N1, N2)
# @show l, m, LInf(laplace*Ylm + l*(l+1)*Ylm)

# con

#---------------------------------------------------------------
# Test laplace 
#---------------------------------------------------------------
# for l in 0:4
    # for m in 0:l
        # Ylm =  map((θ,ϕ)->ScalarSPH(l,m,θ,ϕ), N1, N2)
        # @show l, m, LInf(laplace*Ylm + l*(l+1)*Ylm)
    # end
    # println()
# end

# @show cond(Array(laplace))

# F = eigen(Array(laplace))
# plot(F.values[1:15])
# savefig("eigenvalues.pdf")
# close()

