#---------------------------------------------------------------
# LaplaceOnASphere
# Soham 5/20
# Construct new sparse FD operators that respect
# the symmetries around the poles for scalars and vectors
#---------------------------------------------------------------

using SparseArrays, LinearAlgebra, PyPlot

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
    if i < 1 # north pole
        i = 1 - i 
        j = mod1(j + nj÷2, nj)
    elseif i > ni # south pole
        i = 2ni + 1 - i 
        j = mod1(j + nj÷2, nj)
    elseif j < 1 ||  j > nj  # crossing 0 or 2π
        j = mod1(j, nj)
    end
    return (i,j)
end

function flipsign(i::Int, ni::Int)
    if i < 1 || i > ni # across the poles
        return -1
    else
        return +1
    end
end

function stencil(order::Int, index::Int)
    if order == 2
        return (-1/2, 0, 1/2)[index+2]
    elseif order == 4
        return (1/12, -2/3, 0, 2/3, -1/12)[index+3]
    elseif order == 6
        return (-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60)[index+4]
    elseif order == 8
        return (1/280, -4/105, 1/5, -4/5, 0, 4/5, -1/5, 4/105, -1/280)[index+5]	
    else
        errorexit("Invalid order")
    end
end

function Dθ(ni::Int, nj::Int, order::Int)
    I, J, V = (Int[], Int[], Float64[])
    for i in 1:ni, j in  1:nj
        for Δi in -order÷2:order÷2
            push!(I, join(i, j, ni, nj)) 
            push!(J, join(wrap(i+Δi, j, ni, nj)..., ni, nj)) 
            push!(V, stencil(order, Δi))
        end
    end 
    return dropzeros(sparse(I, J, V/(π/ni)))
end

function Dϕ(ni::Int, nj::Int, order::Int)
    I, J, V = (Int[], Int[], Float64[])
    for i in 1:ni, j in  1:nj
        for Δj in -order÷2:order÷2
            push!(I, join(i, j, ni, nj)) 
            push!(J, join(wrap(i, j+Δj, ni, nj)..., ni, nj)) 
            push!(V, stencil(order, Δj))
        end
    end 
    return dropzeros(sparse(I, J, V/(2π/nj)))
end

function Dθ̄(ni::Int, nj::Int, order::Int)
    I, J, V = (Int[], Int[], Float64[])
    for i in 1:ni, j in  1:nj
        for Δi in -order÷2:order÷2
            push!(I, join(i, j, ni, nj)) 
            push!(J, join(wrap(i+Δi, j, ni, nj)..., ni, nj)) 
            push!(V, flipsign(i+Δi, ni)*stencil(order, Δi))
        end
    end 
    return dropzeros(sparse(I, J, V/(π/ni)))
end

function Dϕ̄(ni::Int, nj::Int, order::Int)
    return Dϕ(ni, nj, order)
end

function S0(ni::Int, nj::Int, u::Function)
    I, J, V = (Int[], Int[], Complex{Float64}[])
    for j in 1:nj, i in  1:ni
        push!(I, join(i, j, ni, nj))
        push!(J, join(i, j, ni, nj)) 
        push!(V, u(collocation(i, j, ni, nj)...))
    end
    return sparse(I,J,V)
end

#---------------------------------------------------------------
# Convert to higher order operators
#---------------------------------------------------------------

if true
    E1 = zeros(30)
    E2 = zeros(30)
    EL = zeros(Complex, 30)
    h1 = zeros(30)
    h2 = zeros(30)
    l,m    = (6,4)
    order  = 8
    
    for index in 1:20
        println("Computing at index = $index")
        N1, N2 = (index*10, 2index*10)
        D1, D2 = (Dθ(N1, N2, order), Dϕ(N1, N2, order))
        Ylm    = map((x,y)->ScalarSPH(l,m,x,y), N1, N2)
        dYlmdθ = map((x,y)->dYdθ(l,m,x,y), N1, N2)
        dYlmdϕ = map((x,y)->dYdϕ(l,m,x,y), N1, N2)
        h1[index] =  π/N1
        h2[index] = 2π/N1
        E1[index] = LInf(D1*Ylm - dYlmdθ)
        E2[index] = LInf(D2*Ylm - dYlmdϕ)

        D1̄, D2̄ = (Dθ̄(N1, N2, order), Dϕ̄(N1, N2, order))
        S2 = S0(N1, N2, (x,y)->1/sin(x)^2)
        S1 = S0(N1, N2, (x,y)->cos(x)/sin(x))
        laplace = S1*D1 + D1̄*D1 + S2*D2̄*D2
        EL[index] = L2(laplace*Ylm + l*(l+1)*Ylm)
    end
    loglog(h1, h1.^8, "--")
    loglog(h1, h1.^4, "--")
    loglog(h1, h1.^2, "--")
    loglog(h1, E1, "r-o")
    loglog(h2, E2, "g-o")
    loglog(h1, EL, "k-o")
    savefig("convergence.pdf")
    close()
end

