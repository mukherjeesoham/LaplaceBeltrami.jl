#---------------------------------------------------------------
# LaplaceOnASphere
# Soham 5/20
# Construct new sparse FD operators that respect
# the symmetries around the poles for scalars and vectors
#---------------------------------------------------------------

using PyPlot, Arpack

#---------------------------------------------------------------
# Test convergence 
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

        # Laplace with the sinθ factors taken out
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

if true
    order  = 8
    @time N1, N2 = (30, 40)
    @time D1, D2 = (Dθ(N1, N2, order), Dϕ(N1, N2, order))
    @time D1̄, D2̄ = (Dθ̄(N1, N2, order), Dϕ̄(N1, N2, order))
    @time S2     = S0(N1, N2, (x,y)->1/sin(x)^2)
    @time S1     = S0(N1, N2, (x,y)->cos(x)/sin(x))
    @time laplace = S1*D1 + D1̄*D1 + S2*D2̄*D2
    @time λ, ϕ = eigs(laplace; nev=20, which=:SM)
    
    λexp = []
    for l in 0:10
        for m in -l:l
            append!(λexp, -l*(l+1))
        end
    end
    
    plot(λ, "ro")
    plot(λexp[1:10], "g-o"; markersize=4)
    savefig("eigenvalues.pdf")
    close()

    for i in 1:7
        imshow(reshape(real.(ϕ[:,i]), N1, N2))
        colorbar()
        savefig("eigenfunc$i.pdf")
        close()
    end

end

