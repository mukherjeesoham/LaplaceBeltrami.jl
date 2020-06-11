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

#---------------------------------------------------------------
# Do a coordinate transformation and check if 
# we recover the eigenfunctions
# We work with the coordinates μ, ν.
# Ideally, we'd want to take the derivatives of the tensor
# components numerically, but then again, we'll need
# to define new derivative operators that take into account
# the parity of the tensor components across the poles. 
# See Table 1 in <https://arxiv.org/abs/1211.6632>
#---------------------------------------------------------------

N1, N2 = 10, 20
Dμ, Dν = Dθ(N1, N2), Dϕ(N1, N2)
Dμ̄, Dν̄ = Dθ̄(N1, N2), Dϕ̄(N1, N2)

h11   = S0(N1, N2, (μ,ν)->1)
h12   = h21 = S0(N1, N2, (μ,ν)->0)
h22   = S0(N1, N2, (μ,ν)->1)
csc2μ = S0(N1, N2, (μ,ν)->csc(μ)^2)
cscμ  = S0(N1, N2, (μ,ν)->csc(μ))
cotμ  = S0(N1, N2, (μ,ν)->cot(μ))

# Compute the tensor component derivatives in advance
# These parity conditions however, are not complicated
# for the operations on tensors and "work" the same way
# as for scalars; i.e. no sign change. 
D1h11 = Dμ*h11 
D2h11 = Dν*h11
D1h12 = D1h21 = Dμ*h12
D2h12 = D2h21 = Dν*h12
D1h22 = Dμ*h22
D2h22 = Dν*h22

D1    = Dμ
D2    = Dν
D1D1  = Dμ̄*Dμ
D2D2  = Dν̄*Dν
D1D2  = Dμ̄*Dν
D2D1  = Dν̄*Dμ
invdetg2 = 1/((h12*h21 - h11*h22).^2)

# Now construct the laplace operator
# TODO: Figure out D1D2 or D2D1 for the mixed derivatives
laplace = (invdetg2*(-h12*h21*h22 + h11*h22*h22)*D1D1 
         + invdetg2*(-h11*h22*h21*csc2μ + h11*h11*h22*csc2μ)*D2*D2
         + invdetg2*((1/2)*D2h22*h11*h11*csc2μ + (1/2)*D2h21*h11*h12*csc2μ 
                     + (1/2)*D2h12*h11*h21*csc2μ - D2h11*h12*h21*csc2μ 
                     + (1/2)*D2h11*h11*h22*csc2μ + (1/2)*D1h22*h11*h12*cscμ 
                     - (1/2)*D1h21*h12*h12*cscμ  + (1/2)*D1h12*h12*h21*cscμ 
                     - D1h12*h11*h22*cscμ + (1/2)*D1h11*h12*h22*cscμ)*D2
         + invdetg2*(-D1h22*h12*h21 + (1/2)*D1h22*h11*h22 
                     + D1h21*h12*h22 + (1/2)*D1h12*h21*h22
                     - (1/2)*D1h11*h22*h22 - h12*h21*h22*cotμ 
                     + h11*h22*h22*cotμ + D2h22*h11*h21*cscμ
                     + (1/2)*D2h21*h12*h21*cscμ  - (1/2)*D2h12*h21*h21*cscμ
                     - D2h21*h11*h22*cscμ + (1/2)*D2h11*h21*h22*cscμ)*D1
         + invdetg2*(h12*h12*cscμ + h12*h21*h21*cscμ - h11*h12*h22*cscμ - h11*h21*h22*cscμ)*D1D2) 

