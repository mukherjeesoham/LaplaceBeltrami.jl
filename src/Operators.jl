using LinearAlgebra

export M2N_Ylm, N2M_Ylm, M2N_Ψlm, N2M_Ψlm, S_Ylm, S_Ψlm

function M2N_Ylm(S::SphericalHarmonics)
    A = zeros(Complex, S.n*2(S.n), (S.l)^2 + 2*(S.l) + 1)
    for index in CartesianIndices(A)
        (i,j) = split(index.I[1], S.n)
        (l,m) = split(index.I[2])
        (θ,ϕ) = grid(i,j,S.n)
        A[index] = Ylm(l, m, θ, ϕ)
    end
    return A
end

function N2M_Ylm(S::SphericalHarmonics)
    return pinv(M2N_Ylm(S))
end

function M2N_Ψlm(S::SphericalHarmonics)
    A = zeros(Complex, 2*S.n*2(S.n), (S.l)^2 + 2*(S.l) + 1)
    for index in CartesianIndices(A)
        (i,j,a) = split3(index.I[1], S.n)
        (l,m) = split(index.I[2])
        (θ,ϕ) = grid(i,j,S.n)
        A[index] = (a == 1 ? dYlmdθ(l, m, θ, ϕ) : dYlmdϕ(l, m, θ, ϕ))
    end
    return A
end

function N2M_Ψlm(S::SphericalHarmonics)
    return pinv(M2N_Ψlm(S))
end

function S_Ylm(S::SphericalHarmonics, u::Function)
    N = S.n
    A = Diagonal(zeros(N*2N)) 
    for index in 1:N*2N
        i, j = split(index, N) 
        A[index, index] = u(grid(i, j, N)...)
    end
    return A
end

function S_Ψlm(S::SphericalHarmonics, u1::Function, u2::Function)
    N = S.n
    A = Diagonal(zeros(2N*2N)) 
    for index in 1:2N
        i, j, a = split3(index, N) 
        A[index, index] = (a == 1 ? u1(grid(i,j,N)...) : u2(grid(i,j,N)...))
    end
    return A
end
