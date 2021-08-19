#-----------------------------------------------------
# Check finite differences 
# Soham 07/2021
#-----------------------------------------------------
# TODO: 
#       - Test scalar derivatives
#       - Test convergence of scalar derivatives
#       - Test gradient 
#       - Test convergence of gradient
#       - Test scalar derivatives
#       - Test convergence of vector derivatives
#       - Test divergence 
#       - Test convergence of divergence
#       - Test Laplace
#       - Test convergence of Laplace

using LinearAlgebra, Test, FFTW
lmax = 3

# For finite difference operators to work we need
# points across the poles on the sphere. However, FastSphericalHarmonics 
# uses odd number of points. Hence, we'll interpolate. This also requires us
# to rethink the FD operators as maps rather than matrix operators.
function deriv_DFT(u::Vector{T})::Vector{T} where {T<:Real}
    N  = length(u)
    # FIXME: Fix length of arrays
    dũ = -im * (2π / N) * (-N/2 : 1 : N/2 - 1) .* fftshift(fft(u)) 
    return ifft(ifftshift(dũ))
end

function interp_DFT(u::Vector{T}, ϕ::T)::Vector{T} where {T<:Real}
    N  = length(u)
    ũ = fft(u) 
    k = 0:1:N-1
    φ = exp.(im .*k .* ϕ)
    return dot(ũ, φ)
end

function interp_DFT(u::Array{T,2}, i::Int, j::Int)::T where {T<:Real}
    θ, ϕ = collocation(i, j, size(u)...)
    return interp_DFT(u[i, :], ϕ + π) 
end

function Dθ(scalar::Array{T,2})::Array{T,2} where {T<:Real}
    stencil = [-1/2, 0, 1/2]
    dscalar = similar(scalar)
    for index in CartesianIndices(scalar)
        if (i - 1) <= 0 # north pole 
            dscalar[index] = dot(stencil, [interp_DFT(scalar[i,j]), scalar[i, j], scalar[i+1,j]]) 
        elseif (i + 1) >= size(scalar)[1] # south pole
            dscalar[index] = dot(stencil, [scalar[i-1,j], scalar[i, j], interp_DFT(scalar[i,j])]) 
        else # in the middle
            dscalar[index] = dot(stencil, [scalar[i-1, j], scalar[i, j], scalar[i, j]])
        end
    end
    return dscalar
end

function Dϕ(scalar::Array{T,2})::Array{T,2} where {T}
    ni, nj = size(scalar)
    dscalar = similar(scalar) 
    for j in 1:nj
        dscalar[j] .= deriv_DFT(scalar[:,j])  
    end
    return dscalar
end

# Test collocation points
x = map((x,y)->x, lmax)
y = map((x,y)->y, lmax)
for index in CartesianIndices(x)
    @test all((x[index], y[index]) .== collocation(index.I..., size(x)...))
end

# Test maps
f(x::Vector) = sin(x[1])*cos(x[2]) + tan(x[1]*x[2])
u = map((x,y)->f([x,y]), lmax) 
v = map((x,y)->f([x,y]), size(x)...) 
@test all(u .== v)


# Test scalar derivatives
@show typeof(u) 
dudϕ = Dϕ(u)
dudθ = Dθ(u)
