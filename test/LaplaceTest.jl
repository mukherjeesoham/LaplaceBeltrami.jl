#---------------------------------------------------------------
# Use spin-weighted spherical harmonics to compute the 
# Laplace operator using FastSphericalHarmonics.jl
# Soham M 03/21
#---------------------------------------------------------------

using Test, FastSphericalHarmonics, Arpack

# Compute the Laplace operator
lmax = 23
F⁰ = map((μ,ν)->Ylm(2,-1,μ,ν) + Ylm(2,0,μ,ν) + Ylm(2,1,μ,ν), lmax) 
C⁰ = spinsph_transform(F⁰, 0)
A  = Laplace{Float64}(lmax)
λ, u = eigs(A; nev=60, which=:LR)
@show λ[2:4]

# Orthonormalize the eigenfunctions
u1, u2, u3 = evaluate(u[:, 2:4], lmax) 
x, y, z = gramschmidt(u1, u2, u3, lmax)

# Compute the derivatives and hence the metric transformation
# FIXME: Check if the coordinate transformation here is nice. i.e., if the x, y, z coordinates
# are indeed just a rotation of the x, y, z after a rotation.
function cart2spherical(x::Array{T,2}, y::Array{T,2}, z::Array{T,2}) where {T} 
    # TODO: Check if r is constant
    @assert all(imag.(x) .< 1e-12)
    @assert all(imag.(y) .< 1e-12) 
    @assert all(imag.(z) .< 1e-12)
    x = real.(x)
    y = real.(y)
    z = real.(z)
    r = sqrt.(x.^2 + y.^2 + z.^2)
    θ = acos.(z ./ r)
    ϕ = atan.(y, x) 
    return (θ, ϕ)
end

function derivative(F⁰::Array{T,2}) where {T}
    C⁰ = spinsph_transform(Complex.(F⁰), 0) 
    ðC¹ = spinsph_eth(C⁰, 0)
    ðF¹ = spinsph_evaluate(ðC¹, 1)
    ∂θF = real.(ðF¹) 
    cscθ∂ϕF = imag(ðF¹)
    return (∂θF, cscθ∂ϕF) 
end

function qinv(a::Int, b::Int) 
    if a == b == 1
        return 1
        return map((μ,ν)->1, lmax)
    elseif a == b == 2
        return map((μ,ν)->csc(μ), lmax)
    else
        return map((μ,ν)->0, lmax)
    end
end

# TODO: Check this routine with an understood coordinate transformation. Actually check the entire 
# routine with a known coordinate transformation
function qtransform(x::Array{T,2}, y::Array{T,2}, z::Array{T,2}, lmax::Int) where {T}
    μ, ν = cart2spherical(x,y,z)
    dμdθ, cscθdμdϕ = derivative(μ)  
    dνdθ, cscθdνdϕ = derivative(ν)  
    sinθ =  map((μ,ν)->sin(μ), lmax)
    dμdϕ = sinθ .* cscθdμdϕ
    dνdϕ = sinθ .* cscθdνdϕ
    ginvμμ = dμdθ .* dμdθ .* qinv(1,1) + dμdθ .* dμdϕ .* qinv(1,2) + dμdϕ .* dμdθ .* qinv(2,1) +  dμdϕ .* dμdϕ .* qinv(2,2)
    ginvμν = dμdθ .* dνdθ .* qinv(1,1) + dμdθ .* dνdϕ .* qinv(1,2) + dμdϕ .* dνdθ .* qinv(2,1) +  dμdϕ .* dνdϕ .* qinv(2,2)
    ginvνν = dνdθ .* dνdθ .* qinv(1,1) + dνdθ .* dνdϕ .* qinv(1,2) + dνdϕ .* dνdθ .* qinv(2,1) +  dνdϕ .* dνdϕ .* qinv(2,2)
    return (ginvμμ, ginvμν, ginvνν)
end

μ, ν = cart2spherical(x,y,z)
plot(μ, ν, lmax, "coordinates")
# g11, g12, g13 = qtransform(x, y, z, lmax)

