#---------------------------------------------------------------
# Use spin-weighted spherical harmonics to compute the 
# Laplace operator using FastSphericalHarmonics.jl
# Soham M 03/21
#---------------------------------------------------------------

using FastSphericalHarmonics, LinearAlgebra, CairoMakie
export gramschmidt, plot, evaluate

function Base. map(u::Function, lmax::Int)
   N = lmax + 1
   θ, ϕ = sph_points(N) 
   return [u(θ, ϕ) for θ in θ, ϕ in ϕ]
end

function quad(F⁰::Array{T,2}, lmax::Int) where {T}
    C⁰ = spinsph_transform(F⁰,0) 
    return 4π*C⁰[sph_mode(0,0)]
end

function LinearAlgebra.dot(u::Array{T,2}, v::Array{T,2}, lmax::Int) where {T} 
    return quad(u.*v, lmax)
end

function evaluate(u::Array{T,2}, lmax::Int) where {T}
    u1 = spinsph_evaluate(Vec2C(u[:,1], lmax), 0) 
    u2 = spinsph_evaluate(Vec2C(u[:,2], lmax), 0) 
    u3 = spinsph_evaluate(Vec2C(u[:,3], lmax), 0) 
    return (u1, u2, u3)
end

function gramschmidt(u1::Array{T,2}, u2::Array{T,2}, u3::Array{T,2}, lmax::Int) where {T}
    # Orthonormalize
    u1 = u1 ./ sqrt(dot(u1, u1, lmax))
    u2 = u2 - dot(u1, u2, lmax) .* u1
    u2 = u2 ./ sqrt(dot(u2, u2, lmax))
    u3 = u3 - dot(u1, u3, lmax) .* u1 - dot(u2, u3, lmax) .* u2
    u3 = u3 ./ sqrt(dot(u3, u3, lmax))

    # Check if the process worked
    if false
        @show dot(u1, u1, lmax)
        @show dot(u1, u2, lmax)
        @show dot(u1, u3, lmax)
        @show dot(u2, u1, lmax)
        @show dot(u2, u2, lmax)
        @show dot(u2, u3, lmax)
        @show dot(u3, u1, lmax)
        @show dot(u3, u2, lmax)
        @show dot(u3, u3, lmax)
    end
    
    return (u1, u2, u3)
end

function plot(u1::Array{T,2}, u2::Array{T,2}, u3::Array{T,2}, lmax::Int, string::String) where {T}
    N = lmax + 1
    latglq, longlq = sph_points(N) 
    println("Starting plotting figure")
    @time begin
        fig = Figure(resolution = (1600, 400))
        _, co1 = contourf(fig[1, 1][1, 1], latglq, longlq, u1, levels = 10)
        Colorbar(fig[1, 1][1, 2], co1, width = 20)
        _, co2 = contourf(fig[1, 2][1, 1], latglq, longlq, u2, levels = 10)
        Colorbar(fig[1, 2][1, 2], co2, width = 20)
        _, co3 = contourf(fig[1, 3][1, 1], latglq, longlq, u3, levels = 10)
        Colorbar(fig[1, 3][1, 2], co3, width = 20)
        save("output/$string.pdf", fig)
    end
end

function plot(u1::Array{T,2}, u2::Array{T,2}, lmax::Int, string::String) where {T}
    N = lmax + 1
    latglq, longlq = sph_points(N) 
    println("Starting plotting figure")
    @time begin
        fig = Figure(resolution = (1000, 400))
        _, co1 = contourf(fig[1, 1][1, 1], latglq, longlq, u1, levels = 10)
        Colorbar(fig[1, 1][1, 2], co1, width = 20)
        _, co2 = contourf(fig[1, 2][1, 1], latglq, longlq, u2, levels = 10)
        Colorbar(fig[1, 2][1, 2], co2, width = 20)
        save("output/$string.pdf", fig)
    end
end

function plot(u1::Array{T,2}, lmax::Int, string::String) where {T}
    N = lmax + 1
    latglq, longlq = sph_points(N) 
    println("Starting plotting figure")
    @time begin
        fig = Figure(resolution = (400, 400))
        _, co1 = contourf(fig[1, 1][1, 1], latglq, longlq, u1, levels = 10)
        Colorbar(fig[1, 1][1, 2], co1, width = 20)
        save("output/$string.pdf", fig)
    end
end
