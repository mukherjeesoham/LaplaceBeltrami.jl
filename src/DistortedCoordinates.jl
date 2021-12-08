#---------------------------------------------------------------
# LaplaceOnASphere
# Soham 8/2020
# Choose a smooth coodinate transformation and compute the associated metric
#---------------------------------------------------------------
# (r, θ, ϕ) / (x, y, z) are the *good* coordinates (metric is diagonal)
# (s, μ, ν) / (u, v, w) are the *bad* coordinates  (metric is not diagonal)
# TODO: Check normalization of the surface normal
# TODO: Check surface integral
# TODO: Check Ricci scalar for both 2D and 3D space.

using StaticArrays
export q_μν
export uvw_of_xyz, xyz_of_uvw, rθϕ_of_sμν, sμν_of_rθϕ


Random.seed!(42)
A = rand(3,3)
R = eigen(A + A').vectors
# Does R work and not b?
b = zeros(3) 

function uvw_of_xyz(xyz::AbstractArray) 
    return SVector{3}(R*xyz + b)
end

function xyz_of_uvw(uvw::AbstractArray) 
    return SVector{3}(inv(R) * (uvw - b))
end

function rθϕ_of_sμν(sμν::AbstractArray) 
    return (cartesian2spherical ∘ xyz_of_uvw ∘ spherical2cartesian)(sμν)
end

function sμν_of_rθϕ(rθϕ::AbstractArray) 
    return (cartesian2spherical ∘ uvw_of_xyz ∘ spherical2cartesian)(rθϕ)
end

# dx^a / dX^b
# FIXME: Do we have NaNs here?
function J(sμν::AbstractArray)
    return jacobian(rθϕ_of_sμν, sμν)
end

# g_ab (X) = (dx^m /b dX^a) (dx_n / dX_)  g_mn (x) 
function q_sμν(sμν::AbstractArray)
    q_rθϕ  = (q ∘ rθϕ_of_sμν)(sμν)
    Jdrds  = J(sμν) 
    return SMatrix{3,3}(Jdrds * q_rθϕ * Jdrds')
end

# n_a (X) = (dx^m / dX^a) n_m (x)
function n_sμν(sμν::AbstractArray) 
    n_rθϕ = (n ∘ rθϕ_of_sμν)(sμν)
    Jdrds  = J(sμν)
    return SVector{3}(Jdrds * n_rθϕ)
end

# (2) g_ab (X) = (3) g_ab (X) - (3) n_a (X) (3) n_b (X)
function q_μν(μν::AbstractArray) 
    qsμν = q_sμν([1.0, μν...])   
    nsμν = n_sμν([1.0, μν...]) 
    return SMatrix{2,2}([qsμν[a,b] + nsμν[a] * nsμν[b] for a in 2:3, b in 2:3]) 
end
