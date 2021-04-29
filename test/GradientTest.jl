#-----------------------------------------------------
# Test gradients with spin-weighted spherical harmonics
# Soham 04/2021
#-----------------------------------------------------

using FastSphericalHarmonics, LinearAlgebra, Test, ForwardDiff

function chop2(x)
    x = (abs(x) < 1e-10 ? 0.0 : x)
    return x
end

chop2(x::Complex) = Complex(chop2(real(x)), chop2(imag(x)))

lmax  = 6
scalar(x) = sin(x[1])*sin(x[2])
vector(x) = ForwardDiff.gradient(scalar, x)
# vector(x) = [-sin(x[1])*sin(x[2]), cos(x[1])*cos(x[2])] 

# Use AD to compute the gradient
F⁰    = map((μ,ν)->scalar([μ, ν]), lmax) 
∂F⁰   = map((μ,ν)->Complex(vector([μ,ν])...), lmax)
sinθ  = map((μ,ν)->sin(μ), lmax) 
cscθ  = map((μ,ν)->csc(μ), lmax) 

# Use spin-weighted spherical harmonics to compute the gradient. Note
# we have added the factor of (-1 / √2) to fix things for the case with cosθ.
# FIXME: There's something more concerning than factors of √2 and minus signs here.  
C⁰    = spinsph_transform(F⁰, 0)
ðC¹   = spinsph_eth(Complex.(C⁰), 0)
NF¹   = spinsph_evaluate(ðC¹, 1) 
@test_broken -real.(∂F⁰) .+ 1 ≈ 1 .+ real.(NF¹)
@test_broken -imag.(∂F⁰) .+ 1 ≈ 1 .+ sinθ .* imag.(NF¹)

@debug display(chop2.(C⁰))

# display(real.(NF¹))
# display(real.(∂F⁰))

# Check what the coefficents should be
C¹ = spinsph_transform(-Complex.(real.(∂F⁰), cscθ .* imag.(∂F⁰)), 1)

display(chop2.(ðC¹))
display(chop2.(C¹))

F¹ = spinsph_evaluate(C¹, 1) 
@test real.(∂F⁰) .+ 1 ≈  - real.(F¹) .+ 1
@test imag.(∂F⁰) .+ 1 ≈  - sinθ .* imag.(F¹) .+ 1


@debug println("C¹i real and imag")
@debug display(chop!(real.(C¹)))
@debug display(chop!(imag.(C¹)))
@debug println("\nðC¹ real and imag")
@debug display(chop!(real.(ðC¹)))
@debug display(chop!(imag.(ðC¹)))

if false
# Checking parity and orthogonality. Start with a Ylm, and then compute 
# sYlm using eth and eth bar (ladder) operators.
s0Ylm = map((μ,ν)->Ylm(2,2,μ,ν) + Ylm(2,0,μ,ν) + Ylm(2,-2,μ,ν), lmax) 
C⁰ = spinsph_transform(s0Ylm, 0)
# FIXME: Why do I need to cast coefficent arrays to be explicitly complex here? 
C¹ = spinsph_eth(Complex.(C⁰), 0)       
C² = spinsph_eth(Complex.(C¹), 1)
C̄¹ = spinsph_ethbar(C², 2)
C̄⁰ = spinsph_ethbar(C¹, 1)
s1Ylm = spinsph_evaluate(C¹, 1) 
s̄1Ylm = spinsph_evaluate(C̄¹, 1) 

# The ladder operators should preserve the coefficents
@show maximum(abs.(C⁰ - real.(C̄⁰)))
@test C⁰ ≈ real.(C̄⁰)
end
