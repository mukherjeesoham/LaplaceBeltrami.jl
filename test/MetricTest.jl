#---------------------------------------------------------------
# LaplaceOnASphere
# Soham 3/20
# Test metric functions
#---------------------------------------------------------------

S = SphericalHarmonics(4)

hab = zeros(Complex, (2,2))
for index in CartesianIndices(hab)
    a, b = index.I
    hab[index] = invhab(S, a, b, π/3, π/6)
end

display(hab)
@show sqrtdeth(S, π/3, π/6)
