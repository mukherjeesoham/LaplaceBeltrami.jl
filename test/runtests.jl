#---------------------------------------------------------------
# LaplaceOnASphere
# Soham 8/20
#---------------------------------------------------------------
using Test, LaplaceOnASphere

libraries = ["Coupling"]
libraries = ["Quadrature"]
libraries = ["Convergence"]

for file in libraries
    @info "Testing $file"
    include("$(file)Test.jl")
end


