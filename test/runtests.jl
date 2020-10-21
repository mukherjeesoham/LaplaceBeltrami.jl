#---------------------------------------------------------------
# LaplaceOnASphere
# Soham 8/20
#---------------------------------------------------------------
using Test, LaplaceOnASphere

libraries = ["Coupling"]
libraries = ["Convergence"]
libraries = ["LinearMaps"]

for file in libraries
    @info "Testing $file"
    include("$(file)Test.jl")
end


