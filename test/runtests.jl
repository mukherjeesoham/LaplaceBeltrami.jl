#---------------------------------------------------------------
# LaplaceOnASphere
# Soham 8/20
#---------------------------------------------------------------
using Test, LaplaceOnASphere

libraries = ["Orthogonality"]
libraries = ["Coupling"]

for file in libraries
    @info "Testing $file"
    include("$(file)Test.jl")
end


