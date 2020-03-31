using LaplaceOnASphere, Test

libraries = ["Transformation"]
libraries = ["Operator"]
libraries = ["Eigen"]

for file in libraries
    @info "Testing $file"
    include("$(file)Test.jl")
end


