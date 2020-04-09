using LaplaceOnASphere, Test

libraries = ["Transformation"]
libraries = ["Operator"]
libraries = ["Eigen"]
libraries = ["Metric"]

for file in libraries
    @info "Testing $file"
    include("$(file)Test.jl")
end


