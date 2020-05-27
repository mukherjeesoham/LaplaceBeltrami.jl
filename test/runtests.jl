using LaplaceOnASphere, Test

libraries = ["Transformation", "Operator", "Jacobian", "Metric", "Eigen"]
libraries = ["NewOperator"]

for file in libraries
    @info "Testing $file"
    include("$(file)Test.jl")
end


