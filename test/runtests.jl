using LaplaceOnASphere, Test

libraries = ["Eigen"]

for file in libraries
    @info "Testing $file"
    include("$(file)Test.jl")
end


