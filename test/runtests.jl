using LaplaceOnASphere, Test

libraries = ["Operator"]
libraries = ["Transformation"]

for file in libraries
    @info "Testing $file"
    include("$(file)Test.jl")
end


