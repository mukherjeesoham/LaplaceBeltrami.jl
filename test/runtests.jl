using LaplaceOnASphere, Test

libraries = ["Operator"]

for file in libraries
    @info "Testing $file"
    include("$(file)Test.jl")
end


