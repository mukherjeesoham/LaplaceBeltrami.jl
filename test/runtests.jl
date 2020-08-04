using LaplaceOnASphere, Test

libraries = ["Debug"]

for file in libraries
    @info "Testing $file"
    include("$(file)Test.jl")
end


