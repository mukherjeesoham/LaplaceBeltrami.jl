using LaplaceOnASphere, Test

libraries = ["Monday"]

for file in libraries
    @info "Testing $file"
    include("$(file)Test.jl")
end


