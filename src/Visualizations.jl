#-----------------------------------------------------
# Visualize fields on a sphere and their modes
# Soham 05/2021
#-----------------------------------------------------

using Plots
export visualize

function visualize(C⁰::AbstractMatrix{T}) where {T<:Real}
    x = 1:size(C⁰)[1]
    plot(x, sum(abs, C⁰, dims=2)) 
    datetime = now()
    savefig("./plots/modes/S-$datetime.png")
end

function visualize(U::AbstractMatrix{T}, s::Symbol) where {T<:Real}
    @assert s == :contour
    contourf(U)
    datetime = now()
    savefig("./plots/modes/C-$datetime.png")
end

function visualize(C⁰::AbstractMatrix{T}) where {T<:StaticArrays.SVector{2, Float64}}
    x = 1:size(C⁰)[1]
    C1⁰ = map(x->x[1], C⁰)
    C2⁰ = map(x->x[2], C⁰)
    plot(x, sum(abs, C1⁰, dims=2)) 
    plot!(x, sum(abs, C2⁰, dims=2)) 
    datetime = now()
    savefig("./plots/modes/V-$datetime.png")
end

