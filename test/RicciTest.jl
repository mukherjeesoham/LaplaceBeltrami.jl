#---------------------------------------------------------------
#e LaplaceOnASphere
# Soham 9/20
# Compute the Ricci scalar
#---------------------------------------------------------------
# Convert metric components from spherical polar to Cartesian
# Take the second derivatives of the metric in Cartesian coordinates using scalar
# spherical harmonics. 
# Compute the Ricci scalar from the metric derivatives

using Einsum, TensorOperations

function convert_tensor_components_to_cartesian(SH::SphericalHarmonics{T},
                                                gθθ::Array{Complex{T},1}, 
                                                gθϕ::Array{Complex{T},1}, 
                                                gϕϕ::Array{Complex{T},1})::Dict where {T}

    # gxx corresponds to the inverse metric with indices up; we assume grr = 1 for invertibility.
    dxdr = map(SH, (μ, ν)-> sin(μ)*cos(ν)) 
    dxdθ = map(SH, (μ, ν)-> cos(μ)*sin(ν)) 
    dxdϕ = map(SH, (μ, ν)->-sin(μ)*sin(ν))

    dydr = map(SH, (μ, ν)-> sin(μ)*sin(ν)) 
    dydθ = map(SH, (μ, ν)-> cos(μ)*sin(ν)) 
    dydϕ = map(SH, (μ, ν)-> sin(μ)*cos(ν))
               
    dzdr = map(SH, (μ, ν)-> cos(μ)) 
    dzdθ = map(SH, (μ, ν)->-sin(μ)) 
    dzdϕ = map(SH, (μ, ν)-> 0)

    grr = 1

    # Store all the components in a dictionary
    g   = Dict()
    g["gxx"] = dxdr.*dxdr.*grr + dxdθ.*dxdθ.*gθθ + dxdθ.*dxdϕ.*gθϕ + dxdϕ.*dxdϕ.*gϕϕ 
    g["gxy"] = dxdr.*dydr.*grr + dxdθ.*dydθ.*gθθ + dxdθ.*dydϕ.*gθϕ + dxdϕ.*dydϕ.*gϕϕ
    g["gxz"] = dxdr.*dzdr.*grr + dxdθ.*dzdθ.*gθθ + dxdθ.*dzdϕ.*gθϕ + dxdϕ.*dzdϕ.*gϕϕ 
    g["gyy"] = dydr.*dydr.*grr + dydθ.*dydθ.*gθθ + dydθ.*dydϕ.*gθϕ + dydϕ.*dydϕ.*gϕϕ
    g["gyz"] = dydr.*dzdr.*grr + dydθ.*dzdθ.*gθθ + dydθ.*dzdϕ.*gθϕ + dydϕ.*dzdϕ.*gϕϕ 
    g["gzz"] = dzdr.*dzdr.*grr + dzdθ.*dzdθ.*gθθ + dzdθ.*dzdϕ.*gθϕ + dzdϕ.*dzdϕ.*gϕϕ

    return g
end

function convert_vector_components_to_cartesian(SH::SphericalHarmonics{T},
                                                vθ::Array{Complex{T},1}, 
                                                vϕ::Array{Complex{T},1})::NTuple{3, Array{Complex{T},1}} where {T}
    # vx corresponds to the components of a vector, and not a co-vector. We have also  
    # assumed dx/dr is zero.
    dxdr = map(SH, (μ, ν)-> sin(μ)*cos(ν)) 
    dxdθ = map(SH, (μ, ν)-> cos(μ)*sin(ν)) 
    dxdϕ = map(SH, (μ, ν)->-sin(μ)*sin(ν))

    dydr = map(SH, (μ, ν)-> sin(μ)*sin(ν)) 
    dydθ = map(SH, (μ, ν)-> cos(μ)*sin(ν)) 
    dydϕ = map(SH, (μ, ν)-> sin(μ)*cos(ν))
               
    dzdr = map(SH, (μ, ν)-> cos(μ)) 
    dzdθ = map(SH, (μ, ν)->-sin(μ)) 
    dzdϕ = map(SH, (μ, ν)-> 0)

    vx = dxdθ.*vθ + dxdϕ.*vϕ
    vy = dydθ.*vθ + dydϕ.*vϕ
    vz = dzdθ.*vθ + dzdϕ.*vϕ
    return (vx, vy, vz)
end

function derivatives(SH::SphericalHarmonics{T}, u::Array{Complex{T},1})::NTuple{2, Array} where {T}
    S, S̄ = scalar_op(SH)
    V, V̄ = vector_op(SH)
    du   = reshape(V*(S̄*u), (:,2))
    return (du[:,1], du[:,2])
end

function invert_metric(invg::Dict)
    invgxx, invgxy, invgxz, invgyy, invgyz, invgzz = (invg["gxx"], invg["gxy"], invg["gxz"], 
                                                      invg["gyy"], invg["gyz"], invg["gzz"])
    gxx, gxy, gxz, gyy, gyz, gzz = (similar(invgxx), similar(invgxy), similar(invgxz), 
                                    similar(invgyy), similar(invgyz), similar(invgzz))
    for index in CartesianIndices(invgxx)
        ginv = inv([invgxx[index] invgxy[index] invgxz[index];
                    invgxy[index] invgyy[index] invgyz[index];
                    invgxz[index] invgyz[index] invgzz[index]])
        (gxx[index], gxy[index], gxz[index], 
         gyy[index], gyz[index], gzz[index])  = (ginv[1,1], ginv[1,2], ginv[1,3], 
                                                 ginv[2,2], ginv[2,3], ginv[3,3])
    end
    
    g = Dict()
    g["gxx"] = gxx
    g["gxy"] = gxy
    g["gxz"] = gxz
    g["gyy"] = gyy
    g["gyz"] = gyz
    g["gzz"] = gzz
    return g
end

function first_derivatives_of_the_metric(g::Dict)::Dict
    # Compute first derivatives
    gxxdθ, gxxdϕ = derivatives(SH, g["gxx"]) 
    gxydθ, gxydϕ = derivatives(SH, g["gxy"]) 
    gxzdθ, gxzdϕ = derivatives(SH, g["gxz"]) 
    gyydθ, gyydϕ = derivatives(SH, g["gyy"]) 
    gyzdθ, gyzdϕ = derivatives(SH, g["gyz"]) 
    gzzdθ, gzzdϕ = derivatives(SH, g["gzz"]) 
    
    dgdx   = Dict()
    # Project them back onto Cartesian coordinates
    dgdx["gxxdx"], dgdx["gxxdy"], dgdx["gxxdz"] = convert_vector_components_to_cartesian(SH, gxxdθ, gxxdϕ) 
    dgdx["gxydx"], dgdx["gxydy"], dgdx["gxydz"] = convert_vector_components_to_cartesian(SH, gxydθ, gxydϕ) 
    dgdx["gxzdx"], dgdx["gxzdy"], dgdx["gxzdz"] = convert_vector_components_to_cartesian(SH, gxzdθ, gxzdϕ) 
    dgdx["gyydx"], dgdx["gyydy"], dgdx["gyydz"] = convert_vector_components_to_cartesian(SH, gyydθ, gyydϕ) 
    dgdx["gyzdx"], dgdx["gyzdy"], dgdx["gyzdz"] = convert_vector_components_to_cartesian(SH, gyzdθ, gyzdϕ) 
    dgdx["gzzdx"], dgdx["gzzdy"], dgdx["gzzdz"] = convert_vector_components_to_cartesian(SH, gzzdθ, gzzdϕ) 

    return dgdx
end

function second_derivatives_of_the_metric(dgdx::Dict)

    # Compute second derivatives
    gxxdxdθ, gxxdxdϕ = derivatives(SH, dgdx["gxxdx"]) 
    gxxdydθ, gxxdydϕ = derivatives(SH, dgdx["gxxdy"]) 
    gxxdzdθ, gxxdzdϕ = derivatives(SH, dgdx["gxxdz"]) 
    gxydxdθ, gxydxdϕ = derivatives(SH, dgdx["gxydx"]) 
    gxydydθ, gxydydϕ = derivatives(SH, dgdx["gxydy"]) 
    gxydzdθ, gxydzdϕ = derivatives(SH, dgdx["gxydz"]) 
    gxzdxdθ, gxzdxdϕ = derivatives(SH, dgdx["gxzdx"]) 
    gxzdydθ, gxzdydϕ = derivatives(SH, dgdx["gxzdy"]) 
    gxzdzdθ, gxzdzdϕ = derivatives(SH, dgdx["gxzdz"]) 
    gyydxdθ, gyydxdϕ = derivatives(SH, dgdx["gyydx"]) 
    gyydydθ, gyydydϕ = derivatives(SH, dgdx["gyydy"]) 
    gyydzdθ, gyydzdϕ = derivatives(SH, dgdx["gyydz"]) 
    gyzdxdθ, gyzdxdϕ = derivatives(SH, dgdx["gyzdx"]) 
    gyzdydθ, gyzdydϕ = derivatives(SH, dgdx["gyzdy"]) 
    gyzdzdθ, gyzdzdϕ = derivatives(SH, dgdx["gyzdz"]) 
    gzzdxdθ, gzzdxdϕ = derivatives(SH, dgdx["gzzdx"]) 
    gzzdydθ, gzzdydϕ = derivatives(SH, dgdx["gzzdy"]) 
    gzzdzdθ, gzzdzdϕ = derivatives(SH, dgdx["gzzdz"]) 

    ddgdxx = Dict()
    # Project second derivatives back to Cartesian components
    ddgdxx["gxxdxdx"], ddgdxx["gxxdxdy"], ddgdxx["gxxdxdz"] = convert_vector_components_to_cartesian(SH, gxxdxdθ, gxxdxdϕ) 
    ddgdxx["gxxdydx"], ddgdxx["gxxdydy"], ddgdxx["gxxdydz"] = convert_vector_components_to_cartesian(SH, gxxdydθ, gxxdydϕ) 
    ddgdxx["gxxdzdx"], ddgdxx["gxxdzdy"], ddgdxx["gxxdzdz"] = convert_vector_components_to_cartesian(SH, gxxdzdθ, gxxdzdϕ) 
    ddgdxx["gxydxdx"], ddgdxx["gxydxdy"], ddgdxx["gxydxdz"] = convert_vector_components_to_cartesian(SH, gxydxdθ, gxydxdϕ) 
    ddgdxx["gxydydx"], ddgdxx["gxydydy"], ddgdxx["gxydydz"] = convert_vector_components_to_cartesian(SH, gxydydθ, gxydydϕ) 
    ddgdxx["gxydzdx"], ddgdxx["gxydzdy"], ddgdxx["gxydzdz"] = convert_vector_components_to_cartesian(SH, gxydzdθ, gxydzdϕ) 
    ddgdxx["gxzdxdx"], ddgdxx["gxzdxdy"], ddgdxx["gxzdxdz"] = convert_vector_components_to_cartesian(SH, gxzdxdθ, gxzdxdϕ) 
    ddgdxx["gxzdydx"], ddgdxx["gxzdydy"], ddgdxx["gxzdydz"] = convert_vector_components_to_cartesian(SH, gxzdydθ, gxzdydϕ) 
    ddgdxx["gxzdzdx"], ddgdxx["gxzdzdy"], ddgdxx["gxzdzdz"] = convert_vector_components_to_cartesian(SH, gxzdzdθ, gxzdzdϕ) 
    ddgdxx["gyydxdx"], ddgdxx["gyydxdy"], ddgdxx["gyydxdz"] = convert_vector_components_to_cartesian(SH, gyydxdθ, gyydxdϕ) 
    ddgdxx["gyydydx"], ddgdxx["gyydydy"], ddgdxx["gyydydz"] = convert_vector_components_to_cartesian(SH, gyydydθ, gyydydϕ) 
    ddgdxx["gyydzdx"], ddgdxx["gyydzdy"], ddgdxx["gyydzdz"] = convert_vector_components_to_cartesian(SH, gyydzdθ, gyydzdϕ) 
    ddgdxx["gyzdxdx"], ddgdxx["gyzdxdy"], ddgdxx["gyzdxdz"] = convert_vector_components_to_cartesian(SH, gyzdxdθ, gyzdxdϕ) 
    ddgdxx["gyzdydx"], ddgdxx["gyzdydy"], ddgdxx["gyzdydz"] = convert_vector_components_to_cartesian(SH, gyzdydθ, gyzdydϕ) 
    ddgdxx["gyzdzdx"], ddgdxx["gyzdzdy"], ddgdxx["gyzdzdz"] = convert_vector_components_to_cartesian(SH, gyzdzdθ, gyzdzdϕ) 
    ddgdxx["gzzdxdx"], ddgdxx["gzzdxdy"], ddgdxx["gzzdxdz"] = convert_vector_components_to_cartesian(SH, gzzdxdθ, gzzdxdϕ) 
    ddgdxx["gzzdydx"], ddgdxx["gzzdydy"], ddgdxx["gzzdydz"] = convert_vector_components_to_cartesian(SH, gzzdydθ, gzzdydϕ) 
    ddgdxx["gzzdzdx"], ddgdxx["gzzdzdy"], ddgdxx["gzzdzdz"] = convert_vector_components_to_cartesian(SH, gzzdzdθ, gzzdzdϕ) 

    return ddgdxx
end

function translate(index::CartesianIndex{N})::String where {N}
    coords = ["x","y","z"]
    if N == 2
        if index.I[1] < index.I[2]
            return "g" * coords[index.I[1]] * coords[index.I[2]] 
        else
            return "g" * coords[index.I[2]] * coords[index.I[1]] 
        end
    elseif N == 3
        if index.I[1] < index.I[2]
            return "g" * coords[index.I[1]] * coords[index.I[2]] * "d" * coords[index.I[3]]  
        else
            return "g" * coords[index.I[2]] * coords[index.I[1]] * "d" * coords[index.I[3]]  
        end
    elseif N == 4
        if index.I[1] < index.I[2]
            return "g" * coords[index.I[1]] * coords[index.I[2]] * "d" * coords[index.I[3]] * "d" * coords[index.I[4]]
        else
            return "g" * coords[index.I[2]] * coords[index.I[1]] * "d" * coords[index.I[3]] * "d" * coords[index.I[4]]
        end
    end
end

struct Tensor{T,D}
    value::Array{Complex{T},D}
end

function Base.zero(u::Type{Tensor{T}}) where {T}
    # FIXME: How to pass the 3 there? 
    return Tensor(zeros(T,3,3))
end

function Base. *(u::Tensor{T}, v::Tensor{T}) where {T}
    return Tensor(u.val.*v.val)
end

function Base. +(u::Tensor{T}, v::Tensor{T}) where {T}
    return Tensor(u.val .+ v.val)
end


function convert_dict_to_arrays(invg::Dict, dgdx::Dict, ddgdxx::Dict)::NTuple{3, Array{Tensor}}
    # Store these into arrays to use repeated summation
    dg  = Array{Array,3}(undef,3,3,3)
    for index in CartesianIndices(dg)
        dg[index] = dgdx[translate(index)]  
    end

    ddg  = Array{Array,4}(undef,3,3,3,3)
    for index in CartesianIndices(ddg)
        ddg[index] = ddgdxx[translate(index)]  
    end

    ginv  = Array{Array,2}(undef,3,3)
    for index in CartesianIndices(ginv)
        ginv[index] = invg[translate(index)]  
    end

    return (Tensor.(ginv), Tensor.(dg), Tensor.(ddg))
end

function ricci(SH::SphericalHarmonics{T}, invgθθ::Array{Complex{T},1}, 
                                          invgθϕ::Array{Complex{T},1},
                                          invgϕϕ::Array{Complex{T},1})::Array{Complex{T},1} where {T}
    invg = convert_tensor_components_to_cartesian(SH, invgθθ, invgθϕ, invgϕϕ)
    println("Finished converting metric components to Cartesian components")
    g    = invert_metric(invg)
    println("Finished inverting metric")
    dgdx = first_derivatives_of_the_metric(g)
    println("Finished computing first derivatives metric")
    ddgdxx  = second_derivatives_of_the_metric(dgdx)
    println("Finished computing second derivatives metric")
    invg, dg, ddg = convert_dict_to_arrays(invg, dgdx, ddgdxx) 
    println("Finished converting dictionaries to arrays")

    R = Tensor(map(SH, (μ,ν)->0))
    # FIXME: Fix the zero(T) error. This should be not too difficult to fix.

    @show typeof(invg[1,1])
    @tensor begin 
        R = invg[i,j]*invg[i,j]
        # R:= invg[i,j]*(- (1/2)*(ddg[i,j,a,b] + ddg[a,b,i,j] - ddg[i,j,j,a] - ddg[j,b,i,a])*invg[a,b]
                       # + (1/2)*(dg[a,c,i]*dg[b,d,j] + dg[i,c,a]*dg[j,d,b] - dg[i,c,a]*dg[j,b,d])*invg[a,b]*invg[c,d]
                       # - (1/4)*(dg[j,c,i] + dg[i,c,j] - dg[i,j,c])*(2*dg[b,d,a] -dg[a,b,d])*invg[a,b]*invg[c,d])
    end

    return R
end

# Test the Ricci scalar
SH  = SphericalHarmonics(4)
hμμ = map(SH, (μ,ν)->hinv(1,1,μ,ν))
hμν = map(SH, (μ,ν)->hinv(1,1,μ,ν))
hνν = map(SH, (μ,ν)->hinv(1,1,μ,ν))

# Let the games begin.
ricci(SH, hμμ, hμν, hνν)
