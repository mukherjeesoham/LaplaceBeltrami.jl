#-----------------------------------------------------
# Implement filtering for scalar and vector spherical
# harmonics
# Soham 05/2021
#-----------------------------------------------------
using FastSphericalHarmonics

function lmax(C⁰::AbstractMatrix{T}) where {T<:Number}
    return first(size(C⁰)) - 1
end

function Base.filter(ulm::AbstractMatrix{T}, s::Int) where {T}
    mask = 0.0 .* ulm
    for l in 0:lmax(ulm), m in (-l):l 
        if abs(m) <= l
            if s == 0
                mask[spinsph_mode(s, l, m)] = 1.0
            elseif abs(s) == 1
                mask[spinsph_mode(s, l, m)] = [1.0, 1.0]
            end
        end
    end
    return mask .* C⁰ 
end

function prolongate(ulm::AbstractMatrix{T}, s::Int) where {T}
    # FIXME: Does this work?
    pulm = zeros(T,  2 * lmax(ulm) + 1, 4 *lmax(ulm) + 1)
    for l in 0:lmax(ulm), m in (-l):l 
        if abs(m) <= l
            pulm[spinsph_mode(s, l, m)] = ulm[spinsph_mode(s, l, m)] 
        end
    end
    return pulm
end


function restrict(ulm::AbstractMatrix{T}) where {T}
    return ulm[1:(lmax(ulm) ÷ 2) + 1, 1:2*(lmax(ulm) ÷ 2) + 1]
end
