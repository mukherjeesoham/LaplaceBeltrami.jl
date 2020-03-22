function analyticYlm(S::SphericalHarmonics, ulm::Array{Any, 1}, θ::Number, ϕ::Number)
    u = 0
    for l in 0:S.l, m in -l:l
        u += ulm[join(l,m)]*Ylm(l, m, θ, ϕ)
    end 
    return u
end

function analyticΨlm(S::SphericalHarmonics, ulm::Array{Any, 1}, a::Int, θ::Number, ϕ::Number)
    u = 0
    for l in 0:S.l, m in -l:l
        u += ulm[join(l,m)]*Ψlm(l, m, θ, ϕ)[a]
    end 
    return u
end

S = SphericalHarmonics(2, 4)
u = map_to_grid(S, (x, y)->sin(x))
ulm = N2M_Ylm(S)*u
ua  = map_to_grid(S, (x,y)->analyticYlm(S, ulm, x, y))
@test maximum(abs.(u - ua)) < 1e-12

uvec = map_to_grid(S, (x,y)->sin(x), (x,y)->cos(y))
uveclm = N2M_Ψlm(S)*uvec

# Rewrite the modal array
for l in 0:S.l, m in -l:l 
    if l == 2
        uveclm[join(l,m)] = Complex(1,0) 
    else
        uveclm[join(l,m)] = Complex(0,0)
    end
end

# Now construct the nodal array
uveca  = map_to_grid(S, (x,y)->analyticΨlm(S, uveclm, 1, x, y),
                        (x,y)->analyticΨlm(S, uveclm, 2, x, y))
uvecalm = N2M_Ψlm(S)*uveca

# Now compare the modal space
@test maximum(abs.(uvecalm - uveclm)) < 1e-12
