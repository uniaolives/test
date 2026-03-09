# OrbPhysics.jl

mutable struct Orb
    stability::Float64
    frequency::Float64
end

function collapse(orb::Orb, data::Vector{Float64})
    # Colapso probabilístico baseado na estabilidade
    filter(x -> rand() < orb.stability, data)
end

function detect_orb(rf::Float64, mesh::Float64)
    lambda = (rf / 1e9) * mesh
    if lambda > 0.618
        return Orb(lambda, rf)
    end
    return nothing
end
