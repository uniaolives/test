# modules/julia/src/temporal_geodesics.jl

module TemporalGeodesics

using DifferentialEquations
using LinearAlgebra

export calculate_geodesic, synchronicity_field

"""
    synchronicity_field(u, p, t)

Define o campo de fluxo de sincronicidade na AMAS.
u: coordenadas (x, y, z)
p: parâmetros do Totem e anomalia geofísica
"""
function synchronicity_field(du, u, p, t)
    x, y, z = u
    B_amas, phi, totem_resonance = p

    # Dinâmica inspirada no atrator de Aizawa mas acoplada à magnetometria
    du[1] = totem_resonance * (z - 0.7) * x - 3.5 * y
    du[2] = 3.5 * x + (z - 0.7) * y
    du[3] = 0.6 + 0.95 * z - (z^3)/3.0 - (x^2 + y^2) * (1.0 + 0.25 * z) + B_amas * z * (x^3)
end

"""
    calculate_geodesic(start_coord, target_coord, B_amas, totem_resonance)

Calcula a trajetória de mínima resistência (geodésica) entre dois pontos temporais.
"""
function calculate_geodesic(start_u, B_amas, totem_resonance; t_span=(0.0, 10.0))
    phi = 1.618033988749895
    p = [B_amas, phi, totem_resonance]

    prob = ODEProblem(synchronicity_field, start_u, t_span, p)
    sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8)

    return sol
end

end # module
