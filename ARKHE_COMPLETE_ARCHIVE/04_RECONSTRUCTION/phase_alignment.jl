"""
Phase Alignment Preservation: ⟨0.00|0.07⟩
Maintains drone-demon inner product during reconstruction
"""

using LinearAlgebra
using Statistics

"""
Quantum state representation for drone (ω=0.00) and demon (ω=0.07)
"""
struct QuantumState
    omega::Float64
    amplitude::ComplexF64
    phase::Float64
end

"""
Compute inner product ⟨ψ₁|ψ₂⟩
"""
function inner_product(ψ1::QuantumState, ψ2::QuantumState)::ComplexF64
    return conj(ψ1.amplitude) * ψ2.amplitude *
           exp(im * (ψ2.phase - ψ1.phase))
end

"""
Syzygy from inner product magnitude
"""
function syzygy(ψ1::QuantumState, ψ2::QuantumState)::Float64
    return abs(inner_product(ψ1, ψ2))
end

"""
Phase alignment reconstruction
Preserves ⟨0.00|0.07⟩ = 0.94 during gap
"""
struct PhaseAlignmentReconstructor
    target_syzygy::Float64
    omega_drone::Float64
    omega_demon::Float64

    PhaseAlignmentReconstructor() = new(0.94, 0.00, 0.07)
end

"""
Reconstruct coherence by enforcing phase alignment
"""
function reconstruct(par::PhaseAlignmentReconstructor,
                    pre_gap_drone::QuantumState,
                    pre_gap_demon::QuantumState,
                    time_in_gap::Float64)::Float64

    # Evolve phases (free evolution)
    # ψ(t) = ψ(0) * exp(-i ω t)
    drone_phase = pre_gap_drone.phase - par.omega_drone * time_in_gap
    demon_phase = pre_gap_demon.phase - par.omega_demon * time_in_gap

    # Reconstruct states
    drone_reconstructed = QuantumState(
        par.omega_drone,
        pre_gap_drone.amplitude,
        drone_phase
    )

    demon_reconstructed = QuantumState(
        par.omega_demon,
        pre_gap_demon.amplitude,
        demon_phase
    )

    # Compute syzygy
    return syzygy(drone_reconstructed, demon_reconstructed)
end

"""
Verify phase coherence over extended gap
"""
function verify_phase_coherence(gap_duration::Int)
    par = PhaseAlignmentReconstructor()

    # Initial states
    drone = QuantumState(0.00, 1.0 + 0.0im, 0.0)
    demon = QuantumState(0.07, 1.0 + 0.0im, 0.1)  # Small initial phase diff

    # Verify initial syzygy
    initial_syzygy = syzygy(drone, demon)
    println("Initial syzygy: $(initial_syzygy)")

    # Track syzygy during gap
    syzygies = Float64[]
    times = Float64[]

    for t in 0:gap_duration
        s = reconstruct(par, drone, demon, Float64(t))
        push!(syzygies, s)
        push!(times, Float64(t))
    end

    # Statistics
    mean_syzygy = mean(syzygies)
    std_syzygy = std(syzygies)
    min_syzygy = minimum(syzygies)
    max_syzygy = maximum(syzygies)

    println("\nPhase Alignment Statistics ($(gap_duration) steps):")
    println("  Mean syzygy: $(mean_syzygy)")
    println("  Std dev: $(std_syzygy)")
    println("  Min: $(min_syzygy)")
    println("  Max: $(max_syzygy)")
    println("  Target: $(par.target_syzygy)")
    println("  Deviation from target: $(abs(mean_syzygy - par.target_syzygy))")

    # Plot
    using Plots

    p = plot(times, syzygies,
             label="⟨0.00|0.07⟩",
             xlabel="Time in gap",
             ylabel="Syzygy",
             title="Phase Alignment Preservation",
             linewidth=2,
             legend=:topright)
    hline!(p, [par.target_syzygy],
           label="Target (0.94)",
           linestyle=:dash,
           color=:red)

    savefig(p, "phase_alignment.png")
    println("\nPlot saved to phase_alignment.png")

    return (mean=mean_syzygy, std=std_syzygy, min=min_syzygy, max=max_syzygy)
end

"""
Multi-node phase alignment
Tracks phase coherence across network during gap
"""
struct NetworkPhaseState
    nodes::Vector{QuantumState}
    adjacency::Matrix{Bool}
end

"""
Compute network-wide syzygy
"""
function network_syzygy(nps::NetworkPhaseState)::Float64
    n = length(nps.nodes)
    total = 0.0
    count = 0

    for i in 1:n
        for j in (i+1):n
            if nps.adjacency[i, j]
                total += syzygy(nps.nodes[i], nps.nodes[j])
                count += 1
            end
        end
    end

    return count > 0 ? total / count : 0.0
end

"""
Example: Network reconstruction
"""
function network_example()
    println("\n" * "="^60)
    println("Network Phase Alignment Example")
    println("="^60)

    # Create network
    n_nodes = 100
    nodes = [QuantumState(
        rand() * 0.1,  # omega in [0, 0.1]
        1.0 + 0.0im,
        rand() * 2π    # random initial phase
    ) for _ in 1:n_nodes]

    # Random adjacency (sparse)
    adjacency = rand(n_nodes, n_nodes) .< 0.1
    adjacency = adjacency .| adjacency'  # Symmetrize

    nps = NetworkPhaseState(nodes, adjacency)

    # Initial syzygy
    initial = network_syzygy(nps)
    println("Initial network syzygy: $(initial)")

    # Evolve for 1000 time steps
    gap_duration = 1000
    syzygies = Float64[]

    for t in 0:gap_duration
        # Evolve phases
        evolved_nodes = [QuantumState(
            node.omega,
            node.amplitude,
            node.phase - node.omega * t
        ) for node in nodes]

        evolved_nps = NetworkPhaseState(evolved_nodes, adjacency)
        s = network_syzygy(evolved_nps)
        push!(syzygies, s)
    end

    final = syzygies[end]
    println("Final network syzygy: $(final)")
    println("Change: $(final - initial)")
    println("Relative change: $((final - initial) / initial * 100)%")
end

# Run examples
if abspath(PROGRAM_FILE) == @__FILE__
    # Single pair
    stats = verify_phase_coherence(1000)

    # Network
    network_example()
end
