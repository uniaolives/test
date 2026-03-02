#!/usr/bin/env bash
# GEOMETRIC BASH DEMO - Trajectory through state space

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source "$DIR/geometric-bash.sh"

echo "🚀 Starting Geometric Bash Demo"
echo "-------------------------------"

# 1. Initialize 4D state space
declare -A MY_STATE
geometric::init MY_STATE 4
STATE_SPACE=("${MY_STATE[@]}") # Sync with global for basin_trap demo

# 2. Define attractor (target state)
declare -A TARGET=(
    ["basis_0"]="0.9"
    ["basis_1"]="0.9"
    ["basis_2"]="0.9"
    ["basis_3"]="0.9"
)

echo "Target Attractor defined."

# 3. Move toward attractor
echo "Executing gradient descent toward attractor..."
geometric::gradient_descent MY_STATE TARGET 0.2 5 0.01

# 4. Check if in attractor basin
if geometric::in_basin MY_STATE TARGET 0.5; then
    echo "✅ Successfully converged to attractor basin (radius 0.5)"
else
    echo "❌ Failed to reach attractor basin"
fi

# 5. Basin Trap Demo
echo ""
echo "Basin Trap Demo:"
basin_trap TARGET 0.5 "echo 'Executing critical command within safe basin...'" || echo "Critical command skipped (outside basin)"

# 6. Integrated Information (Φ) demo
echo ""
echo "Coherence Metrics (Φ):"
partition_a=(0.9 0.8)
partition_b=(0.85 0.75)
phi=$(geometric::phi partition_a partition_b)
echo "System Coherence: Φ=$phi"

echo ""
echo "Demo Complete. Trajectory finalized."
